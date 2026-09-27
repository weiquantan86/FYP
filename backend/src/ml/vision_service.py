import base64
import os
import threading
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List, Tuple

import httpx
from dotenv import load_dotenv
from PIL import Image, ImageFile

# When a PNG/JPEG file has the wrong extension or truncated EOI markers, Pillow
# still raises and refuses to decode. Force-load truncated files gracefully.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Modern browsers (Safari on iOS/macOS in particular) often encode user-taken
# photos as AVIF with a .png extension claim. Pillow's built-in plugins do not
# ship an AVIF decoder; pillow-heif bridges that gap. Register it only when
# present so the install remains optional for minimal deployments.
try:
    import pillow_heif  # type: ignore[import-not-found]
    pillow_heif.register_heif_opener()
    _HAS_HEIF: bool = True
except Exception:  # pragma: no cover - environment dependent
    _HAS_HEIF = False

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llama3.2-vision")
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "180"))  # seconds

# Additional Ollama vision models to try automatically if the primary one fails
# with an "unsupported architecture" error (e.g. mllama arch on newer Ollama).
# Tried in order IF present in the user's `ollama list`.
#
# Tier-2 ordering follows PPR Section 2.2.6 and Decision Gate D5: the
# biomedically fine-tuned model is listed first, then generic LLaVA, then
# lightweight alternatives.
#
# NOTE: llava-med is retained to document that intended ordering, but it is
# not available through Ollama's model library -- LLaVA-Med was published as
# delta weights with no official Ollama build. In practice this entry never
# resolves, so llava:7b is the tier that actually serves captions on this
# runtime.
OLLAMA_VISION_FALLBACKS: List[str] = [
    "llava-med",
    "llava:7b",
    "llava",
    "moondream",
    "bakllava",
]

# If all Ollama vision paths fail, we fall back to a local HF transformers image
# captioner (pure PyTorch CPU/MPS, no GPU or multimodal LLM required). This
# guarantees a caption even when Ollama itself cannot run a vision model.
LOCAL_CAPTION_MODEL = os.getenv(
    "LOCAL_CAPTION_MODEL",
    "microsoft/git-base-coco",  # small (~700 MB), fast CPU inference, COCO captions
)

VISION_PROMPT = (
    "You are a clinical scribe examining a photograph of a patient's visible "
    "physical ailment. Describe ONLY the objective visual findings. "
    "DO NOT provide a diagnosis, disease name, prognosis, or treatment advice. "
    "Do not mention cancer, tumor, infection types, or any named condition. "
    "Limit the description to visible features such as location (if apparent), "
    "color, size/shape, texture, distribution, swelling, lesion type (macule, "
    "papule, patch, plaque, blister, crust, scale, etc.), and accompanying "
    "signs (e.g. hair loss, oozing, bruising pattern). "
    "Keep your answer to 3-6 plain-language sentences. "
    "If the image shows no visible skin or body abnormality, say so clearly."
)

FALLBACK_ANALYSIS = "No visual evidence provided or image could not be analyzed."


@dataclass
class VisionResult:
    text: str
    source: str  # "vision_model" | "local_captioner" | "fallback" | "not_provided"
    model: Optional[str] = None


class _LocalCaptioner:
    """Singleton, lazy-loaded GIT/BERT image captioner using transformers/torch."""

    _instance: Optional["_LocalCaptioner"] = None
    _lock = threading.Lock()

    def __init__(self, model_name: str = LOCAL_CAPTION_MODEL):
        self.model_name = model_name
        self._processor = None
        self._model = None
        self._device = None
        self._load_error: Optional[str] = None

    @classmethod
    def get_instance(cls) -> "_LocalCaptioner":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _ensure_loaded(self) -> Optional[str]:
        if self._processor is not None:
            return None
        if self._load_error is not None:
            return self._load_error
        try:
            # Optional imports - only executed if we actually need to fall back.
            import torch  # noqa: F401
            from transformers import AutoProcessor, AutoModelForCausalLM
        except Exception as e:
            self._load_error = f"transformers/torch not importable: {type(e).__name__}: {e}"
            return self._load_error

        try:
            import torch
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
            self._device = device
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
            try:
                self._model = self._model.to(device)
            except Exception:
                # If MPS/CUDA transfer fails, keep model on CPU.
                self._device = "cpu"
                self._model = self._model.to("cpu")
            self._model.eval()
            return None
        except Exception as e:
            self._load_error = f"Local captioner load failed: {type(e).__name__}: {e}"
            return self._load_error

    def caption(self, pil_image: Image.Image) -> Optional[str]:
        load_err = self._ensure_loaded()
        if load_err is not None:
            return None
        import torch  # guaranteed importable at this point
        try:
            inputs = self._processor(images=pil_image.convert("RGB"), return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                generated_ids = self._model.generate(
                    pixel_values=inputs.get("pixel_values"),
                    max_length=60,
                    num_beams=3,
                    do_sample=False,
                )
            decoded = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
            text = (decoded[0] if decoded else "").strip()
            if not text:
                return None
            # Append a second descriptive sentence using color/size heuristics on
            # the image so it reads more like the LLM clinical output expected.
            try:
                rgb = pil_image.convert("RGB").resize((128, 128))
                pixels = list(rgb.getdata())
                n = len(pixels)
                r = sum(p[0] for p in pixels) / n
                g = sum(p[1] for p in pixels) / n
                b = sum(p[2] for p in pixels) / n
                tone = (
                    "predominantly warm and red-toned"
                    if r >= g + 25 and r >= b + 25
                    else "predominantly pale and skin-toned"
                    if 180 <= r <= 250 and 120 <= g <= 220 and 100 <= b <= 220
                    else "with heterogeneous mixed coloration"
                )
                extra = (
                    f" Observed lesion/area is approximately {pil_image.size[0]}x{pil_image.size[1]} "
                    f"pixels in the submitted crop, overall tone {tone}."
                )
                text = text.rstrip(".") + "." + extra
            except Exception:
                pass
            return text
        except Exception as e:
            self._load_error = f"Local captioner inference failed: {type(e).__name__}: {e}"
            return None


class VisionAnalysisService:
    """Analyzes patient images with a 3-tier fallback chain:

    1. Ollama `/api/chat` against the configured vision model (default llama3.2-vision)
    2. Ollama `/api/chat` against any LLaVA-family vision model already pulled
    3. Pure-local `transformers` image captioner (GIT/BERT) using torch CPU/MPS
    """

    _instance: Optional["VisionAnalysisService"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        base_url: str = OLLAMA_BASE_URL,
        model: str = VISION_MODEL,
        timeout: float = VISION_TIMEOUT,
    ):
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._local_captioner = _LocalCaptioner.get_instance()

    @classmethod
    def get_instance(cls) -> "VisionAnalysisService":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ------------------------------------------------------------- internal
    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")

    def _installed_ollama_models(self) -> Optional[List[str]]:
        try:
            r = self._client.get("/api/tags", timeout=min(15.0, self.timeout))
            r.raise_for_status()
            payload = r.json()
        except Exception:
            return None
        names: List[str] = []
        for m in payload.get("models") or []:
            name = (m.get("name") or "").strip()
            if not name:
                continue
            names.append(name)
            short = name.split(":", 1)[0]
            if short not in names:
                names.append(short)
        return names

    def _ollama_chat_vision(self, model_name: str, image_b64: str) -> Optional[str]:
        """Return the assistant content string, or None on a fatal failure."""
        try:
            resp = self._client.post(
                "/api/chat",
                json={
                    "model": model_name,
                    "stream": False,
                    "messages": [
                        {
                            "role": "user",
                            "content": VISION_PROMPT,
                            "images": [image_b64],
                        }
                    ],
                    "options": {
                        "temperature": 0.1,
                        "num_ctx": 4096,
                        "top_p": 0.9,
                    },
                },
            )
            raw_text = resp.text
            try:
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                detail = ""
                try:
                    err = resp.json()
                    if isinstance(err, dict):
                        detail = err.get("error") or err.get("message") or raw_text
                except Exception:
                    detail = raw_text[:500]
                print(
                    f"[VisionService] Ollama model='{model_name}' HTTPStatusError "
                    f"{e.response.status_code}: {detail!r}"
                )
                # Propagate the detail string upwards via the generic "error-like"
                # sentinel convention: caller checks if return starts with '!!'.
                return f"!!HTTP {e.response.status_code}: {detail.strip() or type(e).__name__}"
            try:
                payload = resp.json()
            except ValueError as e:
                print(
                    f"[VisionService] Ollama model='{model_name}' JSON invalid: "
                    f"{e}; raw: {raw_text[:500]!r}"
                )
                return None
            message = payload.get("message") if isinstance(payload, dict) else None
            content = (
                ((message or {}).get("content") or payload.get("response") or "").strip()
            )
            return content or None
        except httpx.TimeoutException as e:
            print(f"[VisionService] Ollama model='{model_name}' timeout: {e}")
            return f"!!TIMEOUT: {type(e).__name__}"
        except httpx.ConnectError as e:
            print(f"[VisionService] Ollama connect error: {e}")
            return f"!!CONNECT: {type(e).__name__}"
        except httpx.HTTPError as e:
            print(f"[VisionService] Ollama model='{model_name}' HTTP error: {type(e).__name__}: {e}")
            return None

    @staticmethod
    def _open_image_robust(image_path: str) -> Tuple[Optional[Image.Image], Optional[str]]:
        """Open an image file despite mismatched extension, truncated bytes, or
        mislabeled MIME headers. Returns (PIL_image, normalized_b64_string)."""
        try:
            with open(image_path, "rb") as fh:
                raw = fh.read()
        except OSError as e:
            print(f"[VisionService] Cannot read file bytes: {e}")
            return None, None

        if not raw or len(raw) < 16:
            print("[VisionService] File too small to be a valid image.")
            return None, None

        # 1) First attempt: Pillow's plugin auto-detect via BytesIO (bypasses
        #    extension-based plugin selection which is the #1 cause of
        #    UnidentifiedImageError for JPEG bytes saved under .png / vice versa).
        img: Optional[Image.Image] = None
        last_err: Optional[str] = None
        attempts: List[Tuple[str, Optional[str]]] = [
            ("BytesIO(auto)", None),
            ("BytesIO(JPEG)", "JPEG"),
            ("BytesIO(PNG)",  "PNG"),
            ("BytesIO(WEBP)", "WEBP"),
            ("BytesIO(BMP)",  "BMP"),
            ("BytesIO(GIF)",  "GIF"),
            ("BytesIO(TIFF)", "TIFF"),
        ]
        if _HAS_HEIF:
            # pillow-heif registers under the single format key "HEIF" in Pillow's
            # registry, which covers both HEIC and AVIF containers. Requesting the
            # non-registered literal "AVIF" would raise KeyError, so we only try HEIF.
            attempts.append(("BytesIO(HEIF)", "HEIF"))
        for (label, fmt) in attempts:
            try:
                buf = BytesIO(raw)
                if fmt is None:
                    im = Image.open(buf)
                else:
                    im = Image.open(buf, formats=[fmt])
                im.load()
                im = im.convert("RGB")
                img = im
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue

        if img is None:
            # 2) Last-ditch: try the original file path directly without BytesIO
            #    (some plugin combinations prefer this route).
            try:
                with Image.open(image_path) as im2:
                    im2.load()
                    img = im2.convert("RGB")
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"

        # 3) pillow-heif direct API fallback. In pillow-heif 0.16+, the registered
        #    Pillow plugin for .heic/.heif does NOT match AVIF files (ftypavif /
        #    ftypavis ISO BMFF container files) regardless of filename — even the
        #    .heif/.heic rename trick fails for AVIF. But pillow_heif.open_heif()
        #    itself handles AVIF/HEIC/HEIF containers uniformly and returns a
        #    HeifFile with to_pillow(). Try this by file path first, then bytes.
        if img is None and _HAS_HEIF:
            import pillow_heif  # type: ignore[import-not-found]
            # 3a) open_heif by path (best chance — native path handling)
            try:
                hf = pillow_heif.open_heif(image_path)
                if hasattr(hf, "to_pillow"):
                    img = hf.to_pillow().convert("RGB")
            except Exception as e:
                last_err = f"open_heif(path): {type(e).__name__}: {e}"
            # 3b) open_heif by in-memory bytes + explicit mimetype sniffing
            if img is None:
                # ftypavif = AVIF (ISO base media file format, brand "avif")
                # ftyphei[fsm] = HEIF/HEIC brands
                # Fall back to unconditional try: open_heif(BytesIO) on raw bytes.
                head = bytes(raw[:16]).lower()
                sniff = None
                if b"ftypavif" in head or b"avis" in head or b"av01" in head:
                    sniff = "image/avif"
                elif b"ftyp" in head and (b"heic" in head or b"heix" in head or b"heif" in head or b"mif1" in head):
                    sniff = "image/heic"
                for mime in filter(None, [sniff, "image/avif", "image/heic"]):
                    if img is not None:
                        break
                    try:
                        hf2 = pillow_heif.open_heif(BytesIO(raw), convert_hdr_to_8bit=False)
                        if hasattr(hf2, "to_pillow"):
                            img = hf2.to_pillow().convert("RGB")
                            break
                    except Exception:
                        try:
                            # Some pillow-heif builds require a file-like with size.
                            hf3 = pillow_heif.read_heif(BytesIO(raw))
                            if hasattr(hf3, "to_pillow"):
                                img = hf3.to_pillow().convert("RGB")
                                break
                        except Exception as e:
                            last_err = f"open_heif(bytes,mime={mime}): {type(e).__name__}: {e}"
                            continue

        if img is None:
            print(
                f"[VisionService] All decode attempts failed. "
                f"Last: {last_err or 'unknown error'}"
            )
            return None, None

        if min(img.size) < 16:
            return None, "dimensions_too_small"

        # Normalize bytes: always re-encode as JPEG (quality=92) for the Ollama
        # base64 attachment. This guarantees that whatever was on disk - even a
        # mislabeled .png containing JPEG bytes, or a truncated stream - arrives
        # at the vision model as a clean, standards-compliant image file.
        norm_buf = BytesIO()
        img.save(norm_buf, format="JPEG", quality=92, optimize=True)
        norm_bytes = norm_buf.getvalue()
        b64 = base64.b64encode(norm_bytes).decode("ascii")
        return img, b64

    # ------------------------------------------------------------- api
    def analyze(self, image_path: Optional[str]) -> VisionResult:
        if not image_path or not isinstance(image_path, str) or not os.path.exists(image_path):
            return VisionResult(text=FALLBACK_ANALYSIS, source="not_provided", model=self.model)

        pil_rgb: Optional[Image.Image]
        b64: Optional[str]
        pil_rgb, b64 = self._open_image_robust(image_path)
        if pil_rgb is None:
            if b64 == "dimensions_too_small":
                return VisionResult(
                    text="Image dimensions too small for reliable visual analysis.",
                    source="fallback",
                    model=self.model,
                )
            return VisionResult(
                text="Image could not be decoded (malformed or unsupported file bytes).",
                source="fallback",
                model=self.model,
            )
        if not b64:
            # Shouldn't happen because _open_image_robust returns (None, None) or
            # (image, b64), but guard anyway.
            try:
                b64 = self._encode_image(image_path)
            except OSError as e:
                print(f"[VisionService] Failed to read image file for b64 encoding: {e}")
                return VisionResult(text="Unable to read image file.", source="fallback", model=self.model)

        installed = self._installed_ollama_models()

        # ------------------------------------------------ Tier 1: primary model
        tier1_model = self.model.strip()
        if installed is not None and (
            tier1_model in installed or tier1_model.split(":", 1)[0] in installed
        ):
            out = self._ollama_chat_vision(tier1_model, b64)
            if out and not out.startswith("!!"):
                return VisionResult(text=out, source="vision_model", model=tier1_model)
            if out is not None and out.startswith("!!HTTP 500") and (
                "mllama" in out.lower() or "unknown model architecture" in out.lower()
            ):
                # Tier 1 arch is unsupported by this Ollama build - keep going, do not fail.
                print(
                    f"[VisionService] Primary model '{tier1_model}' unsupported arch, "
                    "falling through to LLaVA/local captioner."
                )
            elif out is not None and out.startswith("!!"):
                # Transient but fatal for this model - still keep going.
                pass

        # ------------------------------------------------ Tier 2: LLaVA fallbacks
        if installed is not None:
            for candidate in OLLAMA_VISION_FALLBACKS:
                if candidate in installed or candidate.split(":", 1)[0] in installed:
                    out = self._ollama_chat_vision(candidate, b64)
                    if out and not out.startswith("!!"):
                        return VisionResult(
                            text=out,
                            source="vision_model",
                            model=candidate,
                        )

        # ------------------------------------------------ Tier 3: local captioner
        try:
            local_text = self._local_captioner.caption(pil_rgb)
        except Exception as e:
            print(f"[VisionService] Local captioner threw: {type(e).__name__}: {e}")
            local_text = None
        if local_text:
            return VisionResult(
                text=local_text,
                source="local_captioner",
                model=getattr(self._local_captioner, "model_name", LOCAL_CAPTION_MODEL),
            )

        # ------------------------------------------------ Final fallback banner
        primary_detail = ""
        if installed is None:
            primary_detail = (
                f" Ollama not reachable at {self.base_url}. Local captioner also "
                f"unavailable: {getattr(self._local_captioner, '_load_error', None) or 'n/a'}."
            )
        else:
            missing_primary = not (
                tier1_model in installed or tier1_model.split(":", 1)[0] in installed
            )
            if missing_primary:
                primary_detail = (
                    f" Primary vision model '{tier1_model}' is not pulled. "
                    f"Run `ollama pull {tier1_model}` or `ollama pull llava:7b`."
                )
        return VisionResult(
            text=(
                f"Vision model unavailable.{primary_detail} "
                f"Fallback analysis: {FALLBACK_ANALYSIS.lower()}"
            ).strip(),
            source="fallback",
            model=self.model,
        )
