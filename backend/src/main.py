import os
import sys
import uuid
from datetime import date

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

# Make `routers/` importable from this file even if run as module __main__
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from app.database import Base, SessionLocal, engine
from app.models import Clinician, User
from app.schemas import UserCreate, UserResponse

# Ensure all models are loaded before Base.metadata.create_all
_ = Clinician

from routers.triage import router as triage_router  # noqa: E402  (after sys.path setup)

load_dotenv()

Base.metadata.create_all(bind=engine)

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(
    title="Intelligent Multimodal Patient Triage System",
    description=(
        "Production triage pipeline: FastAPI gateway + Celery GPU workers. "
        "Local Whisper audio, VLM vision, Instructor-constrained LLM synthesis."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.include_router(triage_router)


# ---------------------- Dependency ----------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------- Health ----------------------
@app.get("/health", tags=["ops"])
def health():
    return {"status": "ok", "service": "triage-gateway"}


# ---------------------- Users (Patient registration) ----------------------
@app.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED, tags=["users"])
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if payload.email:
        existing = db.query(User).filter(User.email == payload.email).first()
        if existing:
            raise HTTPException(status_code=409, detail="A user with this email already exists")

    user = User(
        id=uuid.uuid4(),
        first_name=payload.first_name.strip(),
        last_name=payload.last_name.strip(),
        dob=payload.dob,
        email=payload.email.strip().lower() if payload.email else None,
        role=(payload.role or "patient").lower(),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/users/{user_id}", response_model=UserResponse, tags=["users"])
def get_user(user_id: str, db: Session = Depends(get_db)):
    try:
        u = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id")
    user = db.query(User).filter(User.id == u).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
