import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import './App.css';
import { createUser, submitTriage } from './api';
import usePolling from './hooks/usePolling';

function getProcessingMessage(status, attempt) {
  if (status === 'COMPLETED') return 'Analysis complete.';
  if (status === 'FAILED') return 'Processing failed.';
  const stages = [
    'Queued for analysis...',
    'Analyzing audio & images...',
    'Synthesizing clinical summary...',
    'Running safety guardrail audit...',
  ];
  const idx = Math.min(stages.length - 1, Math.floor(attempt / 5));
  return stages[idx];
}

function App() {
  const [step, setStep] = useState(1);
  const [patientInfo, setPatientInfo] = useState({ firstName: '', lastName: '', dob: '' });
  const [patientId, setPatientId] = useState(null);
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const audioChunksRef = useRef([]);
  // EC1 (microphone permission denied / unavailable) fallback state.
  const [micUnavailable, setMicUnavailable] = useState(false);
  const [manualTranscript, setManualTranscript] = useState('');

  // Submission & polling state
  const [submitError, setSubmitError] = useState(null);
  const [jobId, setJobId] = useState(null);
  const poll = usePolling(jobId, { intervalMs: 3000, timeoutMs: 10 * 60 * 1000 });

  // ------------------------------- lifecycle
  useEffect(() => {
    return () => {
      if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        try { mediaRecorder.stop(); } catch (_) { /* ignore */ }
        try {
          mediaRecorder.stream.getTracks().forEach((t) => t.stop());
        } catch (_) { /* ignore */ }
      }
      setImagePreview(null);
      // Blob URLs are cleaned up when user re-records; leaving them in session
      // avoids ERR_ABORTED while audio element still references them.
    };
  }, [mediaRecorder]);

  // ------------------------------- Helpers
  const goNext = useCallback((next) => {
    setSubmitError(null);
    setStep(next);
  }, []);

  const resetAll = useCallback(() => {
    poll.stop();
    setStep(1);
    setPatientInfo({ firstName: '', lastName: '', dob: '' });
    setPatientId(null);
    setAudioFile(null);
    setAudioUrl(null);
    setImageFile(null);
    setImagePreview(null);
    setSubmitError(null);
    setJobId(null);
    setMicUnavailable(false);
    setManualTranscript('');
  }, [poll]);

  const mimeType = useMemo(() => {
    if (typeof MediaRecorder === 'undefined') return null;
    return MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
      ? 'audio/webm;codecs=opus'
      : 'audio/webm';
  }, []);

  const extension = mimeType ? mimeType.split('/')[1].split(';')[0] : 'webm';

  // ------------------------------- Step 1: Patient info
  const handlePatientSubmit = async (e) => {
    e.preventDefault();
    setSubmitError(null);
    try {
      const created = await createUser({
        first_name: patientInfo.firstName.trim(),
        last_name: patientInfo.lastName.trim(),
        dob: patientInfo.dob,
        role: 'patient',
      });
      setPatientId(created.id);
      goNext(2);
    } catch (err) {
      console.error('[TriageForm] createUser error:', err);
      setSubmitError(
        (err.response?.data?.detail) ||
        'Unable to create patient record. Please verify info and try again.'
      );
    }
  };

  // ------------------------------- Step 2: Audio recording
  const startRecording = async () => {
    if (audioUrl) setAudioUrl(null);
    if (audioFile) setAudioFile(null);
    setSubmitError(null);
    setMicUnavailable(false);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!mimeType) throw new Error('MediaRecorder is not supported in this browser.');
      const recorder = new MediaRecorder(stream, { mimeType });
      setMediaRecorder(recorder);
      audioChunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) audioChunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(audioChunksRef.current, { type: mimeType });
        const file = new File([blob], `recording.${extension}`, { type: mimeType });
        const url = URL.createObjectURL(file);
        setAudioFile(file);
        setAudioUrl(url);
        try {
          stream.getTracks().forEach((t) => t.stop());
        } catch (_) { /* ignore */ }
      };
      recorder.start();
      setIsRecording(true);
    } catch (err) {
      console.error('[TriageForm] startRecording error:', err);
      // EC1: offer a typed-text fallback rather than just an error banner,
      // so the patient isn't blocked from submitting when the browser or
      // OS denies microphone access.
      setMicUnavailable(true);
      setSubmitError(
        'Microphone access was denied or is unavailable. You can type your symptoms below instead.'
      );
    }
  };

  const stopRecording = () => {
    if (!mediaRecorder) return;
    if (mediaRecorder.state === 'inactive') return;
    try { mediaRecorder.stop(); } catch (_) { /* ignore */ }
    setIsRecording(false);
  };

  const reRecord = () => {
    try { if (audioUrl) setAudioUrl(null); } catch (_) {}
    setAudioFile(null);
  };

  // ------------------------------- Step 3: Image upload
  const handleImageChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (imagePreview) URL.revokeObjectURL(imagePreview);
    setImageFile(file);
    setImagePreview(URL.createObjectURL(file));
  };

  const removeImage = () => {
    if (imagePreview) {
      try { URL.revokeObjectURL(imagePreview); } catch (_) { /* ignore */ }
    }
    setImageFile(null);
    setImagePreview(null);
  };

  // ------------------------------- Step 4: Submit + poll
  const submitDisabled = !audioFile && !imageFile && !manualTranscript.trim();

  const handleSubmitTriage = async () => {
    setSubmitError(null);
    setJobId(null);

    try {
      const res = await submitTriage({
        patientId,
        audio: audioFile,
        image: imageFile,
        manualTranscript: manualTranscript.trim(),
      });

      if (res.status === 202) {
        setJobId(res.data.job_id);
        goNext(5);
        return;
      }

      if (res.status === 503) {
        setSubmitError(
          res.data?.detail ||
          'Systems currently under heavy load. Please try again in 5 minutes.'
        );
        return;
      }

      if (res.status === 415) {
        setSubmitError(
          `One of the uploaded files is not supported. (${res.data?.detail || 'Unsupported Media Type'})`
        );
        return;
      }

      // 4xx / 5xx generic
      setSubmitError(
        res.data?.detail ||
        `Submission failed (HTTP ${res.status}). Please try again.`
      );
    } catch (err) {
      console.error('[TriageForm] submitTriage error:', err);
      setSubmitError('Network error. Check that the backend server is running and try again.');
    }
  };

  // ------------------------------- Render
  const pollMsg = getProcessingMessage(poll.status, poll.attempt);

  return (
    <div className="app-container">
      <h1>Patient Triage</h1>

      {submitError && (
        <div className="banner banner-error" role="alert">
          {submitError}
        </div>
      )}

      {step === 1 && (
        <section className="step-container">
          <h2>Enter Patient Information</h2>
          <form onSubmit={handlePatientSubmit}>
            <label className="visually-hidden" htmlFor="firstName">First Name</label>
            <input
              id="firstName"
              type="text"
              placeholder="First Name"
              value={patientInfo.firstName}
              onChange={(e) => setPatientInfo({ ...patientInfo, firstName: e.target.value })}
              required
            />
            <label className="visually-hidden" htmlFor="lastName">Last Name</label>
            <input
              id="lastName"
              type="text"
              placeholder="Last Name"
              value={patientInfo.lastName}
              onChange={(e) => setPatientInfo({ ...patientInfo, lastName: e.target.value })}
              required
            />
            <label className="visually-hidden" htmlFor="dob">Date of Birth</label>
            <input
              id="dob"
              type="date"
              value={patientInfo.dob}
              onChange={(e) => setPatientInfo({ ...patientInfo, dob: e.target.value })}
              required
            />
            <button type="submit">Next</button>
          </form>
        </section>
      )}

      {step === 2 && (
        <section className="step-container">
          <h2>Record Your Symptoms</h2>
          <p className="muted">
            Please describe how you are feeling, how long you&apos;ve felt this way,
            and any pain or symptoms you are experiencing.
          </p>

          {!isRecording && !audioFile && (
            <button onClick={startRecording} aria-label="Start recording">Start Recording</button>
          )}

          {isRecording && (
            <div>
              <p className="recording-indicator" aria-live="polite">Recording…</p>
              <button onClick={stopRecording} aria-label="Stop recording">Stop Recording</button>
            </div>
          )}

          {audioFile && (
            <div>
              <audio controls src={audioUrl} aria-label="Recording preview" />
              <div className="button-group">
                <button onClick={reRecord}>Re-record</button>
                <button onClick={() => goNext(3)}>Next</button>
              </div>
            </div>
          )}

          {micUnavailable && !audioFile && (
            <div className="manual-transcript-fallback">
              <label htmlFor="manualTranscript">
                Type your symptoms instead
              </label>
              <textarea
                id="manualTranscript"
                rows={5}
                placeholder="Describe how you are feeling, how long you've felt this way, and any pain or symptoms..."
                value={manualTranscript}
                onChange={(e) => setManualTranscript(e.target.value)}
              />
              <div className="button-group">
                <button onClick={() => { setMicUnavailable(false); setSubmitError(null); }}>
                  Try Microphone Again
                </button>
                <button onClick={() => goNext(3)} disabled={!manualTranscript.trim()}>
                  Next
                </button>
              </div>
            </div>
          )}
        </section>
      )}

      {step === 3 && (
        <section className="step-container">
          <h2>Upload an Image (Optional)</h2>
          <p className="muted">
            Upload a photo of any visible physical symptom (e.g. rash, swelling).
            You may skip if symptoms are internal only.
          </p>

          {!imageFile && (
            <label className="file-picker">
              <input
                type="file"
                accept="image/jpeg,image/png,image/webp,image/bmp,image/tiff"
                onChange={handleImageChange}
              />
            </label>
          )}

          {imageFile && (
            <div>
              <img src={imagePreview} alt="Uploaded symptom" className="preview-img" />
              <div className="button-group">
                <button onClick={removeImage}>Change Image</button>
                <button onClick={() => goNext(4)}>Next</button>
              </div>
            </div>
          )}

          {!imageFile && (
            <button className="secondary" onClick={() => goNext(4)}>Skip</button>
          )}
        </section>
      )}

      {step === 4 && (
        <section className="step-container">
          <h2>Confirm Submission</h2>
          <ul className="confirm-list">
            <li>
              Audio: {audioFile
                ? `${audioFile.name} (${Math.round(audioFile.size/1024)} KB)`
                : manualTranscript.trim()
                  ? `Typed symptoms: "${manualTranscript.trim().slice(0, 60)}${manualTranscript.trim().length > 60 ? '…' : ''}"`
                  : 'Not provided'}
            </li>
            <li>Image: {imageFile ? `${imageFile.name} (${Math.round(imageFile.size/1024)} KB)` : 'Not provided'}</li>
          </ul>
          <button
            onClick={handleSubmitTriage}
            disabled={submitDisabled || poll.loading}
          >
            Submit Triage
          </button>
          {submitDisabled && (
            <p className="muted small">At least one of audio, typed symptoms, or an image is required.</p>
          )}
        </section>
      )}

      {step === 5 && (
        <section className="step-container">
          {poll.status === 'COMPLETED' && (
            <>
              <h2>Submission Successful</h2>
              <p>
                Your triage information has been processed. A clinician will review it shortly.
              </p>
              {poll.result?.error && (
                <div className="banner banner-warn">
                  Processing note: {poll.result.error}
                </div>
              )}
              <button onClick={resetAll}>Start New Triage</button>
            </>
          )}

          {poll.status === 'FAILED' && (
            <>
              <h2>Processing Issue</h2>
              <p>{poll.result?.error || poll.error || 'Unable to complete processing.'}</p>
              <p className="muted">Your intake was received and a clinician can still review it.</p>
              <div className="button-group">
                <button onClick={poll.restart}>Retry Analysis</button>
                <button className="secondary" onClick={resetAll}>Start New Triage</button>
              </div>
            </>
          )}

          {poll.status === 'TIMEOUT' && (
            <>
              <h2>Still Processing</h2>
              <p>{poll.error || 'Processing is taking longer than expected.'}</p>
              <div className="button-group">
                <button onClick={poll.restart}>Resume Waiting</button>
                <button className="secondary" onClick={resetAll}>Start New Triage</button>
              </div>
            </>
          )}

          {(poll.status === 'POLLING' || poll.status === 'IDLE') && (
            <>
              <h2>Analyzing your submission…</h2>
              <div className="spinner" aria-hidden="true" />
              <p className="muted" aria-live="polite">{pollMsg}</p>
              <p className="small muted">Attempt {poll.attempt || 0} · Job ID: {jobId}</p>
            </>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
