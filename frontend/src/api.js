import axios from 'axios';

export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

const client = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Accept': 'application/json' },
});

// --- Health ---------------------------------------------------------------
export const getHealth = async () => {
  const { data } = await client.get('/health');
  return data;
};

// --- Users ----------------------------------------------------------------
export const createUser = async (payload) => {
  const { data } = await client.post('/users', payload, {
    headers: { 'Content-Type': 'application/json' },
  });
  return data;
};

export const getUser = async (userId) => {
  const { data } = await client.get(`/users/${userId}`);
  return data;
};

// --- Triage ---------------------------------------------------------------
export const submitTriage = async ({ patientId, audio, image, manualTranscript }) => {
  const form = new FormData();
  form.append('patient_id', patientId);
  if (audio) form.append('audio', audio);
  if (image) form.append('image', image);
  // EC1 fallback: typed symptom text when microphone access was denied.
  if (!audio && manualTranscript) form.append('manual_transcript', manualTranscript);

  try {
    const { status, data } = await client.post('/api/v1/triage/submit', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return { status, data };
  } catch (err) {
    // Surface structured response status for caller to handle 503 / 415
    if (err.response) {
      return {
        status: err.response.status,
        data: err.response.data,
        _error: err,
      };
    }
    throw err;
  }
};

export const getTriageStatus = async (jobId) => {
  const { data } = await client.get(`/api/v1/triage/status/${jobId}`);
  return data;
};

export const getTriageQueue = async (options = {}) => {
  const { statuses = 'COMPLETED,FAILED', limit = 50 } = options;
  const { data } = await client.get('/api/v1/triage/queue', {
    params: { statuses, limit },
  });
  return data;
};

// --- Submission detail (compatibility with pre-v2 detail endpoint) --------
export const getSubmissionDetail = async (submissionId) => {
  // We reuse the status endpoint for detail; the clinician dashboard also
  // accesses final_summary via this same payload shape.
  return getTriageStatus(submissionId);
};

export const updateSubmissionStatus = async (submissionId, action, payload = {}) => {
  // action = 'review' | 'override' | 'escalate'
  // payload.clinician_notes = string (REQUIRED >= 10 chars for override / escalate)
  // payload.override_severity = 'none' | 'mild' | 'moderate' | 'severe' | 'unknown' (only for 'override')
  const { data } = await client.patch(
    `/api/v1/triage/review/${submissionId}`,
    {
      action,
      clinician_notes: payload.clinician_notes ?? null,
      override_severity: payload.override_severity ?? null,
    },
    {
      headers: { 'Content-Type': 'application/json' },
    },
  );
  return data;
};
