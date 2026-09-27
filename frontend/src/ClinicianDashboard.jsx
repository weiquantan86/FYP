import { useCallback, useEffect, useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  API_BASE_URL,
  getTriageQueue,
  getTriageStatus,
  updateSubmissionStatus,
} from './api';
import './ClinicianDashboard.css';

function ageFromDob(dobIso) {
  if (!dobIso) return null;
  const dob = new Date(dobIso);
  const today = new Date();
  let a = today.getFullYear() - dob.getFullYear();
  const m = today.getMonth() - dob.getMonth();
  if (m < 0 || (m === 0 && today.getDate() < dob.getDate())) a -= 1;
  return a;
}

function uploadName(fullPath) {
  if (!fullPath) return '';
  return fullPath.includes('/') ? fullPath.split('/').pop() : fullPath;
}

function statusBadgeClass(status) {
  switch (status?.toUpperCase?.() ?? status) {
    case 'COMPLETED':
    case 'REVIEWED':
      return 'status-badge success';
    case 'FAILED':
    case 'GUARDRAIL_ABORTED':
      return 'status-badge warn';
    case 'ESCALATED':
      return 'status-badge failed';
    case 'PROCESSING':
      return 'status-badge processing';
    case 'PENDING':
      return 'status-badge pending';
    default:
      return 'status-badge';
  }
}

function ClinicianDashboard() {
  const navigate = useNavigate();
  const [queueResp, setQueueResp] = useState({ items: [], total: 0 });
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [loadError, setLoadError] = useState(null);
  // ---- HITL Override Severity form state (report L491-492: 10-char mandatory comment) ----
  const [overrideOpen, setOverrideOpen] = useState(false);
  const [overrideSeverity, setOverrideSeverity] = useState('severe');
  const [overrideNotes, setOverrideNotes] = useState('');
  const [overrideError, setOverrideError] = useState(null);
  // ---- HITL Escalate form state (report §3.4: forces severity=severe, same mandatory comment) ----
  const [escalateOpen, setEscalateOpen] = useState(false);
  const [escalateNotes, setEscalateNotes] = useState('');
  const [escalateError, setEscalateError] = useState(null);
  const [submittingAction, setSubmittingAction] = useState(null); // 'review' | 'override' | 'escalate'

  const loadQueue = useCallback(async () => {
    setLoadError(null);
    try {
      const data = await getTriageQueue({
        statuses: 'COMPLETED,FAILED,PROCESSING,PENDING,GUARDRAIL_ABORTED,REVIEWED,ESCALATED',
        limit: 100,
      });
      setQueueResp(data);
    } catch (e) {
      console.error('[ClinicianDashboard] loadQueue error:', e);
      setLoadError(e?.message || 'Unable to reach triage API.');
    }
  }, []);

  useEffect(() => {
    loadQueue();
  }, [loadQueue]);

  const selectItem = useCallback(async (queueItem) => {
    setSelected(queueItem);
    setDetail(null);
    try {
      const d = await getTriageStatus(queueItem.id);
      setDetail(d);
    } catch (e) {
      console.error('[ClinicianDashboard] selectItem error:', e);
    }
  }, []);

  const markReviewed = useCallback(async () => {
    if (!selected) return;
    setSubmittingAction('review');
    setOverrideError(null);
    try {
      const updated = await updateSubmissionStatus(selected.id, 'review', {});
      setDetail(updated);
      await loadQueue();
    } catch (e) {
      console.error('[ClinicianDashboard] markReviewed error:', e);
      setOverrideError(e?.response?.data?.detail || e?.message || 'Unable to record reviewed action.');
    } finally {
      setSubmittingAction(null);
    }
  }, [selected, loadQueue]);

  const submitOverride = useCallback(async () => {
    if (!selected) return;
    // Client-side 10-character mandatory comment validator. Matches the server-side
    // ReviewSubmissionRequest._mandatory_comment_if_overridden Pydantic check.
    const trimmed = (overrideNotes || '').trim();
    if (trimmed.length < 10) {
      setOverrideError(
        `Mandatory comment must be at least 10 characters. You have ${trimmed.length} character${trimmed.length === 1 ? '' : 's'}.`
      );
      return;
    }
    setOverrideError(null);
    setSubmittingAction('override');
    try {
      const updated = await updateSubmissionStatus(selected.id, 'override', {
        clinician_notes: trimmed,
        override_severity: overrideSeverity,
      });
      setDetail(updated);
      setOverrideOpen(false);
      setOverrideNotes('');
      await loadQueue();
    } catch (e) {
      console.error('[ClinicianDashboard] submitOverride error:', e);
      setOverrideError(e?.response?.data?.detail || e?.message || 'Unable to save override action.');
    } finally {
      setSubmittingAction(null);
    }
  }, [selected, overrideNotes, overrideSeverity, loadQueue]);

  const submitEscalate = useCallback(async () => {
    if (!selected) return;
    // Same client-side 10-character mandatory comment validator as override.
    // Severity is not selected here — the backend forces it to 'severe' for
    // action='escalate' (routers/triage.py), so there is nothing to pick.
    const trimmed = (escalateNotes || '').trim();
    if (trimmed.length < 10) {
      setEscalateError(
        `Mandatory comment must be at least 10 characters. You have ${trimmed.length} character${trimmed.length === 1 ? '' : 's'}.`
      );
      return;
    }
    setEscalateError(null);
    setSubmittingAction('escalate');
    try {
      const updated = await updateSubmissionStatus(selected.id, 'escalate', {
        clinician_notes: trimmed,
      });
      setDetail(updated);
      setEscalateOpen(false);
      setEscalateNotes('');
      await loadQueue();
    } catch (e) {
      console.error('[ClinicianDashboard] submitEscalate error:', e);
      setEscalateError(e?.response?.data?.detail || e?.message || 'Unable to escalate submission.');
    } finally {
      setSubmittingAction(null);
    }
  }, [selected, escalateNotes, loadQueue]);

  const summary = detail?.data ?? null;
  const isGuardrailFallback = !!summary?._fallback_reason || summary?.requires_manual_intake;

  function uploadName(fullPath) {
    if (!fullPath) return '';
    const p = String(fullPath).trim();
    if (!p) return '';
    if (p.includes('/')) return p.split('/').pop();
    if (p.includes('\\')) return p.split('\\').pop();
    return p;
  }

  const audioSrc = useMemo(() => {
    const path =
      detail?.audio_file_path ||
      selected?.audio_file_path ||
      summary?.audio_file_path ||
      null;
    const name = uploadName(path);
    if (!name) return null;
    return `${API_BASE_URL}/uploads/${encodeURIComponent(name)}`;
  }, [selected, detail, summary]);

  const imageSrc = useMemo(() => {
    const path =
      detail?.image_file_path ||
      selected?.image_file_path ||
      summary?.image_file_path ||
      null;
    const name = uploadName(path);
    if (!name) return null;
    return `${API_BASE_URL}/uploads/${encodeURIComponent(name)}`;
  }, [selected, detail, summary]);

  const rawTranscript =
    detail?.raw_transcript ??
    selected?.raw_transcript ??
    summary?.raw_transcript ??
    null;

  const rawVision =
    detail?.vision_analysis ??
    selected?.vision_analysis ??
    summary?.raw_vision_analysis ??
    null;

  const age = selected ? ageFromDob(selected.patient_dob) : null;

  // ---- HITL surfaced metadata (queue or detail source) ----
  const reviewedAt = detail?.clinician_reviewed_at ?? selected?.clinician_reviewed_at ?? null;
  const clinicianNotes = detail?.clinician_notes ?? selected?.clinician_notes ?? null;
  const overrideSeverityShown = detail?.clinician_override_severity ?? selected?.clinician_override_severity ?? null;

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <div>
          <h1>Clinician Dashboard</h1>
          <p className="muted small">{queueResp.total || 0} recent submissions</p>
        </div>
        <div className="dashboard-actions">
          <button className="secondary" onClick={loadQueue}>Refresh</button>
          <button className="secondary" onClick={() => navigate('/')}>Patient Portal</button>
        </div>
      </header>

      {loadError && (
        <div className="banner banner-error" role="alert">{loadError}</div>
      )}

      <div className="dashboard-content">
        <aside className="queue-panel">
          <h2>Triage Queue</h2>
          <div className="queue-list">
            {queueResp.items.length === 0 ? (
              <p className="muted">No recent submissions.</p>
            ) : (
              queueResp.items.map((sub) => (
                <div
                  key={sub.id}
                  className={`queue-item ${selected?.id === sub.id ? 'active' : ''}`}
                  onClick={() => selectItem(sub)}
                  role="button"
                  tabIndex={0}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') selectItem(sub); }}
                >
                  <div className="queue-item-info">
                    <h3>{sub.patient_name}</h3>
                    <p>Age: {ageFromDob(sub.patient_dob) ?? '—'} · Submitted: {new Date(sub.created_at).toLocaleString()}</p>
                    {sub.error_message && <p className="small muted">Note: {sub.error_message}</p>}
                    {sub.clinician_reviewed_at && (
                      <p className="small">
                        ✅ Reviewed {new Date(sub.clinician_reviewed_at).toLocaleString()}
                        {sub.clinician_override_severity && (
                          <> · Override: <strong>{sub.clinician_override_severity}</strong></>
                        )}
                      </p>
                    )}
                  </div>
                  <span className={statusBadgeClass(sub.status)}>{sub.status}</span>
                </div>
              ))
            )}
          </div>
        </aside>

        <section className="detail-panel">
          {!selected ? (
            <div className="no-selection">
              <p>Select a submission from the queue to view details.</p>
            </div>
          ) : (
            <article className="submission-detail">
              <header className="detail-header">
                <div>
                  <h2>Patient: {selected.patient_name}</h2>
                  <p>
                    DOB: {new Date(selected.patient_dob).toLocaleDateString()}
                    {age != null ? <> · Age: {age}</> : null}
                  </p>
                  {reviewedAt && (
                    <p className="small muted">
                      ✅ Clinician reviewed at {new Date(reviewedAt).toLocaleString()}
                      {overrideSeverityShown && (
                        <> · <strong>Override Severity: {overrideSeverityShown}</strong></>
                      )}
                    </p>
                  )}
                </div>
                <span className={statusBadgeClass(selected.status)}>{selected.status}</span>
              </header>

              {isGuardrailFallback && (
                <div className="banner banner-warn" role="note">
                  {summary._fallback_reason || 'Manual Intake Required'}
                  {summary._guardrail_error && <div className="small">{summary._guardrail_error}</div>}
                </div>
              )}
              {detail?.error && !isGuardrailFallback && (
                <div className="banner banner-warn" role="note">{detail.error}</div>
              )}

              <div className="detail-body">
                {/* ===== Left: Structured AI Summary ===== */}
                <div className="summary-section">
                  <h3>AI Clinical Summary</h3>

                  {!summary && <p className="muted">No structured summary yet (processing…).</p>}

                  {summary?.chief_complaint && (
                    <div className="summary-item">
                      <h4>Chief Complaint</h4>
                      <p>{summary.chief_complaint}</p>
                    </div>
                  )}
                  {summary?.history_of_present_illness && (
                    <div className="summary-item">
                      <h4>History of Present Illness</h4>
                      <p>{summary.history_of_present_illness}</p>
                    </div>
                  )}
                  {summary?.visual_assessment && (
                    <div className="summary-item">
                      <h4>Visual Assessment</h4>
                      <p>{summary.visual_assessment}</p>
                    </div>
                  )}

                  {Array.isArray(summary?.symptoms) && summary.symptoms.length > 0 && (
                    <div className="summary-item">
                      <h4>Structured Symptoms</h4>
                      <ul className="symptom-list">
                        {summary.symptoms.map((s, i) => (
                          <li key={i}>
                            <strong>{s.name}</strong>
                            {s.severity && s.severity !== 'unknown' && (
                              <span className={`badge severity-${s.severity}`}> {s.severity}</span>
                            )}
                            {s.body_location && <span className="muted small"> · {s.body_location}</span>}
                            {s.notes && <p className="small">{s.notes}</p>}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {summary?.reported_duration && (
                    summary.reported_duration.raw_text || summary.reported_duration.days
                  ) && (
                    <div className="summary-item">
                      <h4>Reported Duration</h4>
                      {summary.reported_duration.raw_text && <p>“{summary.reported_duration.raw_text}”</p>}
                      {summary.reported_duration.days != null && (
                        <p className="small muted">≈ {summary.reported_duration.days} day{summary.reported_duration.days === 1 ? '' : 's'}{summary.reported_duration.unit ? ` (${summary.reported_duration.unit})` : ''}</p>
                      )}
                    </div>
                  )}

                  {Array.isArray(summary?.patient_concerns) && summary.patient_concerns.length > 0 && (
                    <div className="summary-item">
                      <h4>Patient-Reported Concerns</h4>
                      <ul>
                        {summary.patient_concerns.map((c, i) => <li key={i}>{c}</li>)}
                      </ul>
                      <p className="small muted">These are verbatim patient worries, not clinical diagnoses.</p>
                    </div>
                  )}

                  {summary?.safety_flag && (
                    <div className="summary-item">
                      <h4>Safety Flag</h4>
                      <p className="small">{summary.safety_flag}</p>
                    </div>
                  )}

                  {/* Edge Case 3 fallback: raw data */}
                  {isGuardrailFallback && (
                    <div className="summary-item raw-fallback">
                      <h4>Raw Transcript</h4>
                      <p className="mono small">{summary.raw_transcript ?? '—'}</p>
                      <h4>Raw Vision Analysis</h4>
                      <p className="mono small">{summary.raw_vision_analysis ?? '—'}</p>
                    </div>
                  )}
                </div>

                {/* ===== Right: Raw Evidence ===== */}
                <div className="evidence-section">
                  <h3>Raw Evidence</h3>

                  {audioSrc ? (
                    <div className="evidence-item">
                      <h4>Audio Recording</h4>
                      <audio controls src={audioSrc} preload="metadata" />
                      <p className="small muted">{uploadName(
                        detail?.audio_file_path ||
                        selected?.audio_file_path ||
                        summary?.audio_file_path ||
                        ''
                      )}</p>
                    </div>
                  ) : (
                    <div className="evidence-item muted small"><h4>Audio Recording</h4><p>Not provided.</p></div>
                  )}

                  {imageSrc ? (
                    <div className="evidence-item">
                      <h4>Uploaded Image</h4>
                      <a href={imageSrc} target="_blank" rel="noreferrer">
                        <img src={imageSrc} alt="Uploaded symptom" className="evidence-img" loading="lazy" />
                      </a>
                      <p className="small muted">{uploadName(
                        detail?.image_file_path ||
                        selected?.image_file_path ||
                        summary?.image_file_path ||
                        ''
                      )}</p>
                    </div>
                  ) : (
                    <div className="evidence-item muted small"><h4>Uploaded Image</h4><p>Not provided.</p></div>
                  )}

                  <div className="evidence-item">
                    <h4>Transcript (Whisper)</h4>
                    <p className="small mono">{rawTranscript && rawTranscript.trim() ? rawTranscript : '—'}</p>
                  </div>
                  <div className="evidence-item">
                    <h4>Image Caption (VLM)</h4>
                    <p className="small mono">{rawVision && rawVision.trim() ? rawVision : '—'}</p>
                  </div>

                  {clinicianNotes && (
                    <div className="evidence-item">
                      <h4>Clinician Notes (HITL Audit)</h4>
                      <p className="small">{clinicianNotes}</p>
                    </div>
                  )}

                  {detail?.data?._stub && (
                    <p className="muted small">
                      Summary is a placeholder (LLM synthesis stage pending on this deployment).
                    </p>
                  )}
                </div>
              </div>

              <footer className="detail-footer">
                <button
                  onClick={markReviewed}
                  disabled={!!submittingAction}
                >
                  {submittingAction === 'review' ? 'Recording…' : 'Mark as Reviewed'}
                </button>
                <button
                  className="override"
                  onClick={() => {
                    setOverrideError(null);
                    setOverrideOpen((v) => !v);
                    setEscalateOpen(false);
                    setEscalateError(null);
                  }}
                  disabled={!!submittingAction}
                >
                  {overrideOpen ? 'Cancel Override' : 'Override Severity'}
                </button>
                <button
                  className="escalate"
                  onClick={() => {
                    setEscalateError(null);
                    setEscalateOpen((v) => !v);
                    setOverrideOpen(false);
                    setOverrideError(null);
                  }}
                  disabled={!!submittingAction}
                >
                  {escalateOpen ? 'Cancel Escalate' : 'Escalate'}
                </button>
              </footer>

              {/* ----- Mandatory-comment Override Severity inline form (report L491-492: >= 10 chars) ----- */}
              {overrideOpen && (
                <section className="override-form" aria-label="Override Severity form">
                  <h3>Override Severity (HITL mandatory comment)</h3>
                  {overrideError && (
                    <div className="banner banner-error" role="alert">{overrideError}</div>
                  )}
                  <p className="small muted">
                    Per clinical-audit protocol (report §3.6): any severity override requires a written
                    justification comment of <strong>at least 10 characters</strong>. The same minimum is
                    enforced server-side via Pydantic schema validation.
                  </p>
                  <div className="form-row">
                    <label htmlFor="override-severity-select">Set new severity:</label>
                    <select
                      id="override-severity-select"
                      value={overrideSeverity}
                      onChange={(e) => setOverrideSeverity(e.target.value)}
                    >
                      <option value="none">None (no impression)</option>
                      <option value="mild">Mild</option>
                      <option value="moderate">Moderate</option>
                      <option value="severe">Severe (consider ER)</option>
                      <option value="unknown">Unknown</option>
                    </select>
                  </div>
                  <div className="form-row">
                    <label htmlFor="override-notes">
                      Justification comment <span className="required">* (required, ≥10 characters)</span>:
                    </label>
                    <textarea
                      id="override-notes"
                      rows={4}
                      placeholder="e.g. Suspect cellulitis given warmth, tenderness and rapid progression history — downgrade visual impression from moderate impression to severe impression with oral antibiotics pending bloods."
                      value={overrideNotes}
                      onChange={(e) => {
                        setOverrideNotes(e.target.value);
                        if (overrideError) setOverrideError(null);
                      }}
                    />
                    <p className="small muted">
                      Character count: <strong>{(overrideNotes || '').length}</strong>
                      {' '}(after trim: {(overrideNotes || '').trim().length})
                      {(overrideNotes || '').trim().length < 10 ? ' — minimum 10 required.' : ' ✓ meets minimum.'}
                    </p>
                  </div>
                  <div className="form-actions">
                    <button
                      className="secondary"
                      onClick={() => {
                        setOverrideOpen(false);
                        setOverrideError(null);
                        setOverrideNotes('');
                      }}
                      disabled={!!submittingAction}
                    >
                      Cancel
                    </button>
                    <button
                      className="override"
                      onClick={submitOverride}
                      disabled={!!submittingAction || (overrideNotes || '').trim().length < 10}
                    >
                      {submittingAction === 'override'
                        ? 'Saving override…'
                        : 'Save Severity Override + Mark Reviewed'}
                    </button>
                  </div>
                </section>
              )}

              {/* ----- Mandatory-comment Escalate inline form (report §3.4: forces severity=severe) ----- */}
              {escalateOpen && (
                <section className="override-form" aria-label="Escalate form">
                  <h3>Escalate (HITL mandatory comment)</h3>
                  {escalateError && (
                    <div className="banner banner-error" role="alert">{escalateError}</div>
                  )}
                  <p className="small muted">
                    Escalate is the stronger action: it forces the severity impression to{' '}
                    <strong>severe</strong> and moves this submission to ESCALATED (report §3.4).
                    There is no severity choice here because the value is fixed server-side. The
                    same ≥10-character mandatory comment requirement applies and is enforced
                    server-side via Pydantic schema validation.
                  </p>
                  <div className="form-row">
                    <label htmlFor="escalate-notes">
                      Justification comment <span className="required">* (required, ≥10 characters)</span>:
                    </label>
                    <textarea
                      id="escalate-notes"
                      rows={4}
                      placeholder="e.g. Rapidly spreading erythema with systemic signs reported on history — escalating for urgent in-person review rather than the scheduled slot."
                      value={escalateNotes}
                      onChange={(e) => {
                        setEscalateNotes(e.target.value);
                        if (escalateError) setEscalateError(null);
                      }}
                    />
                    <p className="small muted">
                      Character count: <strong>{(escalateNotes || '').length}</strong>
                      {' '}(after trim: {(escalateNotes || '').trim().length})
                      {(escalateNotes || '').trim().length < 10 ? ' — minimum 10 required.' : ' ✓ meets minimum.'}
                    </p>
                  </div>
                  <div className="form-actions">
                    <button
                      className="secondary"
                      onClick={() => {
                        setEscalateOpen(false);
                        setEscalateError(null);
                        setEscalateNotes('');
                      }}
                      disabled={!!submittingAction}
                    >
                      Cancel
                    </button>
                    <button
                      className="escalate"
                      onClick={submitEscalate}
                      disabled={!!submittingAction || (escalateNotes || '').trim().length < 10}
                    >
                      {submittingAction === 'escalate' ? 'Escalating…' : 'Escalate Submission'}
                    </button>
                  </div>
                </section>
              )}
            </article>
          )}
        </section>
      </div>
    </div>
  );
}

export default ClinicianDashboard;
