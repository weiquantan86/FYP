import { useEffect, useRef, useState, useCallback } from 'react';
import { getTriageStatus } from '../api';

const TERMINAL_STATUSES = new Set(['COMPLETED', 'FAILED']);

/**
 * Poll `/api/v1/triage/status/{jobId}` with bounded retries.
 *
 * - `intervalMs`: poll cadence (default 3000 as per spec)
 * - `timeoutMs`: max total time to poll before giving up (default 10 min)
 * - Stops automatically when status ∈ {COMPLETED, FAILED}
 * - Safe to call with jobId=null/undefined (sits idle until real id is set)
 *
 * Returns:
 *   { result, status, error, loading, attempt, stop, restart }
 */
export function usePolling(jobId, options = {}) {
  const {
    intervalMs = 3000,
    timeoutMs = 10 * 60 * 1000,
  } = options;

  const [status, setStatus] = useState('IDLE'); // IDLE | POLLING | COMPLETED | FAILED | TIMEOUT
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [attempt, setAttempt] = useState(0);

  const cancelledRef = useRef(false);
  const timerRef = useRef(null);
  const startedAtRef = useRef(0);
  const activeJobIdRef = useRef(null);

  const clearTimer = () => {
    if (timerRef.current !== null) {
      clearTimeout(timerRef.current);
      timerRef.current = null;
    }
  };

  const stop = useCallback(() => {
    cancelledRef.current = true;
    clearTimer();
    setStatus((prev) => (TERMINAL_STATUSES.has(prev) ? prev : 'IDLE'));
  }, []);

  const runOnce = useCallback(async () => {
    if (cancelledRef.current) return;
    if (!activeJobIdRef.current) return;

    setAttempt((n) => n + 1);
    try {
      const payload = await getTriageStatus(activeJobIdRef.current);
      setResult(payload);
      setError(null);

      if (TERMINAL_STATUSES.has(payload.status)) {
        clearTimer();
        setStatus(payload.status);
        return;
      }
    } catch (err) {
      // Transient network errors are swallowed; we keep polling until timeout.
      setError(err?.message || 'Polling error');
    }

    // Global timeout check
    if (Date.now() - startedAtRef.current >= timeoutMs) {
      clearTimer();
      setStatus('TIMEOUT');
      setError('Timed out waiting for triage processing.');
      return;
    }

    timerRef.current = setTimeout(runOnce, intervalMs);
  }, [intervalMs, timeoutMs]);

  useEffect(() => {
    // Reset every time the caller passes a new jobId.
    if (!jobId) {
      stop();
      setResult(null);
      setError(null);
      setAttempt(0);
      activeJobIdRef.current = null;
      return undefined;
    }

    cancelledRef.current = false;
    activeJobIdRef.current = jobId;
    startedAtRef.current = Date.now();
    setAttempt(0);
    setStatus('POLLING');
    setError(null);
    setResult(null);

    // Kick first tick quickly (50ms) so UI updates immediately.
    timerRef.current = setTimeout(runOnce, 50);

    return () => {
      cancelledRef.current = true;
      clearTimer();
    };
  }, [jobId, runOnce, stop]);

  const restart = useCallback(() => {
    cancelledRef.current = false;
    startedAtRef.current = Date.now();
    setAttempt(0);
    setError(null);
    if (activeJobIdRef.current) {
      setStatus('POLLING');
      timerRef.current = setTimeout(runOnce, 50);
    }
  }, [runOnce]);

  return {
    result,
    status,
    error,
    attempt,
    loading: status === 'POLLING',
    stop,
    restart,
  };
}

export default usePolling;
