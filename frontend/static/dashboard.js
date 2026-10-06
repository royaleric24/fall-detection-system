'use strict';

const show = (id, value) => { document.getElementById(id).textContent = value ?? '—'; };

async function poll() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store', signal: AbortSignal.timeout(2000)});
    if (!response.ok) throw new Error('Local HTTP unavailable');
    const state = await response.json();
    const raw = state.raw_prediction;
    show('source-id', state.source_id);
    show('sequence-id', state.sequence_id);
    show('application-status', state.application_status);
    document.getElementById('application-status').dataset.status = state.application_status;
    show('actionable', String(state.prediction_actionable));
    show('pose-status', state.pose_detected === null ? null : state.pose_detected ? 'DETECTED' : 'MISSING');
    show('edge-frame', state.frame_index);
    show('missing-frames', state.consecutive_missing_frames);
    show('recovery-frame', state.recovery_frame_index);
    show('probability', raw.fall_probability === null ? null : `${(raw.fall_probability * 100).toFixed(1)}%`);
    show('raw-label', raw.predicted_label);
    show('context', raw.buffer_length === null ? null : `${raw.buffer_length} frames`);
    show('prediction-frame', raw.frame_index);
    show('mqtt-status', state.mqtt_connected ? 'CONNECTED' : 'DISCONNECTED');
    show('prediction-status', state.cloud_prediction_status);
    show('prediction-age', state.prediction_age_seconds === null ? null : `${state.prediction_age_seconds.toFixed(1)} s`);
    show('http-status', 'Reading authoritative local Edge state');
    document.body.dataset.unavailable = 'false';
  } catch (_) {
    show('http-status', 'Local Dashboard unavailable — displayed values are the last snapshot.');
    document.body.dataset.unavailable = 'true';
  } finally {
    // Schedule after completion so slow requests never overlap.
    setTimeout(poll, 350);
  }
}

poll();
