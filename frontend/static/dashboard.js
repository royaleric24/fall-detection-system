'use strict';

const show = (id, value) => { document.getElementById(id).textContent = value ?? '--'; };
const THRESHOLD_REFERENCE = 0.5; // Chart reference only. Never classifies a prediction.
let probabilityHistory = [];

function drawProbabilityTrend(history) {
  const canvas = document.getElementById('probability-chart');
  const context = canvas.getContext('2d');
  const bounds = canvas.getBoundingClientRect();
  const width = bounds.width;
  const height = bounds.height;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const left = 42, right = width - 20, top = 12, bottom = height - 26;
  const y = probability => bottom - probability * (bottom - top);
  context.font = '11px system-ui, sans-serif';
  context.textBaseline = 'middle';
  context.textAlign = 'right';
  for (const probability of [0, 0.25, 0.5, 0.75, 1]) {
    const reference = probability === THRESHOLD_REFERENCE;
    context.strokeStyle = reference ? '#f3cc80' : '#29394b';
    context.fillStyle = reference ? '#f3cc80' : '#a4b2c3';
    context.lineWidth = 1;
    context.setLineDash(reference ? [5, 5] : []);
    context.beginPath();
    context.moveTo(left, y(probability));
    context.lineTo(right, y(probability));
    context.stroke();
    context.fillText(`${Math.round(probability * 100)}%`, left - 8, y(probability));
  }
  context.setLineDash([]);
  if (history.length === 0) {
    context.fillStyle = '#a4b2c3';
    context.textAlign = 'center';
    context.fillText('Waiting for cloud inference samples', (left + right) / 2, top + 24);
    show('sample-count', 'No prediction samples yet');
    canvas.setAttribute('aria-label', 'Fall probability trend. Waiting for cloud inference samples.');
    return;
  }
  const first = history[0], last = history[history.length - 1];
  const span = last.frame_index - first.frame_index;
  const x = frame => span === 0 ? (left + right) / 2 : left + (frame - first.frame_index) / span * (right - left);
  context.strokeStyle = '#87c0ff';
  context.fillStyle = '#87c0ff';
  context.lineWidth = 2.5;
  context.lineJoin = 'round';
  context.beginPath();
  history.forEach((point, index) => {
    if (index === 0) context.moveTo(x(point.frame_index), y(point.fall_probability));
    else context.lineTo(x(point.frame_index), y(point.fall_probability));
  });
  context.stroke();
  for (const point of history) {
    context.beginPath();
    context.arc(x(point.frame_index), y(point.fall_probability), 2.5, 0, Math.PI * 2);
    context.fill();
  }
  context.fillStyle = '#a4b2c3';
  context.textAlign = span === 0 ? 'center' : 'left';
  context.fillText(String(first.frame_index), x(first.frame_index), height - 9);
  if (span !== 0) {
    context.textAlign = 'right';
    context.fillText(String(last.frame_index), right, height - 9);
  }
  show('sample-count', `${history.length} inference sample${history.length === 1 ? '' : 's'} · frames ${first.frame_index}–${last.frame_index}`);
  canvas.setAttribute('aria-label', `Fall probability trend. ${history.length} cloud inference samples. Latest ${(last.fall_probability * 100).toFixed(1)} percent at frame ${last.frame_index}.`);
}

function renderState(state) {
  const raw = state.raw_prediction;
  show('source-id', state.source_id);
  show('sequence-id', state.sequence_id);
  const statusText = {
    INITIALIZING: 'INITIALIZING', NORMAL: 'NORMAL', FALL: 'FALL DETECTED',
    POSE_LOST: 'POSE LOST', RECOVERING: 'RECOVERING',
  };
  show('application-status', statusText[state.application_status] ?? state.application_status);
  const statusNode = document.getElementById('application-card');
  statusNode.dataset.status = state.application_status;
  show('actionable', state.prediction_actionable ? 'YES' : 'NO');
  show('decision-note', raw.predicted_label === null ? 'Waiting for the first cloud prediction.' :
    state.prediction_actionable ? 'Current cloud prediction is actionable.' :
      'Raw model output suppressed by pose-validity gate.');
  const poseText = state.pose_detected === null ? '--' : state.pose_detected ? 'DETECTED' : 'MISSING';
  show('pose-status', poseText);
  show('health-pose-status', poseText);
  document.getElementById('health-pose-status').dataset.tone = state.pose_detected === null ? 'neutral' : state.pose_detected ? 'good' : 'warning';
  show('edge-frame', state.frame_index);
  show('missing-frames', state.consecutive_missing_frames);
  show('recovery-frame', state.recovery_frame_index);
  show('probability', raw.fall_probability === null ? null : `${(raw.fall_probability * 100).toFixed(1)}%`);
  show('raw-label', raw.predicted_label === null ? null : raw.predicted_label.replace('_', '-').toUpperCase());
  document.getElementById('raw-label').dataset.label = raw.predicted_label ?? '';
  show('context', raw.buffer_length === null ? null : `${raw.buffer_length} frames`);
  show('prediction-frame', raw.frame_index);
  const transport = state.mqtt_connected ? 'CONNECTED' : 'DISCONNECTED';
  show('mqtt-status', transport);
  show('header-mqtt-status', transport);
  document.getElementById('mqtt-status').dataset.tone = state.mqtt_connected ? 'good' : 'warning';
  document.getElementById('mqtt-badge').dataset.tone = state.mqtt_connected ? 'good' : 'warning';
  show('prediction-status', state.cloud_prediction_status.replace('_', ' '));
  document.getElementById('prediction-status').dataset.tone = state.cloud_prediction_status === 'ACTIVE' ? 'good' : state.cloud_prediction_status === 'STALE' ? 'warning' : 'neutral';
  show('prediction-age', state.prediction_age_seconds === null ? 'Waiting for a prediction' : `Last new prediction ${state.prediction_age_seconds.toFixed(1)} s ago`);
  // Replace with server-owned history. Polling never appends a probability sample.
  probabilityHistory = state.probability_history;
  drawProbabilityTrend(probabilityHistory);
  show('http-status', 'Local Dashboard connected');
  document.getElementById('connection-warning').hidden = true;
  document.body.dataset.unavailable = 'false';
}

async function poll() {
  try {
    const response = await fetch('/api/state', {cache: 'no-store', signal: AbortSignal.timeout(2000)});
    if (!response.ok) throw new Error('Local HTTP unavailable');
    renderState(await response.json());
  } catch (_) {
    show('http-status', 'Dashboard connection lost · last snapshot');
    document.getElementById('connection-warning').hidden = false;
    document.body.dataset.unavailable = 'true';
  } finally {
    // Schedule after completion so slow requests never overlap.
    setTimeout(poll, 350);
  }
}

window.addEventListener('resize', () => drawProbabilityTrend(probabilityHistory));
drawProbabilityTrend(probabilityHistory);
poll();
