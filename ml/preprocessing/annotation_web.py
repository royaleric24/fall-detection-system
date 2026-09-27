"""Loopback-only, neutral-payload web view for the Stage 3.2f local tool."""

from __future__ import annotations

import copy
import json
import re
import secrets
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from ml.preprocessing.annotation_pilot import PilotError
from ml.preprocessing.annotation_tool import (
    BOUNDARIES, COORDINATES, EVENTS, REASONS, STATUSES, AnnotationSession,
)
from ml.preprocessing.manual_annotation_contract import AnnotationContractError


HTML = """<!doctype html>
<html lang="en"><meta charset="utf-8"><title>Manual AVI annotation</title>
<style>
body {font: 15px system-ui; max-width: 980px; margin: 20px auto; color: #222}
img {max-width: 100%; border: 1px solid #aaa} button, select, input {margin: 3px; padding: 5px}
table {border-collapse: collapse} td, th {border: 1px solid #ccc; padding: 4px}
textarea {width: 95%; height: 70px} #message {min-height: 1.5em}
</style>
<h1>Manual AVI annotation</h1><p id="clip"></p><img id="frame" alt="Original AVI frame">
<p id="position"></p>
<p><button data-step="-10">−10</button><button data-step="-1">Previous</button>
<button data-step="1">Next</button><button data-step="10">+10</button>
<input id="jump" type="number" min="0"><button id="go">Go to frame</button></p>
<p>Event presence <select id="event"></select></p>
<table><thead><tr><th>Boundary</th><th>Status</th><th>Set current frame</th><th>Coordinates</th></tr></thead><tbody id="boundaries"></tbody></table>
<p>Reason scope <select id="reason-scope"></select></p><p><select id="reasons" multiple size="8"></select>
<button id="apply-reasons">Apply reason flags</button></p>
<p>Optional note</p><textarea id="note"></textarea>
<p><button id="save">Save draft</button><button id="finalize">Validate and finalize</button></p>
<p id="message" role="status"></p>
<script>
const TOKEN = __TOKEN__;
const CONFIG = __CONFIG__;
let state = null;
let noteInitialized = false;
let renderEpoch = 0;
let actionQueue = Promise.resolve();
let currentImageURL = null;
const $ = id => document.getElementById(id);
function showReasonFlags() {
  if (!state) return;
  const scope = $('reason-scope').value;
  const selected = scope === 'event' ? state.event_presence_reason_flags : state.boundaries[scope].reason_flags;
  for (const option of $('reasons').options) option.selected = selected.includes(option.value);
}
function send(action) {
  const epoch = ++renderEpoch;
  actionQueue = actionQueue.then(async () => {
    const response = await fetch('/action', {method:'POST', headers:{'Content-Type':'application/json', 'X-Tool-Token':TOKEN}, body:JSON.stringify(action)});
    const body = await response.json();
    if (!response.ok) { $('message').textContent = body.error || 'Action rejected'; return; }
    $('message').textContent = body.message || '';
    if (epoch === renderEpoch) await refresh(epoch);
  }).catch(() => { $('message').textContent = 'Display or action request failed'; });
  return actionQueue;
}
async function refresh(epoch = ++renderEpoch) {
  const candidate = await (await fetch('/state', {cache:'no-store'})).json();
  if (epoch !== renderEpoch) return;
  const url = '/frame?clip=' + encodeURIComponent(candidate.neutral_clip_id) + '&index=' + candidate.frame_index;
  const response = await fetch(url, {cache:'no-store'});
  if (!response.ok || response.headers.get('Content-Type') !== 'image/png' ||
      response.headers.get('X-Frame-Index') !== String(candidate.frame_index) ||
      response.headers.get('X-Neutral-Clip-Id') !== candidate.neutral_clip_id) {
    throw new Error('Frame identity check failed');
  }
  const imageURL = URL.createObjectURL(await response.blob());
  const ready = new Image();
  ready.id = 'frame'; ready.alt = 'Original AVI frame'; ready.src = imageURL;
  try { await ready.decode(); } catch (error) { URL.revokeObjectURL(imageURL); throw error; }
  if (epoch !== renderEpoch) { URL.revokeObjectURL(imageURL); return; }
  const previousImageURL = currentImageURL;
  currentImageURL = imageURL;
  state = candidate;
  $('frame').replaceWith(ready);
  $('clip').textContent = state.neutral_clip_id;
  $('position').textContent = `frame_index ${state.frame_index} / ${state.frame_count - 1} · ${state.time_ms} ms clip-relative`;
  if (previousImageURL) URL.revokeObjectURL(previousImageURL);
  $('event').value = state.event_presence || '';
  $('boundaries').replaceChildren();
  for (const name of CONFIG.boundaries) {
    const row = document.createElement('tr');
    const title = document.createElement('td'); title.textContent = name; row.append(title);
    const cell = document.createElement('td'); const select = document.createElement('select');
    for (const value of CONFIG.statuses) { const option = new Option(value,value); select.add(option); }
    select.value = state.boundaries[name].status;
    select.onchange = () => send({verb:'status',boundary:name,status:select.value});
    cell.append(select); row.append(cell);
    const marks = document.createElement('td');
    for (const coordinate of CONFIG.coordinates) {
      const button = document.createElement('button'); button.textContent = coordinate.replace('_plausible_frame','').replace('_frame','');
      button.onclick = () => send({verb:'mark',boundary:name,coordinate}); marks.append(button);
    }
    row.append(marks);
    const values = document.createElement('td');
    values.textContent = CONFIG.coordinates.map(key => key + '=' + (state.boundaries[name][key] ?? '—')).join(', ');
    row.append(values); $('boundaries').append(row);
  }
  showReasonFlags();
  if (!noteInitialized) { $('note').value = state.annotator_note || ''; noteInitialized = true; }
  if (state.finalized) {
    $('message').textContent = 'Finalized original is immutable';
    for (const element of document.querySelectorAll('button, select, input, textarea')) element.disabled = true;
  }
}
for (const value of ['', ...CONFIG.events]) $('event').add(new Option(value || 'Select event',value));
for (const value of ['event', ...CONFIG.boundaries]) $('reason-scope').add(new Option(value,value));
for (const value of CONFIG.reasons) $('reasons').add(new Option(value,value));
for (const button of document.querySelectorAll('[data-step]')) button.onclick = () => send({verb:'step',amount:Number(button.dataset.step)});
$('go').onclick = () => send({verb:'jump',index:Number($('jump').value)});
$('event').onchange = () => send({verb:'event',value:$('event').value});
$('reason-scope').onchange = showReasonFlags;
$('apply-reasons').onclick = () => send({verb:'reasons',scope:$('reason-scope').value,flags:[...$('reasons').selectedOptions].map(o=>o.value)});
$('save').onclick = () => send({verb:'save_draft',note:$('note').value});
$('finalize').onclick = () => send({verb:'finalize',note:$('note').value});
refresh().catch(() => { $('message').textContent = 'Display verification failed'; });
</script></html>
"""


def frame_png_bytes(frame: Any) -> bytes:
    """Lossless transient display encoding; never writes or replaces source AVI."""
    import cv2

    ok, encoded = cv2.imencode(".png", frame)
    if not ok:
        raise PilotError("Could not render selected frame")
    return encoded.tobytes()


def annotator_state(session: AnnotationSession) -> dict[str, Any]:
    """Explicit public projection; source path, subject and class never cross HTTP."""
    return dict(**session.public, frame_index=session.frames.index,
                time_ms=round(session.frames.index * 1000 / 20),
                event_presence=session.record["event_presence"],
                event_presence_reason_flags=list(session.record["event_presence_reason_flags"]),
                boundaries=copy.deepcopy(session.record["boundaries"]),
                annotator_note=session.record.get("annotator_note", ""),
                finalized=session.finalized)


def perform_action(session: AnnotationSession, action: dict[str, Any]) -> str:
    """Whitelisted UI commands; no user-supplied path or provenance field."""
    verb = action.get("verb")
    if verb == "step":
        amount = action.get("amount")
        if type(amount) is not int or not -1000 <= amount <= 1000:
            raise PilotError("Invalid frame step")
        session.frames.at(max(0, min(session.frames.frame_count - 1, session.frames.index + amount)))
    elif verb == "jump":
        session.frames.at(action.get("index"))
    elif verb == "event":
        session.set_event_presence(action.get("value"))
    elif verb == "status":
        session.set_status(action.get("boundary"), action.get("status"))
    elif verb == "mark":
        session.mark(action.get("boundary"), action.get("coordinate"))
    elif verb == "reasons":
        session.set_reason_flags(action.get("scope"), action.get("flags"))
    elif verb == "save_draft":
        session.set_note(action.get("note", ""))
        session.save_draft()
        return "Draft saved"
    elif verb == "finalize":
        session.set_note(action.get("note", ""))
        session.finalize()
        return "Finalized"
    else:
        raise PilotError("Unknown annotation action")
    return ""


def make_handler(session: AnnotationSession, token: str) -> type[BaseHTTPRequestHandler]:
    config = dict(events=EVENTS, boundaries=BOUNDARIES, statuses=STATUSES,
                  coordinates=COORDINATES, reasons=REASONS)
    page = HTML.replace("__TOKEN__", json.dumps(token)).replace("__CONFIG__", json.dumps(config))

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: bytes, content_type: str,
                  extra_headers: dict[str, str] | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in (extra_headers or {}).items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, code: int, value: dict[str, Any]) -> None:
            self._send(code, json.dumps(value).encode("utf-8"), "application/json; charset=utf-8")

        def do_GET(self) -> None:
            request = urlsplit(self.path)
            path = request.path
            if path == "/":
                self._send(200, page.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/state":
                self._json(200, annotator_state(session))
            elif path == "/frame":
                try:
                    params = parse_qs(request.query, keep_blank_values=True)
                    if (set(params) != {"clip", "index"} or any(len(values) != 1 for values in params.values())
                            or params["clip"][0] != session.public["neutral_clip_id"]
                            or re.fullmatch(r"0|[1-9][0-9]*", params["index"][0]) is None):
                        raise PilotError("Invalid neutral clip/frame request")
                    index, frame = session.frames.peek(int(params["index"][0]))
                    self._send(200, frame_png_bytes(frame), "image/png", {
                        "X-Frame-Index": str(index),
                        "X-Neutral-Clip-Id": session.public["neutral_clip_id"],
                    })
                except PilotError:
                    self._json(400, {"error": "Frame request rejected"})
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self) -> None:
            if urlsplit(self.path).path != "/action" or self.headers.get("X-Tool-Token") != token:
                self._json(403, {"error": "Action rejected"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise PilotError("Invalid action size")
                action = json.loads(self.rfile.read(length))
                if not isinstance(action, dict):
                    raise PilotError("Action must be an object")
                message = perform_action(session, action)
                self._json(200, {"message": message, "state": annotator_state(session)})
            except (AnnotationContractError, PilotError, ValueError, TypeError, OSError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            pass

    return Handler


def create_server(session: AnnotationSession, port: int = 8765) -> tuple[HTTPServer, str]:
    token = secrets.token_urlsafe(32)
    return HTTPServer(("127.0.0.1", port), make_handler(session, token)), token


def serve_web(session: AnnotationSession, port: int = 8765) -> None:
    server, _ = create_server(session, port)
    print(f"Open http://127.0.0.1:{server.server_port}/ in the local browser; press Ctrl-C to stop.")
    try:
        server.serve_forever()
    finally:
        server.server_close()
