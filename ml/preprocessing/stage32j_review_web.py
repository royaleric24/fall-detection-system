"""Loopback reviewer surface. Administration is CLI-only; no filesystem routes."""
from __future__ import annotations

import argparse
import hmac
import json
import secrets
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from ml.preprocessing import stage32j_review as domain
from ml.preprocessing.annotation_web import frame_png_bytes


class Playback:
    """Ordered source-frame delivery with monotonic display acknowledgements.

    Browser visibility/paint acknowledgements are workflow evidence, not DRM.
    A 200 ms maximum scheduling lag aborts rather than certifying a skipped pass.
    """
    MAX_LAG = 0.200

    def __init__(self, count: int, full: bool, clock: Callable[[], float] = time.monotonic):
        domain.require(type(count) is int and count > 0)
        self.count, self.full, self.clock = count, full, clock
        self.index, self.running, self.served = 0, False, False
        self.rate, self.started = 1.0, 0.0
        self.ticket = secrets.token_urlsafe(24)

    def view(self) -> dict[str, Any]:
        return {'frame_index': self.index, 'ticket': self.ticket, 'running': self.running,
                'full_clip_viewed_at_1x': self.full, 'rate': self.rate,
                'wait_ms': max(0, (self.deadline() - self.clock()) * 1000) if self.running else 0}

    def deadline(self) -> float:
        return self.started + (self.index + 1) / (20 * self.rate)

    def _ticket(self) -> None:
        self.ticket, self.served = secrets.token_urlsafe(24), False

    def start(self, rate: float) -> dict[str, Any]:
        domain.require(type(rate) in (int, float) and rate in (1.0, 0.5))
        domain.require(not self.running and (self.full or rate == 1.0))
        self.index, self.rate, self.running = 0, rate, True
        self.started = self.clock()
        self._ticket()
        return self.view()

    def frame(self, index: int, ticket: str) -> None:
        domain.require(index == self.index and hmac.compare_digest(ticket, self.ticket))
        # Before first pass, idle delivery is frame zero only.
        domain.require(self.full or self.running or index == 0)
        self.served = True

    def ack(self, index: int, ticket: str, visible: bool) -> bool:
        domain.require(self.running and visible is True and self.served and index == self.index
                       and hmac.compare_digest(ticket, self.ticket), 'Invalid frame acknowledgement')
        delay = self.clock() - self.deadline()
        domain.require(delay >= -0.000001, 'Frame acknowledged too early')
        if delay > self.MAX_LAG:
            self.running = False
            raise domain.IntegrityError('Playback synchronization anomaly')
        if self.index == self.count - 1:
            newly_complete = not self.full
            self.full, self.running = True, False
            self._ticket()
            return newly_complete
        self.index += 1
        self._ticket()
        return False

    def stop(self, *, abort: bool = False) -> dict[str, Any]:
        domain.require(abort or self.full, 'First pass cannot be paused')
        self.running = False
        if not self.full:
            self.index = 0
        self._ticket()
        return self.view()

    def step(self, delta: int) -> dict[str, Any]:
        domain.require(self.full and not self.running and type(delta) is int and delta in (-1, 1))
        self.index = min(max(self.index + delta, 0), self.count - 1)
        self._ticket()
        return self.view()


class ReviewerController:
    def __init__(self, run: domain.ReviewRun, clock: Callable[[], float] = time.monotonic):
        self.run, self.clock = run, clock
        self.session: str | None = None
        self.playback: Playback | None = None
        self.frames: list[bytes] = []
        self.case_id = ''
        self.head = b''
        # One clip cache avoids repeatedly decoding identical physical evidence.
        self.cache_alias = ''
        self.cache: list[bytes] = []

    def refresh_head(self) -> None:
        self.head = (self.run.admin / 'HEAD.json').read_bytes()

    def guard(self, session: str) -> Playback:
        domain.require(self.session is not None and session == self.session and self.playback is not None)
        domain.no_symlinks(self.run.admin / 'HEAD.json')
        domain.no_symlinks(self.run.admin / 'faults')
        domain.require(not any((self.run.admin / 'faults').iterdir()), 'Integrity HOLD')
        if (self.run.admin / 'HEAD.json').read_bytes() != self.head:
            # Admin phase changes invalidate in-flight sessions. All normal actions
            # below refresh HEAD after their own validated persistence operation.
            self.run.read()
            self.session = None
            raise domain.ReviewError('Session changed; reopen case')
        return self.playback

    def open(self, case_id: str) -> dict[str, Any]:
        self.session = None  # Reload/case change discards every incomplete pass.
        self.run.start_case(case_id)
        case = self.run.reviewer_case(case_id)
        domain.require(case['case_status'] != 'TECHNICAL_HOLD')
        if self.cache_alias != case['clip_alias']:
            source = self.run.open_frames(case_id)
            self.cache = [frame_png_bytes(source.peek(index)[1]) for index in range(source.frame_count)]
            self.cache_alias = case['clip_alias']
        else:
            context, _ = self.run.read()
            private = self.run.case(context, case_id)
            try:
                domain.resolve_source(Path(context['bundle']['source_root']), context['bundle']['pilot'], private['neutral_clip_id'])
            except (ValueError, OSError) as exc:
                self.run.technical_hold(case_id, 'Cached source AVI integrity failure')
                raise domain.IntegrityError('Source integrity HOLD') from exc
        self.frames, self.case_id = self.cache, case_id
        self.playback = Playback(len(self.frames), case['full_clip_viewed_at_1x'], self.clock)
        self.session = secrets.token_urlsafe(24)
        self.refresh_head()
        return {'session': self.session, 'case': case, 'playback': self.playback.view()}

    def frame(self, session: str, index: int, ticket: str) -> bytes:
        playback = self.guard(session)
        playback.frame(index, ticket)
        return self.frames[index]

    def action(self, request: dict[str, Any]) -> dict[str, Any]:
        action = request.get('action')
        extras = {'start': ('rate',), 'ack': ('frame_index', 'ticket', 'visible'),
                  'pause': (), 'abort': (), 'step': ('delta',)}
        domain.require(action in extras)
        domain.exact_keys(request, ('session', 'action', *extras[action]))
        playback = self.guard(request['session'])
        try:
            if action == 'start':
                domain.require(not playback.running and type(request['rate']) in (int, float)
                               and request['rate'] in (1, 0.5), 'Invalid replay request')
                if request['rate'] == 0.5:
                    domain.require(playback.full)
                    self.run.record_playback(self.case_id, slow=True)
                    self.refresh_head()
                playback.start(request['rate'])
            elif action == 'ack':
                if playback.ack(request['frame_index'], request['ticket'], request['visible']):
                    self.run.record_playback(self.case_id, completed=True)
                    self.refresh_head()
            elif action in ('pause', 'abort'):
                playback.stop(abort=action == 'abort')
            else:
                playback.step(request['delta'])
                self.run.record_playback(self.case_id, step=True)
                self.refresh_head()
        except domain.IntegrityError:
            self.run.technical_hold(self.case_id, 'Playback timing/index synchronization anomaly')
            self.session = None
            raise
        return playback.view()


def form_schema(run: domain.ReviewRun, case_id: str) -> dict[str, Any]:
    case = run.reviewer_case(case_id)
    if case['state'] == 'PHASE1_OPEN':
        result: dict[str, Any] = {'visual_characteristics': list(domain.VISUAL),
                                 'boundary_semantic_clarity': list(domain.CLARITY), 'phase1_visual_rationale': 'text'}
        if case['review_target'] == 'recovery_coverage':
            result.update({key: list(values) for key, values in domain.RECOVERY_ENUMS.items()})
            result.update(recovery_coverage_characteristics=list(domain.COVERAGE), recovery_coverage_rationale='text')
        return result
    return {'candidate_relation': list(domain.RELATIONS), 'mechanisms': list(domain.MECHANISMS),
            'confidence': list(domain.CONFIDENCE), 'protocol_clarification_candidate': 'boolean',
            'protocol_clarification_area': list(domain.CLARIFICATION),
            'protocol_clarification_rationale': 'text', 'phase2_mechanism_rationale': 'text'}


def frozen_definitions() -> dict[str, str]:
    # Only source definitions. Never forward the protocol's provenance/other sections.
    path = 'configs/stage32e_manual_annotation_protocol.json'
    protocol = json.loads(domain._git('show', f'{domain.SPEC_COMMIT}:{path}'))
    return {key: protocol['boundaries'][key] for key in domain.upstream.BOUNDARIES}


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Stage 3.2j review</title>
<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;background:#f7f8fa;color:#142330}button,select,textarea{font:inherit;margin:5px;padding:7px}textarea{display:block;width:95%;height:80px}fieldset{margin:12px 0;border:1px solid #bcc8d0}label{display:block;margin:5px}img{max-width:100%;background:#172027}pre{white-space:pre-wrap;overflow-wrap:anywhere}button:disabled{opacity:.45}#message{color:#9a301a}select[multiple]{min-height:180px;max-width:95%}</style>
<h1>Stage 3.2j review</h1><p>Describe visible structure and plausible mechanisms. Do not supply a replacement frame, interval, label, ranking or decision about which annotation is correct. Do not discuss cases or prior findings during review. Report technical anomalies immediately.</p>
<p id="mode"></p><details><summary>Frozen boundary definitions</summary><pre id="definitions"></pre></details>
<button id="refresh">Refresh phase / case list</button><select id="cases" aria-label="Review case"></select><button id="open">Open selected case</button>
<p id="message" role="status"></p><h2 id="heading"></h2><img id="frame" alt="Source AVI frame"><p id="position"></p>
<div><button id="play">Full 1× / replay</button><button id="slow">Replay 0.5×</button><button id="pause">Pause</button><button id="back">Frame −1</button><button id="next">Frame +1</button><button id="abort">Abort / reset incomplete pass</button></div>
<pre id="display"></pre><form id="form"></form><button id="lock">Validate and lock response permanently</button>
<label>Technical concern (source, indexing, playback, or review integrity)<textarea id="concern"></textarea></label><button id="hold">Record technical HOLD</button>
<script>
'use strict';
const token=new URLSearchParams(location.hash.slice(1)).get('token')||'';
history.replaceState(null,'',location.pathname);
const el=id=>document.getElementById(id);let session=null,info=null,view=null,loop=0,objectURL=null;
const headers={'X-Review-Token':token};
function message(s){el('message').textContent=s;}
async function api(path,body){const response=await fetch(path,{method:body?'POST':'GET',headers:{...headers,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined,cache:'no-store'});if(!response.ok)throw Error('Request rejected or integrity HOLD. Check required fields / phase; contact administrator if unexpected.');return response.json();}
function controls(){const full=view?.full_clip_viewed_at_1x===true,active=view?.running===true;for(const id of ['slow','back','next'])el(id).disabled=!full||active;el('pause').disabled=!full||!active;el('play').disabled=!session||active;el('lock').disabled=!info||!!info.locked_response||info.case_status==='TECHNICAL_HOLD'||(info.state==='PHASE1_OPEN'&&!full);}
async function paint(v){const r=await fetch('/review/frame?'+new URLSearchParams({session,frame_index:v.frame_index,ticket:v.ticket}),{headers,cache:'no-store'});if(!r.ok)throw Error('Frame unavailable. Contact administrator.');const url=URL.createObjectURL(await r.blob());el('frame').src=url;await el('frame').decode();await new Promise(requestAnimationFrame);if(objectURL)URL.revokeObjectURL(objectURL);objectURL=url;el('position').textContent='0-based source AVI frame_index: '+v.frame_index+' / '+(info.frame_count-1)+' | 20 FPS | '+v.rate+'×';}
async function action(action,extra={}){view=await api('/review/playback',{session,action,...extra});controls();return view;}
async function play(rate){const generation=++loop;view=await action('start',{rate});const epoch=performance.now();while(view.running&&generation===loop){const current=view;await paint(current);const deadline=epoch+(current.frame_index+1)*1000/(20*rate);await new Promise(r=>setTimeout(r,Math.max(0,deadline-performance.now())));if(generation!==loop||document.hidden)return;view=await action('ack',{frame_index:current.frame_index,ticket:current.ticket,visible:!document.hidden});}controls();}
function field(name,choices,multiple=false){const label=document.createElement('label');label.textContent=name;const input=document.createElement(Array.isArray(choices)?'select':'textarea');input.name=name;if(Array.isArray(choices)){input.multiple=multiple;if(!multiple)input.add(new Option('Select…',''));for(const x of choices)input.add(new Option(x,x));}label.append(input);el('form').append(label);return input;}
function values(){const data={};for(const input of el('form').elements){if(!input.name)continue;if(input.name.startsWith('mechanism:'))continue;data[input.name]=input.multiple?[...input.selectedOptions].map(x=>x.value):input.value;}if(info.state==='PHASE2_OPEN'){data.mechanisms=[...el('form').querySelectorAll('[data-mechanism]')].filter(x=>x.value).map(x=>({code:x.dataset.mechanism,confidence:x.value}));data.protocol_clarification_candidate=data.protocol_clarification_candidate==='true';if(!data.protocol_clarification_candidate){data.protocol_clarification_area=null;data.protocol_clarification_rationale='';}}return data;}
async function openCase(){++loop;const opened=await api('/review/open',{case_id:el('cases').value});session=opened.session;info=opened.case;view=opened.playback;el('heading').textContent=info.clip_alias+' · '+info.review_target+' · '+info.case_status;el('form').replaceChildren();el('display').textContent='';const schema=await api('/review/schema?case_id='+info.review_case_id);for(const [name,choices] of Object.entries(schema)){if(name==='confidence')continue;if(name==='mechanisms'){const group=document.createElement('fieldset');const legend=document.createElement('legend');legend.textContent='Mechanisms: choose confidence for each selected mechanism';group.append(legend);for(const code of choices){const label=document.createElement('label');label.textContent=code;const input=document.createElement('select');input.name='mechanism:'+code;input.dataset.mechanism=code;input.add(new Option('Not selected',''));for(const value of schema.confidence)input.add(new Option(value,value));label.append(input);group.append(label);}el('form').append(group);}else field(name,choices==='boolean'?['false','true']:choices,name.endsWith('_characteristics'));}if(info.state==='PHASE2_OPEN')el('display').textContent=JSON.stringify(await api('/review/candidates?case_id='+info.review_case_id),null,2);if(info.locked_response){el('display').textContent+='\nLocked response (read only):\n'+JSON.stringify(info.locked_response,null,2);for(const input of el('form').elements)input.disabled=true;}controls();await paint(view);message(view.full_clip_viewed_at_1x?'Full first pass already verified for this clip.':'Complete the continuous 1× pass before locking.');}
async function refresh(){++loop;session=null;info=null;view=null;el('frame').removeAttribute('src');el('form').replaceChildren();el('display').textContent='';const data=await api('/review/cases');el('mode').textContent=data.review_mode+' · '+data.state+(data.review_mode==='NONBLINDED_EXPLORATORY_REVIEW'?' · Exploratory; not independent blinded evidence.':'');el('cases').replaceChildren(...data.cases.map(c=>new Option(c.clip_alias+' · '+c.review_target,c.review_case_id)));el('open').disabled=!['PHASE1_OPEN','PHASE2_OPEN'].includes(data.state);el('definitions').textContent=JSON.stringify(await api('/review/definitions'),null,2);controls();}
function bind(id,fn){el(id).onclick=async e=>{e.preventDefault();try{await fn();}catch(error){++loop;message(error.message);}};}
bind('refresh',refresh);bind('open',openCase);bind('play',()=>play(1));bind('slow',()=>play(.5));bind('pause',async()=>{++loop;await action('pause');});bind('abort',async()=>{++loop;await action('abort');await paint(view);});for(const [id,delta] of [['back',-1],['next',1]])bind(id,async()=>{await action('step',{delta});await paint(view);});bind('lock',async()=>{await api('/review/lock',{case_id:info.review_case_id,phase:info.state==='PHASE1_OPEN'?1:2,response:values()});await openCase();message('Response locked permanently.');});bind('hold',async()=>{++loop;await api('/review/hold',{case_id:info.review_case_id,reason:el('concern').value});session=null;controls();message('Technical HOLD recorded. Contact administrator.');});
document.addEventListener('visibilitychange',()=>{if(document.hidden&&session){++loop;action('abort').catch(()=>{});}});window.addEventListener('pagehide',()=>{++loop;});refresh().catch(e=>message(e.message));
</script></html>'''


class ReviewServer(HTTPServer):
    def __init__(self, run: domain.ReviewRun, port: int = 8766, *, clock: Callable[[], float] = time.monotonic):
        self.controller = ReviewerController(run, clock)
        self.token = secrets.token_urlsafe(32)
        self.definitions = frozen_definitions()
        super().__init__(('127.0.0.1', port), ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer

    def log_message(self, format: str, *args: Any) -> None:
        pass  # Never log token-bearing URLs, paths, responses, or reviewer content.

    def send(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        for key, value in {'Content-Type': content_type, 'Content-Length': str(len(body)), 'Cache-Control': 'no-store',
                           'X-Content-Type-Options': 'nosniff', 'Referrer-Policy': 'no-referrer',
                           'Content-Security-Policy': "default-src 'none'; img-src 'self' blob:; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"}.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def dispatch(self, post: bool) -> None:
        try:
            expected_host = f'127.0.0.1:{self.server.server_port}'
            domain.require(self.headers.get('Host') == expected_host)
            domain.require(self.headers.get('Origin') in (None, f'http://{expected_host}'))
            parsed = urlsplit(self.path)
            domain.require(not parsed.scheme and not parsed.netloc and '%' not in parsed.path and '..' not in parsed.path)
            if not post and self.path == '/':
                self.send(PAGE.encode(), 'text/html; charset=utf-8')
                return
            domain.require(hmac.compare_digest(self.headers.get('X-Review-Token', ''), self.server.token))
            controller, run = self.server.controller, self.server.controller.run
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=True)
            domain.require(all(len(values) == 1 for values in query.values()))
            query = {key: values[0] for key, values in query.items()}
            if post:
                domain.require(not query and self.headers.get('Content-Type') == 'application/json'
                               and self.headers.get('Transfer-Encoding') is None)
                length = int(self.headers.get('Content-Length', '0'))
                domain.require(0 < length <= 65536)
                data = json.loads(self.rfile.read(length))
                domain.require(isinstance(data, dict))
                if parsed.path == '/review/open':
                    domain.exact_keys(data, ('case_id',))
                    result = controller.open(data['case_id'])
                elif parsed.path == '/review/playback':
                    result = controller.action(data)
                elif parsed.path == '/review/lock':
                    domain.exact_keys(data, ('case_id', 'phase', 'response'))
                    run.lock_response(data['phase'], data['case_id'], data['response'])
                    controller.session = None
                    result = {'locked': True}
                elif parsed.path == '/review/hold':
                    domain.exact_keys(data, ('case_id', 'reason'))
                    run.technical_hold(data['case_id'], data['reason'])
                    controller.session = None
                    result = {'case_status': 'TECHNICAL_HOLD'}
                else:
                    raise domain.ReviewError('Unknown endpoint')
            elif parsed.path in ('/review/cases', '/review/definitions'):
                domain.exact_keys(query, ())
                result = run.reviewer_cases() if parsed.path.endswith('cases') else self.server.definitions
            elif parsed.path in ('/review/case', '/review/schema', '/review/candidates'):
                domain.exact_keys(query, ('case_id',))
                case_id = query['case_id']
                result = (run.reviewer_case(case_id) if parsed.path.endswith('/case') else
                          form_schema(run, case_id) if parsed.path.endswith('/schema') else run.candidates(case_id))
            elif parsed.path == '/review/frame':
                domain.exact_keys(query, ('session', 'frame_index', 'ticket'))
                body = controller.frame(query['session'], int(query['frame_index']), query['ticket'])
                self.send(body, 'image/png')
                return
            else:
                raise domain.ReviewError('Unknown endpoint')
            self.send(domain.canonical_bytes(result), 'application/json')
        except (ValueError, TypeError, KeyError, OSError, OverflowError):
            self.send(b'{"error":"Request unavailable; check phase, required fields or contact administrator."}\n', 'application/json', 409)

    def do_GET(self) -> None:
        self.dispatch(False)

    def do_POST(self) -> None:
        self.dispatch(True)

    def do_PUT(self) -> None:
        self.send(b'{"error":"Unavailable method"}', 'application/json', 405)

    do_DELETE = do_PATCH = do_PUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    run = domain.ReviewRun(args.run_directory)
    with ReviewServer(run, args.port) as server:
        print(f'Authorized reviewer UI: http://127.0.0.1:{server.server_port}/#token={server.token}', flush=True)
        server.serve_forever()


if __name__ == '__main__':
    main()
