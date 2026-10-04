"""Synthetic HTTP firewall, playback state machine, and source-AVI integration."""
from __future__ import annotations

import hashlib
import http.client
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlencode

from ml.preprocessing import stage32j_review as d
from ml.preprocessing import stage32j_review_web as web
from tests.test_stage32j_review import make_run, open_phase1, open_phase2, lock_all_phase2, phase1, phase2


class Clock:
    def __init__(self): self.value=0.0
    def __call__(self): return self.value
    def tick(self, seconds): self.value+=seconds


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.p=web.Playback(8,False,self.clock)

    def finish(self, rate=1):
        self.p.start(rate)
        for index in range(8):
            self.assertEqual(self.p.index,index)
            self.p.frame(index,self.p.ticket)
            self.clock.value=self.p.deadline()
            completed=self.p.ack(index,self.p.ticket,True)
        return completed

    def test_first_pass_controls_and_no_completion_flag(self):
        for fn in (lambda:self.p.start(.5),lambda:self.p.step(1),lambda:self.p.stop()):
            with self.assertRaises(d.ReviewError): fn()
        self.assertFalse(self.p.full)
        self.assertTrue(self.finish())
        self.assertTrue(self.p.full);self.assertFalse(self.p.running)

    def test_ack_requires_delivery_visibility_sequential_ticket_and_elapsed_time(self):
        self.p.start(1);ticket=self.p.ticket
        with self.assertRaises(d.ReviewError): self.p.ack(0,ticket,True)
        with self.assertRaises(d.ReviewError): self.p.frame(1,ticket)
        with self.assertRaises(d.ReviewError): self.p.frame(0,'wrong')
        self.p.frame(0,ticket)
        with self.assertRaises(d.ReviewError): self.p.ack(0,ticket,True)
        self.clock.tick(.05)
        with self.assertRaises(d.ReviewError): self.p.ack(0,ticket,False)
        self.assertFalse(self.p.ack(0,ticket,True))
        with self.assertRaises(d.ReviewError): self.p.ack(0,ticket,True)
        self.assertFalse(self.p.full)

    def test_late_delivery_is_integrity_hold(self):
        self.p.start(1);self.p.frame(0,self.p.ticket);self.clock.tick(.3)
        with self.assertRaises(d.IntegrityError): self.p.ack(0,self.p.ticket,True)
        self.assertFalse(self.p.full);self.assertFalse(self.p.running)

    def test_abort_reload_does_not_certify(self):
        self.p.start(1);self.p.frame(0,self.p.ticket);self.clock.tick(.05);self.p.ack(0,self.p.ticket,True)
        self.p.stop(abort=True)
        self.assertEqual(self.p.index,0);self.assertFalse(self.p.full)
        self.assertFalse(web.Playback(8,False,self.clock).full)

    def test_exact_steps_clamps_slow_pause_and_last_frame(self):
        self.finish();self.p.step(1);self.assertEqual(self.p.index,7)
        self.p.step(-1);self.assertEqual(self.p.index,6)
        self.p.step(1);self.assertEqual(self.p.index,7)
        for _ in range(12): self.p.step(-1)
        self.assertEqual(self.p.index,0)
        self.p.step(1);self.assertEqual(self.p.index,1)
        self.p.start(.5);self.assertAlmostEqual(self.p.deadline()-self.clock(),.1)
        with self.assertRaises(d.ReviewError): self.p.step(1)
        self.p.stop();self.assertFalse(self.p.running)
        self.assertFalse(self.finish(.5))


class HTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='stage32j-http-')
        self.root=Path(self.temp.name).resolve();self.run=make_run(self.root,video=True);open_phase1(self.run)
        self.clock=Clock();self.server=web.ReviewServer(self.run,0,clock=self.clock)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.context=self.run.read()[0];self.first=self.context['cases'][0]['review_case_id']
        self.protected=['neutral_clip_id','objective_stage32i_flags','A01','A02','synthetic-clip',
                        self.context['candidate_permutation_seed'],str(self.root)]
        self.before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root/'source').rglob('*.avi')}

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join()
        self.assertEqual(self.before,{name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in self.before})
        self.temp.cleanup()

    def request(self,path,body=None,*,method=None,token=True,headers=None):
        conn=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=5)
        hdr={'X-Review-Token':self.server.token} if token else {}
        if body is not None: hdr['Content-Type']='application/json'
        hdr.update(headers or {})
        conn.request(method or ('POST' if body is not None else 'GET'),path,json.dumps(body) if body is not None else None,hdr)
        response=conn.getresponse();content=response.read();status=response.status;meta=dict(response.getheaders());conn.close()
        return status,content,meta

    def api(self,path,body=None):
        status,content,_=self.request(path,body);self.assertEqual(status,200,content)
        return json.loads(content)

    def assert_safe(self,content):
        text=content.decode() if isinstance(content,bytes) else json.dumps(content)
        for word in self.protected: self.assertNotIn(word,text)

    def denied(self,path,body=None,**kwargs):
        status,content,headers=self.request(path,body,**kwargs);self.assertGreaterEqual(status,400)
        self.assert_safe(content);self.assert_safe(headers)

    def open(self,key=None): return self.api('/review/open',{'case_id':key or self.first})

    def action(self,opened,action,**extra):
        return self.api('/review/playback',{'session':opened['session'],'action':action,**extra})

    def frame(self,opened,view):
        path='/review/frame?'+urlencode({'session':opened['session'],'frame_index':view['frame_index'],'ticket':view['ticket']})
        status,content,_=self.request(path);self.assertEqual(status,200,content);return content

    def finish(self,opened):
        view=self.action(opened,'start',rate=1)
        for index in range(8):
            self.assertEqual(view['frame_index'],index);self.frame(opened,view)
            self.clock.value=self.server.controller.playback.deadline()
            view=self.action(opened,'ack',frame_index=index,ticket=view['ticket'],visible=True)
        self.assertTrue(view['full_clip_viewed_at_1x']);return view

    def test_phase1_html_js_payload_headers_no_protected_data(self):
        status,body,headers=self.request('/',token=False);self.assertEqual(status,200)
        self.assert_safe(body);self.assert_safe(headers);self.assertNotIn(self.server.token.encode(),body)
        self.assertNotIn(b'innerHTML',body)
        for route in ('/review/cases','/review/definitions','/review/case?case_id='+self.first,'/review/schema?case_id='+self.first):
            value=self.api(route);self.assert_safe(value)
            if not route.endswith('definitions'):
                for word in ('preferred_frame','earliest_plausible_frame','latest_plausible_frame','Candidate X','Candidate Y'):
                    self.assertNotIn(word,json.dumps(value))

    def test_auth_host_origin_and_parameter_firewall(self):
        self.denied('/review/cases',token=False)
        self.denied('/review/cases',headers={'X-Review-Token':'wrong'})
        self.denied('/review/cases',headers={'Host':'attacker.invalid'})
        self.denied('/review/cases',headers={'Origin':'https://attacker.invalid'})
        self.denied('/review/case?case_id='+self.first+'&case_id=other')
        self.denied('/review/case?case_id='+self.first+'&path=/tmp/private')

    def test_no_admin_static_traversal_or_symlink_routes(self):
        (self.run.directory/'public-link').symlink_to(self.run.admin)
        for route in ('/_admin/genesis_manifest.json','/admin/advance','/review/../_admin/genesis_manifest.json',
                      '/review/%2e%2e/_admin/genesis_manifest.json','/review/%252e%252e/_admin/genesis_manifest.json',
                      '/public-link/genesis_manifest.json','/review/file?path='+str(self.run.admin),'/etc/passwd'):
            self.denied(route)
        self.denied('/admin/advance',{'state':'PHASE2_OPEN'})
        self.denied('/review/case?case_id=../../private')

    def test_direct_candidate_access_global_gate(self):
        self.denied('/review/candidates?case_id='+self.first)
        opened=self.open();self.finish(opened)
        self.api('/review/lock',{'phase':1,'case_id':self.first,'response':phase1()})
        self.denied('/review/candidates?case_id='+self.first)
        self.denied('/review/lock',{'phase':2,'case_id':self.first,'response':phase2()})

    def test_direct_completion_and_seek_bypasses_rejected(self):
        opened=self.open()
        self.denied('/review/playback',{'session':opened['session'],'action':'complete','full_clip_viewed_at_1x':True})
        self.denied('/review/playback',{'session':opened['session'],'action':'start','rate':1,'full_clip_viewed_at_1x':True})
        for action,extra in (('step',{'delta':1}),('start',{'rate':.5}),('seek',{'frame_index':7}),('pause',{})):
            self.denied('/review/playback',{'session':opened['session'],'action':action,**extra})
        row=phase1();row['full_clip_viewed_at_1x']=True
        self.denied('/review/lock',{'phase':1,'case_id':self.first,'response':row})
        self.denied('/review/lock',{'phase':1,'case_id':self.first,'response':phase1()})

    def test_synthetic_avi_pixels_exact_indices_and_no_interpolation(self):
        import cv2
        import numpy as np
        opened=self.open();view=opened['playback']
        def verify(view):
            data=self.frame(opened,view);frame=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR)
            self.assertTrue(np.all(frame==np.array([view['frame_index']*25,30,200],dtype=np.uint8)))
        verify(view);view=self.finish(opened);verify(view)
        for delta,expected in ((1,7),(-1,6),(1,7)):
            view=self.action(opened,'step',delta=delta);self.assertEqual(view['frame_index'],expected);verify(view)
        for _ in range(9): view=self.action(opened,'step',delta=-1)
        self.assertEqual(view['frame_index'],0);verify(view)
        view=self.action(opened,'step',delta=1);self.assertEqual(view['frame_index'],1);verify(view)
        self.denied('/review/frame?'+urlencode({'session':opened['session'],'frame_index':'1.5','ticket':view['ticket']}))
        self.denied('/review/frame?'+urlencode({'session':opened['session'],'frame_index':'2','ticket':view['ticket']}))
        self.action(opened,'start',rate=.5);self.action(opened,'pause')
        self.api('/review/lock',{'phase':1,'case_id':self.first,'response':phase1()})
        row=self.run.read()[1]['phase1'][self.first];self.assertTrue(row['slow_playback_used']);self.assertTrue(row['frame_step_used'])

    def test_reload_invalidates_ticket_and_incomplete_pass(self):
        opened=self.open();view=self.action(opened,'start',rate=1);self.frame(opened,view)
        reopened=self.open();self.assertFalse(reopened['playback']['full_clip_viewed_at_1x'])
        self.denied('/review/playback',{'session':opened['session'],'action':'ack','frame_index':0,'ticket':view['ticket'],'visible':True})
        self.assertFalse(self.run.reviewer_case(self.first)['full_clip_viewed_at_1x'])

    def test_anomaly_records_hold_and_blocks_progress(self):
        opened=self.open();view=self.action(opened,'start',rate=1);self.frame(opened,view);self.clock.tick(.3)
        self.denied('/review/playback',{'session':opened['session'],'action':'ack','frame_index':0,'ticket':view['ticket'],'visible':True})
        self.assertEqual(self.run.reviewer_case(self.first)['case_status'],'TECHNICAL_HOLD')
        self.denied('/review/open',{'case_id':self.first})

    def test_source_hash_mismatch_becomes_hold_without_decode(self):
        source=self.root/'source'/'Subject.1'/'synthetic.avi';original=source.read_bytes();source.write_bytes(b'corrupted synthetic AVI')
        try:
            with patch.object(d.IndexedFrameSource,'open_avi',side_effect=AssertionError('must not decode mismatch')):
                self.denied('/review/open',{'case_id':self.first})
            self.assertEqual(self.run.reviewer_case(self.first)['case_status'],'TECHNICAL_HOLD')
        finally: source.write_bytes(original)

    def test_cached_source_rechecked_and_hold_is_persisted(self):
        self.open()
        source=self.root/'source'/'Subject.1'/'synthetic.avi';original=source.read_bytes()
        source.write_bytes(b'changed after first decode')
        try:
            self.denied('/review/open',{'case_id':self.first})
            self.assertEqual(self.run.reviewer_case(self.first)['case_status'],'TECHNICAL_HOLD')
        finally: source.write_bytes(original)

    def test_browser_hidden_abort_and_replay_do_not_complete_first_pass(self):
        opened=self.open();self.action(opened,'start',rate=1)
        view=self.action(opened,'abort')
        self.assertFalse(view['full_clip_viewed_at_1x']);self.assertEqual(view['frame_index'],0)
        self.denied('/review/lock',{'phase':1,'case_id':self.first,'response':phase1()})

    def test_malformed_requests_and_forbidden_fields_never_leak(self):
        for body in ({'case_id':self.first,'path':str(self.run.admin)}, {'case_id':['private']}, {'case_id':None}):
            self.denied('/review/open',body)
        for forbidden in d.FORBIDDEN:
            row=phase1();row[forbidden]='X'
            self.denied('/review/lock',{'phase':1,'case_id':self.first,'response':row})

    def test_phase1_immutable_via_http_methods_and_stored_xss_is_text(self):
        opened=self.open();self.finish(opened);row=phase1();row['phase1_visual_rationale']='<img src=x onerror=alert(1)>'
        self.api('/review/lock',{'phase':1,'case_id':self.first,'response':row})
        before=self.run.read()[1]['phase1'][self.first]
        self.denied('/review/lock',{'phase':1,'case_id':self.first,'response':phase1()})
        for method in ('PUT','PATCH','DELETE'): self.denied('/review/lock',{'case_id':self.first},method=method)
        self.assertEqual(self.run.read()[1]['phase1'][self.first],before)
        self.assertEqual(self.api('/review/case?case_id='+self.first)['locked_response'],row)
        self.assertIn('textContent',web.PAGE);self.assertNotIn('innerHTML',web.PAGE)

    def test_recovery_schema_has_no_frame_answers(self):
        case=next(c for c in self.context['cases'] if c['review_target']=='recovery_coverage')
        schema=self.api('/review/schema?case_id='+case['review_case_id'])
        self.assertEqual(set(schema),set(d.PHASE1_INPUT)|set(d.RECOVERY_INPUT))
        self.assertFalse(any('frame' in name for name in schema))
        for field in ('recovery_start_frame','preferred_frame','earliest_plausible_frame','latest_plausible_frame'):
            row=phase1('recovery_coverage');row[field]=3
            self.denied('/review/lock',{'phase':1,'case_id':case['review_case_id'],'response':row})

    def test_phase2_anonymity_locks_and_no_phase3_reviewer_unblinding(self):
        open_phase2(self.run)
        for case in self.context['cases']:
            key=case['review_case_id']
            if case['phase2_required']:
                payload=self.api('/review/candidates?case_id='+key);self.assert_safe(payload)
                self.assertEqual(set(payload),{'X','Y'})
                if case['review_target']=='recovery_coverage':
                    self.assertNotIn('frame',json.dumps(payload));self.assertIn('status',json.dumps(payload))
            else: self.denied('/review/candidates?case_id='+key)
        self.denied('/review/lock',{'phase':2,'case_id':self.first,'response':{}})
        self.api('/review/lock',{'phase':2,'case_id':self.first,'response':phase2()})
        self.denied('/review/lock',{'phase':2,'case_id':self.first,'response':phase2()})
        for case in self.context['cases']:
            if case['phase2_required'] and case['review_case_id']!=self.first:
                self.run.lock_response(2,case['review_case_id'],phase2())
        self.run.advance('PHASE2_LOCKED');self.run.advance('PHASE3_UNBLINDED')
        self.assert_safe(self.api('/review/cases'))
        self.denied('/review/candidates?case_id='+self.first)
        self.denied('/admin/identity_summary');self.denied('/review/identity_summary')
        self.denied('/review/lock',{'phase':1,'case_id':self.first,'response':phase1()})

    def test_admin_phase_change_invalidates_active_session(self):
        opened=self.open()
        self.run.technical_hold(self.first,'Synthetic source concern')
        self.denied('/review/playback',{'session':opened['session'],'action':'start','rate':1})


if __name__ == '__main__': unittest.main()
