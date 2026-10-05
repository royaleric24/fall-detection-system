"""Real three-process AVI/MediaPipe/Mosquitto/GRU smoke; no accuracy evaluation."""

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.mqtt import MQTTConnection
from ml.datasets.extract_pose_video import SOURCE_ROOT, source_metadata

REPO = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(broker_path: Path, edge_python: Path, output: Path) -> dict:
    source = SOURCE_ROOT / "Subject.1/Fall backwards/FallBackwardsS1.avi"
    metadata = source_metadata(source)  # Authorize before hashing or OpenCV access.
    selected = json.loads((REPO / "artifacts/evaluation/stage5/final_ml_config.json").read_text())
    checkpoint = REPO / selected["checkpoint"]["path"]
    before = dict(avi=sha256(source), checkpoint=sha256(checkpoint))
    output.mkdir(parents=True, exist_ok=False)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    config = output / "local-broker.conf"
    config.write_text(f"listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\n")
    env = dict(os.environ, MQTT_HOST="127.0.0.1", MQTT_PORT=str(port), MQTT_USERNAME="", MQTT_PASSWORD="",
               MQTT_TOPIC_PREFIX="fall", CHECKPOINT_PATH=str(checkpoint))
    broker = backend = edge = observer = None
    handles = []
    messages = []
    backend_ready = threading.Event()
    try:
        broker_log = (output / "broker.log").open("x")
        handles.append(broker_log)
        broker = subprocess.Popen([str(broker_path), "-c", str(config)], stdout=broker_log,
                                  stderr=subprocess.STDOUT, cwd=REPO)
        deadline = time.monotonic()+10
        while True:
            if broker.poll() is not None:
                raise RuntimeError("Broker exited; inspect broker.log")
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=.2):
                    break
            except OSError:
                if time.monotonic()>deadline:
                    raise TimeoutError("Broker startup timeout")
                time.sleep(.05)
        # Observer and backend are distinct clients; edge/backend are separate OS processes.
        os.environ.update({k: env[k] for k in ("MQTT_HOST","MQTT_PORT","MQTT_USERNAME","MQTT_PASSWORD","MQTT_TOPIC_PREFIX")})
        def receive(client, userdata, message):
            try:
                data=json.loads(message.payload)
            except (ValueError, UnicodeError):
                return
            messages.append((message.topic,data))
            if message.topic=='fall/status/backend' and data.get('state')=='ready':
                backend_ready.set()
        observer=MQTTConnection(receive,subscription='fall/#')
        observer.start()
        backend_log=(output/'backend.log').open('x')
        handles.append(backend_log)
        backend=subprocess.Popen([sys.executable,'-m','backend.app.main','--summary',str(output/'backend.json')],
                                 cwd=REPO,env=env,stdout=backend_log,stderr=subprocess.STDOUT)
        if not backend_ready.wait(15):
            raise TimeoutError('Backend readiness timeout; inspect backend.log')
        observer.publish('fall/pose/smoke-edge',dict(schema_version=1))  # Intentional malformed-message resilience check.
        edge_log=(output/'edge.log').open('x')
        handles.append(edge_log)
        edge=subprocess.Popen([str(edge_python),'-m','edge.main','--source',str(source),
                               '--source-id','smoke-edge','--sequence-id','stage6-avi-smoke',
                               '--unpaced','--summary',str(output/'edge.json')],cwd=REPO,env=env,
                              stdout=edge_log,stderr=subprocess.STDOUT)
        if edge.wait(timeout=60)!=0:
            raise RuntimeError('Edge failed; inspect edge.log')
        expected_frames=metadata['expected_frame_count']
        expected_predictions=len(range(85,expected_frames,10))
        deadline=time.monotonic()+10
        while len([m for t,m in messages if t=='fall/prediction/smoke-edge'])<expected_predictions:
            if backend.poll() is not None:
                raise RuntimeError('Backend exited unexpectedly')
            if time.monotonic()>deadline:
                raise TimeoutError('Missing predictions')
            time.sleep(.05)
        backend.terminate()
        backend.wait(timeout=10)
        observer.close()
        observer=None
        edge_summary=json.loads((output/'edge.json').read_text())
        backend_summary=json.loads((output/'backend.json').read_text())
        predictions=[m for t,m in messages if t=='fall/prediction/smoke-edge']
        pose_frames=[m for t,m in messages if t=='fall/pose/smoke-edge' and m.get('message_type')=='frame']
        assert len(pose_frames)==edge_summary['frames_published']==backend_summary['frames_received']==expected_frames
        assert [m['frame_index'] for m in predictions]==list(range(85,expected_frames,10))
        assert all(0<=m['fall_probability']<=1 for m in predictions)
        assert backend_summary['malformed']==1 and backend_summary['queue_overflow']==0
        assert backend_summary['gap_resets']==0 and backend_summary['ignored']==0
        assert before==dict(avi=sha256(source),checkpoint=sha256(checkpoint))
        result=dict(stage=6,local_end_to_end='PASS',source=metadata['source_relative_video_path'],
                    subject_id=metadata['subject_id'],test_subjects_accessed=[],
                    broker='Mosquitto 2.0.22',separate_processes=True,
                    source_avi_and_checkpoint_unchanged=True,input_sha256=before,
                    frames_published=edge_summary['frames_published'],pose_frames_received=backend_summary['frames_received'],
                    first_inference_frame=predictions[0]['frame_index'],predictions=len(predictions),
                    example_prediction=predictions[0],edge_errors=edge_summary['errors'],
                    unexpected_backend_errors=0,intentionally_rejected_payloads=1,
                    edge_python=str(edge_python),backend_python=sys.executable,
                    remote_deployment='PENDING_INFRASTRUCTURE_ACCESS',prediction_records=predictions)
        (output/'smoke.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result
    finally:
        if observer:
            observer.close()
        for process in (edge,backend,broker):
            if process and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        for handle in handles:
            handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--broker',type=Path,required=True)
    parser.add_argument('--edge-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.broker.resolve(),args.edge_python.absolute(),args.output.resolve()),indent=2))
