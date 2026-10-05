# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy==2.3.5", "torch==2.8.0", "paho-mqtt==2.1.0"]
# ///
"""Long-running Stage 6 MQTT backend; network callback queues, main thread infers."""

import argparse
import json
import logging
import queue
import signal
import sys
import threading
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.app.inference import InferenceEngine, load_selected_model
from common.mqtt import MQTTConnection, decode_pose

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    model, config = load_selected_model()
    engine = InferenceEngine(model)
    pending: queue.Queue = queue.Queue(maxsize=1024)
    stopped = threading.Event()
    counters = dict(malformed=0, queue_overflow=0, reconnect_resets=0)
    examples = []

    def receive(client, userdata, message):
        try:
            pending.put_nowait((message.topic, message.payload))
        except queue.Full:
            counters["queue_overflow"] += 1
            LOGGER.error("MQTT queue overflow; missing transport frames will reset context")

    connection = MQTTConnection(receive)
    connection.subscription = f"{connection.prefix}/pose/+"
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *unused: stopped.set())
    connection.start()
    generation = connection.generation
    LOGGER.info("backend_ready checkpoint=%s", config["checkpoint"]["path"])
    connection.publish(f"{connection.prefix}/status/backend", dict(schema_version=1, state="ready"))
    try:
        while not stopped.is_set():
            if generation != connection.generation:
                engine.sources.clear()
                generation = connection.generation
                counters["reconnect_resets"] += 1
            try:
                topic, payload = pending.get(timeout=.2)
            except queue.Empty:
                continue
            try:
                message = decode_pose(payload, topic, connection.prefix)
                prediction = engine.handle(message)
                if prediction:
                    connection.publish(f"{connection.prefix}/prediction/{prediction['source_id']}", prediction)
                    if len(examples) < 3:
                        examples.append(prediction)
                    LOGGER.info("prediction source=%s frame=%d length=%d probability=%.6f",
                                prediction["source_id"], prediction["frame_index"],
                                prediction["buffer_length"], prediction["fall_probability"])
            except (ValueError, TypeError, KeyError, UnicodeError) as exc:
                counters["malformed"] += 1
                LOGGER.warning("rejected_payload reason=%s", exc)
            except ConnectionError as exc:
                engine.sources.clear()
                LOGGER.warning("MQTT unavailable reason=%s; restart edge sequences", exc)
    finally:
        connection.close()
        summary = dict(frames_received=engine.frames_received, predictions=engine.predictions,
                       ignored=engine.ignored, gap_resets=engine.gap_resets, **counters,
                       examples=examples, checkpoint=config["checkpoint"])
        LOGGER.info("backend_summary %s", json.dumps(summary))
        if args.summary:
            with args.summary.open("x") as handle:
                json.dump(summary, handle, indent=2, allow_nan=False)
                handle.write("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
