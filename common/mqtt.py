"""Small JSON protocol and Paho connection wrapper shared by edge and backend."""

import json
import math
import os
import re
import threading
import uuid
from typing import Any, Callable

import paho.mqtt.client as mqtt

IDENTIFIER = re.compile(r"[A-Za-z0-9_.-]{1,80}\Z")
FEATURE_DIM = 132


def identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError("Invalid source/sequence/topic identifier")
    return value


def validate_pose(message: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(message, dict) or type(message.get("schema_version")) is not int or message["schema_version"] != 1:
        raise ValueError("Invalid pose schema")
    identifier(message.get("source_id"))
    identifier(message.get("sequence_id"))
    kind = message.get("message_type")
    if kind not in ("sequence_start", "frame", "sequence_end"):
        raise ValueError("Invalid message_type")
    if kind == "frame":
        for name in ("frame_index", "timestamp_ms"):
            if type(message.get(name)) is not int or message[name] < 0:
                raise ValueError(f"Invalid {name}")
        if message.get("timestamp_kind") not in ("clip_relative", "unix_epoch"):
            raise ValueError("Invalid timestamp_kind")
        fps = message.get("fps")
        if type(fps) not in (int, float) or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Invalid FPS")
        values = message.get("features")
        if (not isinstance(values, list) or len(values) != FEATURE_DIM
                or any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > 3.402823466e38 for v in values)):
            raise ValueError("Expected 132 finite float32-representable features")
        if type(message.get("pose_detected")) is not bool:
            raise ValueError("Missing pose_detected flag")
    return message


def decode_pose(payload: bytes, topic: str, prefix: str = "fall") -> dict[str, Any]:
    if len(payload) > 32768:
        raise ValueError("Oversized pose message")
    message = validate_pose(json.loads(payload))
    if topic != f"{identifier(prefix)}/pose/{message['source_id']}":
        raise ValueError("Topic/source mismatch")
    return message


def encode(message: dict[str, Any]) -> str:
    return json.dumps(message, separators=(",", ":"), allow_nan=False)


class MQTTConnection:
    """VERSION2 callbacks, explicit connection/subscription readiness, QoS1 flush."""

    def __init__(self, on_message: Callable | None = None, subscription: str | None = None) -> None:
        self.host = os.getenv("MQTT_HOST", "localhost")
        self.port = int(os.getenv("MQTT_PORT", "1883"))
        self.prefix = identifier(os.getenv("MQTT_TOPIC_PREFIX", "fall"))
        self.ready = threading.Event()
        self.connected = threading.Event()
        self.generation = 0
        self.subscription = subscription
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=f"fall-{uuid.uuid4().hex}")
        username = os.getenv("MQTT_USERNAME", "")
        if username:
            self.client.username_pw_set(username, os.getenv("MQTT_PASSWORD", ""))
        self.client.on_connect = self._connect
        self.client.on_disconnect = self._disconnect
        self.client.on_subscribe = lambda *args: self.ready.set() if not any(r.is_failure for r in args[3]) else None
        if on_message is not None:
            self.client.on_message = on_message

    def _connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            return
        self.generation += 1
        self.connected.set()
        if self.subscription:
            client.subscribe(self.subscription, qos=1)
        else:
            self.ready.set()

    def _disconnect(self, *args) -> None:
        self.ready.clear()
        self.connected.clear()

    def start(self) -> None:
        self.client.connect(self.host, self.port, 60)
        self.client.loop_start()
        if not self.ready.wait(10):
            self.close()
            raise ConnectionError("MQTT connect/subscription timeout")

    def publish(self, topic: str, message: dict[str, Any]) -> None:
        if not self.connected.is_set():
            raise ConnectionError("MQTT disconnected; restart source sequence")
        info = self.client.publish(topic, encode(message), qos=1, retain=False)
        info.wait_for_publish(timeout=10)
        if info.rc != mqtt.MQTT_ERR_SUCCESS or not info.is_published():
            raise ConnectionError("MQTT publication failed/timed out")

    def close(self) -> None:
        self.client.disconnect()
        self.client.loop_stop()
