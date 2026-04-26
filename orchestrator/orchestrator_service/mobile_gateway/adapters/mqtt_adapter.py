#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import ssl
from typing import Any, Callable, Dict, Optional

from ..config.schema import MqttAdapterConfig


class MqttAdapter:
    """Optional northbound MQTT adapter.

    The adapter is intentionally thin: it only transports mobile command/status
    payloads and delegates all command validation/mapping to the shared gateway
    handler.
    """

    def __init__(
        self,
        cfg: MqttAdapterConfig,
        command_handler: Callable[[Dict[str, Any]], None],
        logger: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    ):
        self.cfg = cfg
        self.command_handler = command_handler
        self.logger = logger
        self._client = None
        self._mqtt = None
        self._started = False
        self._status_topic = self._render_topic(self.cfg.topics.status)
        self._ack_topic = self._render_topic(self.cfg.topics.ack)
        self._heartbeat_topic = self._render_topic(self.cfg.topics.heartbeat)
        self._cmd_topic = self._render_topic(self.cfg.topics.cmd)

    def _log(self, level: str, event: str, **data: Any) -> None:
        if self.logger is not None:
            self.logger(level, event, data)

    def _render_topic(self, template: str) -> str:
        try:
            return str(template).format(robot_id=self.cfg.robot_id)
        except Exception:
            return str(template)

    def start(self) -> None:
        if not self.cfg.enabled:
            self._log("info", "mqtt_disabled", enabled=False)
            return
        try:
            import paho.mqtt.client as mqtt  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "MQTT adapter enabled but dependency `paho-mqtt` is missing. "
                "Install with `pip install paho-mqtt`."
            ) from exc

        if not str(self.cfg.broker_host or "").strip():
            raise RuntimeError("MQTT adapter enabled but broker_host is empty")

        self._mqtt = mqtt
        client = mqtt.Client(
            client_id=self.cfg.client_id,
            transport="websockets" if str(self.cfg.transport).lower().startswith("websocket") else "tcp",
        )
        if self.cfg.username:
            client.username_pw_set(self.cfg.username, self.cfg.password or None)
        if str(self.cfg.transport).lower().startswith("websocket"):
            client.ws_set_options(path=str(self.cfg.websocket_path or "/mqtt"))
        if self.cfg.use_tls:
            client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
            client.tls_insecure_set(False)
        client.on_connect = self._on_connect
        client.on_message = self._on_message
        client.on_disconnect = self._on_disconnect
        client.connect_async(
            host=self.cfg.broker_host,
            port=int(self.cfg.broker_port),
            keepalive=int(self.cfg.keepalive_s),
        )
        client.loop_start()
        self._client = client
        self._started = True
        self._log(
            "info",
            "mqtt_starting",
            broker_host=self.cfg.broker_host,
            broker_port=self.cfg.broker_port,
            transport=self.cfg.transport,
            cmd_topic=self._cmd_topic,
        )

    def stop(self) -> None:
        client = self._client
        self._client = None
        self._started = False
        if client is None:
            return
        try:
            client.loop_stop()
        except Exception:
            pass
        try:
            client.disconnect()
        except Exception:
            pass
        self._log("info", "mqtt_stopped")

    def publish_status(self, payload: Dict[str, Any]) -> None:
        self._publish(self._status_topic, payload, retain=self.cfg.retain_status)

    def publish_ack(self, payload: Dict[str, Any]) -> None:
        self._publish(self._ack_topic, payload, retain=False)

    def publish_heartbeat(self, payload: Dict[str, Any]) -> None:
        self._publish(self._heartbeat_topic, payload, retain=False)

    def _publish(self, topic: str, payload: Dict[str, Any], retain: bool) -> None:
        client = self._client
        if not self._started or client is None:
            return
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        info = client.publish(topic, line, qos=int(self.cfg.qos), retain=bool(retain))
        self._log("info", "mqtt_publish", topic=topic, retain=retain, rc=getattr(info, "rc", None))

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        client.subscribe(self._cmd_topic, qos=int(self.cfg.qos))
        self._log("info", "mqtt_connected", topic=self._cmd_topic, reason_code=int(reason_code))

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None) -> None:
        self._log("warn", "mqtt_disconnected", reason_code=int(reason_code))

    def _on_message(self, client, userdata, msg) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception as exc:
            self._log("warn", "mqtt_bad_json", topic=msg.topic, error=str(exc))
            return
        self._log("info", "mqtt_message", topic=msg.topic)
        self.command_handler(payload)
