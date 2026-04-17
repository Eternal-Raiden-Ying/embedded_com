#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from typing import Any, Dict, Optional

try:
    import requests
except Exception:
    requests = None

from .protocol import RemotePredictRequest, RemotePredictResponse, build_predict_multipart


class RemoteGraspClient:
    """Client interface for the remote grasp server.

    The final implementation should absorb the mature request flow currently
    exercised by ``VISTA/grasp_module/simulate_client_request.py``.
    """

    def __init__(self, base_url: str = "", logger=None):
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self._session = None
        self._session_open = False
        self._last_request_ts = 0.0
        self._last_error = ""

    def _log(self, level: str, message: str, **fields: Any) -> None:
        if self.logger is None:
            return
        extra = fields or None
        text = message if not extra else f"{message} | {extra}"
        fn = getattr(self.logger, level, None)
        if callable(fn):
            fn(text)

    def _failure(self, error: str, status_code: Optional[int] = None) -> RemotePredictResponse:
        self._last_error = str(error or "remote_error")
        return RemotePredictResponse(ok=False, payload={}, status_code=status_code, error=self._last_error)

    def _post_json(self, path: str, timeout_s: float, **kwargs: Any) -> RemotePredictResponse:
        self._last_request_ts = time.time()
        if not self._session_open or self._session is None:
            return self._failure("remote_session_not_open")
        if requests is None:
            return self._failure("requests_unavailable")
        if not self.base_url:
            return self._failure("remote_base_url_empty")
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.post(url, timeout=max(0.1, float(timeout_s)), **kwargs)
        except Exception as exc:
            return self._failure(str(exc))
        status_code = int(resp.status_code)
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw_text": resp.text}
        if 200 <= status_code < 300:
            self._last_error = ""
            return RemotePredictResponse(ok=True, payload=payload if isinstance(payload, dict) else {"value": payload}, status_code=status_code)
        return self._failure(str(payload), status_code=status_code)

    def configure(self, base_url: str) -> None:
        """Update the remote endpoint before the next request sequence."""
        self.base_url = str(base_url or "").rstrip("/")

    def open(self) -> None:
        """Prepare the underlying HTTP client session."""
        if requests is None:
            self._session = None
            self._session_open = True
            return
        if self._session is None:
            self._session = requests.Session()
        self._session_open = True

    def init_server(self, timeout_s: float = 15.0) -> Optional[RemotePredictResponse]:
        """Call the remote /api/v1/init endpoint."""
        return self._post_json("/api/v1/init", timeout_s=timeout_s)

    def predict(self, request: RemotePredictRequest) -> Optional[RemotePredictResponse]:
        """Call the remote /api/v1/predict endpoint using multipart payloads."""
        data, files = build_predict_multipart(request)
        return self._post_json(
            "/api/v1/predict",
            timeout_s=request.timeout_s,
            data=data,
            files=files,
        )

    def release_server(self, timeout_s: float = 5.0) -> Optional[RemotePredictResponse]:
        """Call the remote /api/v1/release endpoint."""
        return self._post_json("/api/v1/release", timeout_s=timeout_s)

    def close(self) -> None:
        """Tear down the underlying HTTP client session."""
        if self._session is not None:
            try:
                self._session.close()
            except Exception:
                pass
            self._session = None
        self._session_open = False

    def snapshot(self) -> Dict[str, Any]:
        """Expose remote client state for diagnostics and heartbeats."""
        return {
            "base_url": self.base_url,
            "session_open": self._session_open,
            "last_request_ts": self._last_request_ts,
            "last_error": self._last_error,
        }
