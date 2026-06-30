#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
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

    def _simple_request_dump(self, event: str, *, url: str, timeout_s: float) -> None:
        if not event:
            return
        self._log("info", event, data={"url": url, "timeout_s": float(timeout_s)})

    def _response_dump(
        self,
        event: str,
        *,
        status_code: Optional[int],
        elapsed_ms: Optional[int],
        payload: Any = None,
        error: str = "",
        raw_text: str = "",
    ) -> None:
        if not event:
            return
        json_payload = payload if isinstance(payload, dict) else {}
        dump = {
            "status_code": status_code,
            "elapsed_ms": elapsed_ms,
            "json_status": json_payload.get("status"),
            "json_keys": sorted(str(key) for key in json_payload.keys()),
            "error_message": str(error or json_payload.get("error") or json_payload.get("message") or ""),
            "result": "timeout" if "timeout" in str(error or "").lower() or "timed out" in str(error or "").lower() else ("ok" if not error and status_code is not None and 200 <= int(status_code) < 300 else "error"),
            "body_snippet_1000": str(raw_text or "")[:1000],
        }
        self._log("info", event, data=dump)

    def _post_json(self, path: str, timeout_s: float, response_dump_event: str = "", **kwargs: Any) -> RemotePredictResponse:
        self._last_request_ts = time.time()
        if not self._session_open or self._session is None:
            self._response_dump(response_dump_event, status_code=None, elapsed_ms=0, error="remote_session_not_open")
            return self._failure("remote_session_not_open")
        if requests is None:
            self._response_dump(response_dump_event, status_code=None, elapsed_ms=0, error="requests_unavailable")
            return self._failure("requests_unavailable")
        if not self.base_url:
            self._response_dump(response_dump_event, status_code=None, elapsed_ms=0, error="remote_base_url_empty")
            return self._failure("remote_base_url_empty")
        url = f"{self.base_url}{path}"
        start = time.time()
        try:
            resp = self._session.post(url, timeout=max(0.1, float(timeout_s)), **kwargs)
        except Exception as exc:
            err = str(exc)
            self._response_dump(
                response_dump_event,
                status_code=None,
                elapsed_ms=int(round((time.time() - start) * 1000.0)),
                error=err,
                raw_text="",
            )
            if "timeout" in err.lower() or "timed out" in err.lower():
                return self._failure("timeout")
            return self._failure(err)
        elapsed_ms = int(round((time.time() - start) * 1000.0))
        status_code = int(resp.status_code)
        raw_text = ""
        try:
            payload = resp.json()
        except Exception:
            raw_text = str(resp.text or "")
            payload = {"raw_text": raw_text[:1000]}
        self._response_dump(
            response_dump_event,
            status_code=status_code,
            elapsed_ms=elapsed_ms,
            payload=payload,
            error="" if 200 <= status_code < 300 else str(payload),
            raw_text=raw_text,
        )
        if 200 <= status_code < 300:
            self._last_error = ""
            return RemotePredictResponse(ok=True, payload=payload if isinstance(payload, dict) else {"value": payload}, status_code=status_code)
        return self._failure(str(payload), status_code=status_code)

    def _request_dump(
        self,
        *,
        url: str,
        data: Dict[str, Any],
        files: Dict[str, Any],
    ) -> None:
        metadata = {}
        try:
            metadata = json.loads(str(data.get("metadata") or "{}"))
        except Exception:
            metadata = {"parse_error": "invalid_metadata_json"}
        form = {str(key): str(value) for key, value in data.items() if key != "metadata"}
        file_summary = {}
        for field_name, file_tuple in dict(files or {}).items():
            filename = ""
            content = b""
            content_type = ""
            if isinstance(file_tuple, tuple):
                if len(file_tuple) > 0:
                    filename = str(file_tuple[0])
                if len(file_tuple) > 1:
                    content = file_tuple[1] or b""
                if len(file_tuple) > 2:
                    content_type = str(file_tuple[2])
            file_summary[str(field_name)] = {
                "filename": filename,
                "content_type": content_type,
                "bytes": len(content) if hasattr(content, "__len__") else 0,
            }
        self._log(
            "info",
            "remote_predict_request_dump",
            data={
                "url": url,
                "form": form,
                "metadata": metadata,
                "files": file_summary,
            },
        )

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
        url = f"{self.base_url}/api/v1/init"
        self._simple_request_dump("remote_init_request_dump", url=url, timeout_s=timeout_s)
        return self._post_json("/api/v1/init", timeout_s=timeout_s, response_dump_event="remote_init_response_dump")

    def predict(self, request: RemotePredictRequest) -> Optional[RemotePredictResponse]:
        """Call the remote /api/v1/predict endpoint using multipart payloads."""
        data, files = build_predict_multipart(request)
        url = f"{self.base_url}/api/v1/predict"
        self._request_dump(url=url, data=data, files=files)
        return self._post_json(
            "/api/v1/predict",
            timeout_s=request.timeout_s,
            response_dump_event="remote_predict_response_dump",
            data=data,
            files=files,
        )

    def release_server(self, timeout_s: float = 5.0) -> Optional[RemotePredictResponse]:
        """Call the remote /api/v1/release endpoint."""
        url = f"{self.base_url}/api/v1/release"
        self._simple_request_dump("remote_release_request_dump", url=url, timeout_s=timeout_s)
        return self._post_json("/api/v1/release", timeout_s=timeout_s, response_dump_event="remote_release_response_dump")

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
