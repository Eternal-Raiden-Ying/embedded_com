#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class SystemMetricsSampler:
    """Low-frequency process/system metrics with psutil optional."""

    def __init__(self, module: str, interval_s: float = 1.0):
        self.module = str(module or "process")
        self.interval_s = max(0.5, float(interval_s or 1.0))
        self._last_ts = 0.0
        self._last_proc_cpu = None
        self._last_total_cpu = None
        self._psutil = None
        self._process = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception:
            self._psutil = None
            self._process = None

    def sample_if_due(self, *, force: bool = False) -> Optional[Dict[str, Any]]:
        now = time.time()
        if not force and now - float(self._last_ts or 0.0) < self.interval_s:
            return None
        self._last_ts = now
        if self._psutil is not None and self._process is not None:
            return self._sample_psutil(now)
        return self._sample_proc(now)

    def _sample_psutil(self, now: float) -> Dict[str, Any]:
        psutil = self._psutil
        proc = self._process
        mem = proc.memory_info()
        out: Dict[str, Any] = {
            "ts": now,
            "module": self.module,
            "source": "psutil",
            "process_cpu_percent": proc.cpu_percent(None),
            "system_cpu_percent": psutil.cpu_percent(None),
            "process_rss_mb": float(getattr(mem, "rss", 0) or 0) / (1024.0 * 1024.0),
            "process_vms_mb": float(getattr(mem, "vms", 0) or 0) / (1024.0 * 1024.0),
            "system_mem_percent": float(psutil.virtual_memory().percent),
            "thread_count": int(proc.num_threads()),
        }
        try:
            out["per_cpu_percent"] = psutil.cpu_percent(None, percpu=True)
        except Exception:
            pass
        try:
            out["open_fds"] = int(proc.num_fds())
        except Exception:
            pass
        try:
            out["loadavg"] = list(os.getloadavg())
        except Exception:
            pass
        temp = self._thermal_temp_c()
        if temp is not None:
            out["temperature_c"] = temp
        return out

    def _sample_proc(self, now: float) -> Dict[str, Any]:
        rss_mb, vms_mb = self._proc_mem_mb()
        proc_cpu, total_cpu = self._proc_cpu_percent()
        mem_percent = self._mem_percent()
        out: Dict[str, Any] = {
            "ts": now,
            "module": self.module,
            "source": "procfs",
            "process_cpu_percent": proc_cpu,
            "system_cpu_percent": total_cpu,
            "process_rss_mb": rss_mb,
            "process_vms_mb": vms_mb,
            "system_mem_percent": mem_percent,
            "thread_count": self._thread_count(),
        }
        try:
            out["open_fds"] = len(list(Path("/proc/self/fd").iterdir()))
        except Exception:
            pass
        try:
            out["loadavg"] = list(os.getloadavg())
        except Exception:
            pass
        temp = self._thermal_temp_c()
        if temp is not None:
            out["temperature_c"] = temp
        return out

    def _proc_mem_mb(self) -> tuple[Optional[float], Optional[float]]:
        try:
            page_size = float(os.sysconf("SC_PAGE_SIZE"))
            parts = Path("/proc/self/statm").read_text(encoding="utf-8").split()
            vms = float(parts[0]) * page_size / (1024.0 * 1024.0)
            rss = float(parts[1]) * page_size / (1024.0 * 1024.0)
            return rss, vms
        except Exception:
            return None, None

    def _proc_cpu_percent(self) -> tuple[Optional[float], Optional[float]]:
        try:
            proc_ticks = self._read_proc_ticks()
            total_ticks, idle_ticks = self._read_total_idle_ticks()
            now = time.time()
            if self._last_proc_cpu is None or self._last_total_cpu is None:
                self._last_proc_cpu = (now, proc_ticks)
                self._last_total_cpu = (now, total_ticks, idle_ticks)
                return None, None
            prev_t, prev_proc = self._last_proc_cpu
            _prev_total_t, prev_total, prev_idle = self._last_total_cpu
            dt = max(1e-6, now - float(prev_t))
            proc_delta = max(0.0, float(proc_ticks - prev_proc))
            total_delta = max(0.0, float(total_ticks - prev_total))
            idle_delta = max(0.0, float(idle_ticks - prev_idle))
            clk = float(os.sysconf("SC_CLK_TCK"))
            cpu_count = max(1, int(os.cpu_count() or 1))
            proc_pct = 100.0 * proc_delta / max(1e-6, clk * dt)
            sys_pct = 100.0 * max(0.0, total_delta - idle_delta) / max(1.0, total_delta) if total_delta > 0 else None
            self._last_proc_cpu = (now, proc_ticks)
            self._last_total_cpu = (now, total_ticks, idle_ticks)
            return proc_pct, sys_pct
        except Exception:
            return None, None

    @staticmethod
    def _read_proc_ticks() -> int:
        parts = Path("/proc/self/stat").read_text(encoding="utf-8").split()
        return int(parts[13]) + int(parts[14])

    @staticmethod
    def _read_total_idle_ticks() -> tuple[int, int]:
        first = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(v) for v in first]
        idle = values[3] + (values[4] if len(values) > 4 else 0)
        return sum(values), idle

    @staticmethod
    def _mem_percent() -> Optional[float]:
        try:
            values = {}
            for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
                key, raw = line.split(":", 1)
                values[key] = float(raw.strip().split()[0])
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
            if total and available is not None:
                return 100.0 * (1.0 - available / total)
        except Exception:
            pass
        return None

    @staticmethod
    def _thread_count() -> Optional[int]:
        try:
            return len(list(Path("/proc/self/task").iterdir()))
        except Exception:
            return None

    @staticmethod
    def _thermal_temp_c() -> Optional[float]:
        try:
            temps = []
            for path in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
                raw = path.read_text(encoding="utf-8").strip()
                val = float(raw)
                if val > 1000.0:
                    val /= 1000.0
                if -40.0 <= val <= 125.0:
                    temps.append(val)
            return max(temps) if temps else None
        except Exception:
            return None
