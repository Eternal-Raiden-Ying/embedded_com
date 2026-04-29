#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.runtime_logging import OperatorConsole, RunLogger  # noqa: E402


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _TTY:
    def isatty(self) -> bool:
        return True


class OperatorConsoleColorTest(unittest.TestCase):
    def test_color_never_has_no_ansi(self) -> None:
        lines = []
        with patch.dict(os.environ, {"ROBOT_CONSOLE_COLOR": "never"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("FORCE_COLOR", None)
            console = OperatorConsole(mode="operator", sink=lines.append, stream=_TTY())
            console.emit("[ORCH] STATE IDLE -> SEARCH_TABLE reason=start")
        self.assertEqual(len(lines), 1)
        self.assertIsNone(ANSI_RE.search(lines[0]))

    def test_color_always_has_ansi(self) -> None:
        lines = []
        with patch.dict(os.environ, {"ROBOT_CONSOLE_COLOR": "always"}, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("FORCE_COLOR", None)
            console = OperatorConsole(mode="operator", sink=lines.append)
            console.emit("[VISTA] TARGET mode=TRACK_LOCAL target=apple found=0")
        self.assertEqual(len(lines), 1)
        self.assertRegex(lines[0], ANSI_RE)

    def test_no_color_overrides_force_color(self) -> None:
        lines = []
        with patch.dict(os.environ, {"NO_COLOR": "1", "FORCE_COLOR": "1", "ROBOT_CONSOLE_COLOR": "always"}, clear=False):
            console = OperatorConsole(mode="operator", sink=lines.append, stream=_TTY())
            console.emit("[ORCH] ERROR vision_obs invalid_json")
        self.assertEqual(len(lines), 1)
        self.assertIsNone(ANSI_RE.search(lines[0]))

    def test_jsonl_writer_is_not_colorized(self) -> None:
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp:
            logger = RunLogger("test", tmp, stack_run_id="run_color_test", enable_text_events=False)
            logger.write_jsonl("event", {"event": "[ORCH] ERROR target", "message": "[VISTA] TARGET found=1"})
            path = Path(tmp) / "run_color_test" / "event.jsonl"
            raw = path.read_text(encoding="utf-8")
            self.assertIsNone(ANSI_RE.search(raw))
            payload = json.loads(raw)
            self.assertEqual(payload["event"], "[ORCH] ERROR target")


if __name__ == "__main__":
    unittest.main()
