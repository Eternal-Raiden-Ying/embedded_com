from types import SimpleNamespace

import numpy as np
import pytest

from voice_service.runtime.kws_engine import KwsBackendFatalError, SherpaOnnxKwsBackend


MODEL_FILES = (
    "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
    "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    "tokens.txt",
)


class FakeStream:
    def __init__(self):
        self.accepted = []

    def accept_waveform(self, sample_rate, audio):
        self.accepted.append((sample_rate, audio))


class FakeSpotter:
    def __init__(self, result="你好小车", **kwargs):
        self.kwargs = kwargs
        self.result = result
        self.ready_calls = 0
        self.decode_calls = 0
        self.reset_calls = 0
        self.stream = FakeStream()

    def create_stream(self):
        return self.stream

    def is_ready(self, _stream):
        self.ready_calls += 1
        return self.ready_calls == 1

    def decode_stream(self, _stream):
        self.decode_calls += 1

    def get_result(self, _stream):
        result, self.result = self.result, ""
        return result

    def reset_stream(self, _stream):
        self.reset_calls += 1


def make_cfg(tmp_path, max_errors=3):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    for name in MODEL_FILES:
        (model_dir / name).touch()
    keywords = tmp_path / "extreme.txt"
    keywords.write_text("n i :3.0 #0.08 @你好小车\n", encoding="utf-8")
    return SimpleNamespace(
        model_dir=str(model_dir), keywords_file=str(keywords), provider="cpu",
        num_threads=1, max_active_paths=4, num_trailing_blanks=1,
        max_consecutive_errors=max_errors,
    )


def test_sherpa_backend_feeds_float32_decodes_and_resets_hit(tmp_path):
    created = []

    def factory(**kwargs):
        spotter = FakeSpotter(**kwargs)
        created.append(spotter)
        return spotter

    backend = SherpaOnnxKwsBackend(make_cfg(tmp_path), spotter_factory=factory)
    keyword = backend.process(np.ones(1280, dtype=np.float32) * 0.25)

    assert keyword == "你好小车"
    assert created[0].decode_calls == 1
    assert created[0].reset_calls == 1
    sample_rate, audio = created[0].stream.accepted[0]
    assert sample_rate == 16000
    assert audio.dtype == np.float32
    assert audio.flags.c_contiguous
    assert "keywords_score" not in created[0].kwargs
    assert "keywords_threshold" not in created[0].kwargs


def test_consecutive_decode_errors_become_fatal(tmp_path):
    class BrokenStream(FakeStream):
        def accept_waveform(self, sample_rate, audio):
            raise RuntimeError("decode failed")

    class BrokenSpotter(FakeSpotter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.stream = BrokenStream()

    backend = SherpaOnnxKwsBackend(make_cfg(tmp_path, max_errors=2), spotter_factory=BrokenSpotter)
    assert backend.process(np.zeros(1280, dtype=np.float32)) == ""
    with pytest.raises(KwsBackendFatalError):
        backend.process(np.zeros(1280, dtype=np.float32))
