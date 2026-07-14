#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
连续语音链路诊断：Sherpa KWS -> FSMN-VAD-online -> Online Paraformer
不连接 Orchestrator，不控制机器人。

注意：
- Sherpa KWS Python API不返回校准触发概率，只显示关键词文件中的 score/threshold。
- VAD/ASR“分数”为诊断代理分，用于比较距离、方向、环境，不是模型概率。
"""

from __future__ import annotations
import argparse, csv, math, re, signal, subprocess, sys, time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import sherpa_onnx
from funasr_onnx import Fsmn_vad_online
from funasr_onnx.paraformer_online_bin import Paraformer as OnlineParaformer

try:
    import yaml
except Exception:
    yaml = None

ROOT = Path("/home/aidlux/embedded_com")
SR = 16000

# These are deployment starting points, not model-probability calibrations.
VAD_PROFILES = {
    "normal": {
        "speech_noise_thres": 0.60,
        "max_end_sil": 800,
        "pre_ms": 400,
        "no_speech_ms": 8000,
    },
    "sensitive": {
        "speech_noise_thres": 0.45,
        "max_end_sil": 900,
        "pre_ms": 500,
        "no_speech_ms": 9000,
    },
    "extreme": {
        "speech_noise_thres": 0.30,
        "max_end_sil": 1100,
        "pre_ms": 600,
        "no_speech_ms": 10000,
    },
}

ASR_PROFILES = {
    "fast": [5, 8, 4],       # about 480 ms central chunk
    "balanced": [5, 10, 5], # funasr_onnx online default, about 600 ms
    "accuracy": [5, 12, 6], # more right context, about 720 ms; validate latency
}
CLEAR, HIDE, SHOW = "\033[2J\033[H", "\033[?25l", "\033[?25h"


def ms() -> float:
    return time.monotonic() * 1000.0


def norm(s: Any) -> str:
    return re.sub(r"[\s，。！？、,.!?；;：:（）()\[\]{}“”\"'`]+", "", str(s or "").strip())


def merge_text(old: str, new: str) -> str:
    old, new = norm(old), norm(new)
    if not new:
        return old
    if not old:
        return new
    if new.startswith(old):
        return new
    if old.startswith(new) or new in old:
        return old
    if old in new:
        return new
    for n in range(min(len(old), len(new)), 0, -1):
        if old[-n:] == new[:n]:
            return old + new[n:]
    return old + new


def parse_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return norm(x)
    if isinstance(x, dict):
        for k in ("text", "preds", "sentence", "result", "value", "output"):
            if k in x:
                t = parse_text(x[k])
                if t:
                    return t
        for v in x.values():
            t = parse_text(v)
            if t:
                return t
        return ""
    if isinstance(x, (list, tuple)):
        for v in x:
            t = parse_text(v)
            if t:
                return t
    return ""


def pairs(x: Any) -> Iterable[Tuple[float, float]]:
    if isinstance(x, dict):
        for v in x.values():
            yield from pairs(v)
    elif isinstance(x, (list, tuple, np.ndarray)):
        a = list(x)
        if len(a) >= 2 and all(isinstance(v, (int, float, np.integer, np.floating)) for v in a[:2]):
            yield float(a[0]), float(a[1])
        else:
            for v in a:
                yield from pairs(v)


def rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    y = x.astype(np.float64)
    return float(np.sqrt(np.mean(y * y)))


def dbfs(v: float) -> float:
    return -120.0 if v <= 1e-9 else 20.0 * math.log10(v / 32768.0)


def score(v: float) -> int:
    return int(round(max(0.0, min(100.0, v))))


def bar(v: float, vmax: float = 100, width: int = 28) -> str:
    n = int(round(width * max(0.0, min(1.0, v / max(vmax, 1e-9)))))
    return "█" * n + "·" * (width - n)


def keyword_params(path: Path) -> Dict[str, Tuple[float, float]]:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = re.search(r"@(\S+)", line)
        if not m:
            continue
        sm, tm = re.search(r":([0-9.]+)", line), re.search(r"#([0-9.]+)", line)
        out[m.group(1).replace("_", " ")] = (
            float(sm.group(1)) if sm else 1.0,
            float(tm.group(1)) if tm else 0.25,
        )
    return out


def target_aliases(path: Path) -> Tuple[List[Tuple[str, str]], Dict[str, str], Dict[str, bool]]:
    if yaml is None or not path.is_file():
        return [], {}, {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = data.get("targets", data)
    aliases, names, executable = [], {}, {}
    if not isinstance(items, dict):
        return aliases, names, executable
    for canonical, spec in items.items():
        if not isinstance(spec, dict):
            continue
        names[canonical] = str(spec.get("display_name") or canonical)
        policy = str(spec.get("action_policy") or "")
        status = str(spec.get("support_status") or "")
        executable[canonical] = (
            (policy == "fixed_grasp" and status == "ready")
            or policy == "locate_and_ring"
        )
        if spec.get("selectable", True) is False:
            continue
        if policy in ("docking_only", "place_destination"):
            continue
        for a in list(spec.get("aliases") or []) + [names[canonical], canonical]:
            a = norm(a)
            if a:
                aliases.append((a, canonical))
    aliases.sort(key=lambda z: len(z[0]), reverse=True)
    return aliases, names, executable


def interpret(
    text: str,
    aliases: Sequence[Tuple[str, str]],
    executable: Dict[str, bool],
) -> Tuple[str, str, bool, str]:
    t = norm(text)
    if not t:
        return "EMPTY", "", False, "ASR为空"
    if any(w in t for w in ("停止小车", "停止", "停下", "取消任务", "不要了")):
        return "STOP", "", True, "停止指令"
    for a, c in aliases:
        if a in t:
            ok = bool(executable.get(c, False))
            return (
                "FIND",
                c,
                ok,
                "目标可执行" if ok else "目标已识别但当前未验证执行",
            )
    return "REJECT", "", False, "未命中支持目标"


def make_vad(
    model_dir: Path,
    quant: bool,
    threads: int,
    max_end_sil: int,
    speech_noise_thres: float,
):
    errors = []
    for kw in (
        {
            "quantize": quant,
            "intra_op_num_threads": threads,
            "max_end_sil": max_end_sil,
        },
        {"quantize": quant, "max_end_sil": max_end_sil},
        {"max_end_sil": max_end_sil},
    ):
        try:
            vad = Fsmn_vad_online(str(model_dir), **kw)
            # Support both funasr_onnx 0.4.1 and newer source layouts.
            if hasattr(vad, "vad_scorer"):
                vad.vad_scorer.vad_opts.speech_noise_thres = speech_noise_thres
                vad.vad_scorer.speech_noise_thres = speech_noise_thres
            elif hasattr(vad, "config"):
                vad.config["model_conf"]["speech_noise_thres"] = speech_noise_thres
            vad.max_end_sil = max_end_sil
            return vad
        except TypeError as e:
            errors.append(str(e))
    raise RuntimeError("FSMN-VAD初始化失败: " + " | ".join(errors))


def make_asr(model_dir: Path, quant: bool, threads: int, chunk: List[int]):
    errors = []
    for kw in (
        {"batch_size": 1, "quantize": quant, "intra_op_num_threads": threads, "chunk_size": chunk},
        {"batch_size": 1, "quantize": quant, "intra_op_num_threads": threads},
        {"batch_size": 1, "quantize": quant},
        {"quantize": quant},
        {},
    ):
        try:
            return OnlineParaformer(str(model_dir), **kw)
        except TypeError as e:
            errors.append(str(e))
    raise RuntimeError("Online ASR初始化失败: " + " | ".join(errors))


@dataclass
class Session:
    sid: int
    wake: str
    opened: float
    vad_kw: Dict[str, Any] = field(default_factory=lambda: {"in_cache": [], "is_final": False})
    asr_kw: Dict[str, Any] = field(default_factory=lambda: {"cache": {}, "is_final": False})
    pre: Deque[np.ndarray] = field(default_factory=deque)
    pre_n: int = 0
    started: bool = False
    ended: bool = False
    t0: float = 0.0
    t1: float = 0.0
    end_source: str = ""
    buf: np.ndarray = field(default_factory=lambda: np.empty(0, np.float32))
    merged: str = ""
    raw_partial: str = ""
    raw_final: str = ""
    chunks: int = 0
    infer_ms: float = 0.0
    final_once: bool = False
    noise: List[float] = field(default_factory=list)
    voice: List[float] = field(default_factory=list)


class App:
    def __init__(self, a):
        self.a = a
        self.kws = sherpa_onnx.KeywordSpotter(
            tokens=str(a.kws_dir / "tokens.txt"),
            encoder=str(a.kws_dir / "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx"),
            decoder=str(a.kws_dir / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx"),
            joiner=str(a.kws_dir / "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx"),
            keywords_file=str(a.keywords),
            provider="cpu", num_threads=a.threads,
            max_active_paths=a.max_paths,
            num_trailing_blanks=a.trailing_blanks,
            keywords_score=1.0, keywords_threshold=0.25,
        )
        self.ks = self.kws.create_stream()
        self.vad = make_vad(
            a.vad_dir,
            a.vad_quant,
            a.threads,
            a.vad_max_end_sil,
            a.vad_speech_noise_thres,
        )
        self.asr = make_asr(a.asr_dir, a.asr_quant, a.threads, a.chunk)
        self.kp = keyword_params(a.keywords)
        self.aliases, self.names, self.executable_targets = target_aliases(a.catalog)

        self.running, self.proc = True, None
        self.state, self.msg = "WAIT_WAKE", "请说唤醒词"
        self.s: Optional[Session] = None
        self.sid = 0
        self.rms, self.db = 0.0, -120.0
        self.last_kw = self.partial = self.final = self.intent = self.target = self.verdict = ""
        self.vad_score = self.asr_score = self.exec_score = 0
        self.infer_ms = self.snr = 0.0
        self.wakes = self.stops = self.vstarts = self.vends = self.finals = self.ready = self.timeouts = 0
        self.history = deque(maxlen=6)
        self.hold_until = 0.0
        self.started_at = ms()
        self.last_draw = 0.0

        a.csv.parent.mkdir(parents=True, exist_ok=True)
        self.cf = a.csv.open("a", newline="", encoding="utf-8")
        self.cw = csv.writer(self.cf)
        if a.csv.stat().st_size == 0:
            self.cw.writerow(["time","sid","wake","text","intent","target","verdict","vad_start","vad_end","end_source","speech_ms","snr_db","chunks","infer_ms","vad_proxy","asr_proxy","execution_proxy"])

    def start_audio(self):
        self.proc = subprocess.Popen(
            ["arecord","-q","-D",self.a.device,"-f","S16_LE","-c","1","-r",str(SR),"-t","raw"],
            stdout=subprocess.PIPE, bufsize=0
        )

    def readn(self, n):
        out = bytearray()
        while len(out) < n and self.running:
            b = self.proc.stdout.read(n - len(out))
            if not b: break
            out.extend(b)
        return bytes(out)

    def reset_kws(self):
        try: self.kws.reset_stream(self.ks)
        except Exception: self.ks = self.kws.create_stream()

    def feed_kws(self, x):
        self.ks.accept_waveform(SR, x)
        while self.kws.is_ready(self.ks):
            self.kws.decode_stream(self.ks)
        r = self.kws.get_result(self.ks)
        if r:
            self.reset_kws()
            return str(r).strip()
        return ""

    def open_session(self, kw):
        self.sid += 1
        self.s = Session(self.sid, kw, ms())
        self.state, self.msg = "COMMAND_WAIT", "已唤醒，请说取物指令"
        self.partial = self.final = self.intent = self.target = ""
        self.verdict, self.exec_score = "等待VAD起点", 20

    def add_pre(self, x):
        lim = int(SR * self.a.pre_ms / 1000)
        self.s.pre.append(x.copy()); self.s.pre_n += x.size
        while self.s.pre and self.s.pre_n > lim:
            y = self.s.pre.popleft(); self.s.pre_n -= y.size

    def vad_feed(self, x, final=False):
        self.s.vad_kw["is_final"] = final
        return self.vad(audio_in=x, param_dict=self.s.vad_kw)

    def asr_feed(self, x, final=False):
        self.s.asr_kw.update({
            "is_final": final,
            "chunk_size": self.a.chunk,
            "encoder_chunk_look_back": self.a.enc_lb,
            "decoder_chunk_look_back": self.a.dec_lb,
        })
        t = ms()
        r = self.asr(audio_in=x.astype(np.float32, copy=False), param_dict=self.s.asr_kw)
        self.s.infer_ms += ms() - t
        return parse_text(r)

    def flush(self, final=False):
        step = self.a.step
        while self.s.buf.size >= step:
            x, self.s.buf = self.s.buf[:step], self.s.buf[step:]
            raw = self.asr_feed(x, False)
            self.s.chunks += 1
            self.s.raw_partial = raw
            self.s.merged = merge_text(self.s.merged, raw)
            self.partial = self.s.merged
        if final:
            x, self.s.buf = self.s.buf, np.empty(0, np.float32)
            if x.size == 0: x = np.zeros(1600, np.float32)
            raw = self.asr_feed(x, True)
            self.s.raw_final = raw
            self.s.merged = merge_text(self.s.merged, raw)
        return self.s.merged

    def vad_start(self):
        if self.s.started: return
        self.s.started, self.s.t0 = True, ms()
        self.vstarts += 1
        self.state, self.msg = "SPEECH", "VAD已起声，正在流式识别"
        self.exec_score = 40
        if self.s.pre:
            self.s.buf = np.concatenate(list(self.s.pre)).astype(np.float32, copy=False)

    def vad_end(self, source):
        if not self.s.started or self.s.ended: return
        self.s.ended, self.s.t1, self.s.end_source = True, ms(), source
        self.vends += 1
        self.finalize()

    def handle_vad(self, r):
        for st, ed in pairs(r):
            if st >= 0 and not self.s.started: self.vad_start()
            if ed >= 0 and self.s.started: self.vad_end("fsmn_vad")

    def calc_scores(self, text, supported, intent):
        s = self.s
        nr = float(np.mean(s.noise)) if s.noise else 1.0
        vr = float(np.mean(s.voice)) if s.voice else 0.0
        snr = 20 * math.log10(max(vr,1.0)/max(nr,1.0))
        dur = max(0.0, s.t1-s.t0) if s.t0 and s.t1 else 0.0
        v = (20 if s.started else 0)+(20 if s.ended else 0)+max(0,min(40,(snr+3)*2))+(20 if 250<=dur<=5000 else 8 if dur>=120 else 0)
        a = (45 if text else 0)+(15 if s.chunks else 0)+(20 if supported else 8 if text else 0)+(20 if s.merged and text and (s.merged in text or text in s.merged) else 0)
        e = 20+(20 if s.started else 0)+(15 if s.ended else 0)+(20 if text else 0)+(10 if intent not in ("EMPTY","REJECT") else 0)+(15 if supported else 0)
        return score(v), score(a), score(e), snr

    def finalize(self):
        if self.s.final_once: return
        self.s.final_once = True
        self.state, self.msg = "FINALIZING", "生成最终识别结果"
        try: text = self.flush(True)
        except Exception as ex:
            text = self.s.merged
            self.verdict = f"FINAL异常:{ex}"
        intent, target, supported, reason = interpret(
            text, self.aliases, self.executable_targets
        )
        verdict = "EXECUTE_READY" if supported and intent in ("FIND","STOP") else "REJECT"
        vs, ass, es, snr = self.calc_scores(text, supported, intent)
        speech_ms = int(max(0, self.s.t1-self.s.t0)) if self.s.t0 and self.s.t1 else 0

        self.final, self.intent, self.target = text, intent, target
        self.verdict = f"{verdict}:{reason}"
        self.vad_score, self.asr_score, self.exec_score = vs, ass, es
        self.infer_ms, self.snr = self.s.infer_ms, snr
        self.finals += bool(text); self.ready += verdict == "EXECUTE_READY"
        stamp = datetime.now().isoformat(timespec="seconds")
        self.history.appendleft((stamp[11:19], text, intent, target, verdict, vs, ass, es, snr))
        self.cw.writerow([stamp,self.s.sid,self.s.wake,text,intent,target,verdict,int(self.s.started),int(self.s.ended),self.s.end_source,speech_ms,f"{snr:.2f}",self.s.chunks,f"{self.s.infer_ms:.2f}",vs,ass,es]); self.cf.flush()
        self.state, self.msg = "RESULT", "结果完成，自动返回等待唤醒"
        self.hold_until = ms()+self.a.hold_ms

    def reset_wait(self):
        self.s = None
        self.state, self.msg = "WAIT_WAKE", "请说唤醒词"
        self.hold_until = 0; self.reset_kws()

    def keyword(self, kw):
        self.last_kw = kw
        if kw == self.a.stop_word:
            self.stops += 1
            self.s = None
            self.state, self.msg = "RESULT", "停止词触发"
            self.final, self.intent, self.target = kw, "STOP", ""
            self.verdict, self.exec_score = "STOP KWS触发", 100
            self.hold_until = ms()+self.a.hold_ms
        elif kw == self.a.wake_word and self.state == "WAIT_WAKE":
            self.wakes += 1; self.open_session(kw)

    def command_frame(self, x, rv):
        if not self.s: return
        now = ms()
        if not self.s.started:
            self.s.noise.append(rv); self.add_pre(x)
        try: self.handle_vad(self.vad_feed(x))
        except Exception as ex:
            self.msg = f"VAD异常:{ex}"; return
        if self.s and self.s.started and not self.s.ended:
            self.s.voice.append(rv)
            self.s.buf = np.concatenate([self.s.buf,x])
            try: self.flush(False)
            except Exception as ex: self.msg = f"ASR CHUNK异常:{ex}"
            if now-self.s.t0 >= self.a.max_speech_ms: self.vad_end("max_speech_timeout")
        elif self.s and not self.s.started and now-self.s.opened >= self.a.no_speech_ms:
            self.timeouts += 1
            self.verdict, self.vad_score, self.asr_score, self.exec_score = "NO_SPEECH_TIMEOUT",0,0,20
            self.state, self.msg = "RESULT", "监听窗口未检测到语音"
            self.hold_until = now+self.a.hold_ms

    def draw(self):
        if ms()-self.last_draw < 180: return
        self.last_draw = ms()
        def kp(w):
            s,t = self.kp.get(w,(float("nan"),float("nan")))
            return f"{w}: score={s:g} threshold={t:g}"
        target = self.names.get(self.target,self.target)
        out = [
            CLEAR, "语音链路连续诊断 | Ctrl+C结束", "─"*72,
            f"状态 {self.state:<13} {self.msg}",
            f"音频 RMS={self.rms:7.1f} dBFS={self.db:6.1f} [{bar(max(0,self.db+60),60)}]",
            f"设备 {self.a.device} / 16kHz / {self.a.frame_ms}ms",
            (
                f"配置 VAD={self.a.vad_profile} "
                f"speech_noise={self.a.vad_speech_noise_thres:.2f} "
                f"end_sil={self.a.vad_max_end_sil}ms | "
                f"ASR={self.a.asr_profile} chunk={self.a.chunk}"
            ),
            "", "KWS  "+kp(self.a.wake_word)+" | "+kp(self.a.stop_word),
            f"     最近={self.last_kw or '-'} WAKE={self.wakes} STOP={self.stops}",
            "", f"VAD  start/end={self.vstarts}/{self.vends} timeout={self.timeouts} 代理分={self.vad_score}",
            f"ASR  partial={self.partial or '-'}",
            f"     final={self.final or '-'} latency={self.infer_ms:.1f}ms 代理分={self.asr_score}",
            f"命令 intent={self.intent or '-'} target={target or '-'} 判定={self.verdict or '-'}",
            "", f"执行可用度代理分 {self.exec_score:3d}/100 [{bar(self.exec_score,100,36)}]",
            f"统计 ASR final={self.finals} 可执行={self.ready} CSV={self.a.csv}",
            "", "最近结果："
        ]
        if not self.history: out.append("  暂无")
        for h in self.history:
            tm,txt,it,tg,vd,vs,ass,es,sn = h
            out.append(f"  {tm} “{txt or '-'}” {it}/{self.names.get(tg,tg) or '-'} {vd} V{vs} A{ass} E{es} SNR={sn:.1f}dB")
        out += ["", "注：代理分用于相对比较，不是模型置信概率。", "操作：说“你好小车”，短暂停顿后说指令；“停止小车”可随时触发。"]
        sys.stdout.write("\n".join(out)+"\n"); sys.stdout.flush()

    def run(self):
        self.start_audio()
        n = int(SR*self.a.frame_ms/1000); nb=n*2
        sys.stdout.write(HIDE); sys.stdout.flush()
        try:
            while self.running:
                b=self.readn(nb)
                if len(b)!=nb: raise RuntimeError(f"arecord断流 {len(b)}/{nb}")
                i=np.frombuffer(b,dtype="<i2"); x=i.astype(np.float32)/32768.0
                self.rms=rms(i); self.db=dbfs(self.rms)
                kw=self.feed_kws(x)
                if kw: self.keyword(kw)
                if self.state in ("COMMAND_WAIT","SPEECH"): self.command_frame(x,self.rms)
                if self.state=="RESULT" and self.hold_until and ms()>=self.hold_until: self.reset_wait()
                self.draw()
        except KeyboardInterrupt: pass
        finally: self.close()

    def close(self):
        self.running=False
        if self.proc:
            self.proc.terminate()
            try:self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:self.proc.kill()
        self.cf.flush(); self.cf.close()
        sys.stdout.write(SHOW); sys.stdout.flush()
        print(f"\n结束：WAKE={self.wakes} STOP={self.stops} VAD={self.vstarts}/{self.vends} ASR={self.finals} READY={self.ready}")
        print(f"CSV: {self.a.csv}")


def chunk_type(s):
    v=[int(x) for x in s.split(",")]
    if len(v)!=3: raise argparse.ArgumentTypeError("格式应为5,10,5")
    return v


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="plughw:1,0")
    p.add_argument("--frame-ms", type=int, default=100)
    p.add_argument("--threads", type=int, default=1)

    p.add_argument(
        "--kws-dir",
        type=Path,
        default=ROOT / "Voice/kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20",
    )
    p.add_argument(
        "--keywords",
        type=Path,
        default=ROOT / "Voice/kws/sherpa_custom/keywords.txt",
    )
    p.add_argument("--wake-word", default="你好小车")
    p.add_argument("--stop-word", default="停止小车")
    p.add_argument("--max-paths", type=int, default=4)
    p.add_argument("--trailing-blanks", type=int, default=1)

    p.add_argument(
        "--vad-dir",
        type=Path,
        default=ROOT / "Voice/ONNX/speech_fsmn_vad_zh-cn-16k-common-onnx",
    )
    p.add_argument(
        "--vad-profile",
        choices=sorted(VAD_PROFILES),
        default="sensitive",
    )
    p.add_argument("--vad-speech-noise-thres", type=float, default=None)
    p.add_argument("--vad-max-end-sil", type=int, default=None)
    p.add_argument("--vad-quant", dest="vad_quant", action="store_true")
    p.add_argument("--no-vad-quant", dest="vad_quant", action="store_false")
    p.set_defaults(vad_quant=True)

    p.add_argument(
        "--asr-dir",
        type=Path,
        default=ROOT / "Voice/ONNX/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online-onnx",
    )
    p.add_argument(
        "--asr-profile",
        choices=sorted(ASR_PROFILES),
        default="balanced",
    )
    p.add_argument("--asr-quant", dest="asr_quant", action="store_true")
    p.add_argument("--no-asr-quant", dest="asr_quant", action="store_false")
    p.set_defaults(asr_quant=True)
    p.add_argument("--chunk", type=chunk_type, default=None)

    # Kept for compatibility with the project interface. The installed
    # funasr_onnx 0.4.1 online wrapper does not consume these two fields.
    p.add_argument("--enc-lb", type=int, default=4)
    p.add_argument("--dec-lb", type=int, default=1)

    p.add_argument("--pre-ms", type=int, default=None)
    p.add_argument("--no-speech-ms", type=int, default=None)
    p.add_argument("--max-speech-ms", type=int, default=8000)
    p.add_argument("--hold-ms", type=int, default=2500)
    p.add_argument(
        "--catalog",
        type=Path,
        default=ROOT / "configs/target_catalog.yaml",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=ROOT / f"logs/voice_probe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    )

    a = p.parse_args()

    vp = VAD_PROFILES[a.vad_profile]
    if a.vad_speech_noise_thres is None:
        a.vad_speech_noise_thres = vp["speech_noise_thres"]
    if a.vad_max_end_sil is None:
        a.vad_max_end_sil = vp["max_end_sil"]
    if a.pre_ms is None:
        a.pre_ms = vp["pre_ms"]
    if a.no_speech_ms is None:
        a.no_speech_ms = vp["no_speech_ms"]

    if a.chunk is None:
        a.chunk = list(ASR_PROFILES[a.asr_profile])
    a.step = a.chunk[1] * 960

    if not 0.0 <= a.vad_speech_noise_thres <= 1.5:
        p.error("--vad-speech-noise-thres超出合理范围")
    if a.vad_max_end_sil < 100:
        p.error("--vad-max-end-sil不能小于100ms")

    required = [
        a.keywords,
        a.vad_dir,
        a.asr_dir,
        a.kws_dir / "tokens.txt",
        a.kws_dir / "encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
        a.kws_dir / "decoder-epoch-13-avg-2-chunk-16-left-64.onnx",
        a.kws_dir / "joiner-epoch-13-avg-2-chunk-16-left-64.int8.onnx",
    ]
    miss = [str(x) for x in required if not x.exists()]
    if miss:
        p.error("缺少:\n" + "\n".join(miss))
    return a


def main():
    a=args()
    print("加载 KWS / VAD / Online ASR，请稍候……")
    app=App(a)
    def stop(*_): app.running=False
    signal.signal(signal.SIGINT,stop); signal.signal(signal.SIGTERM,stop)
    app.run()


if __name__=="__main__":
    main()
