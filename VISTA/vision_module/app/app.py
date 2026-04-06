#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import logging
import numpy as np

try:
    import aidcv as cv2
except ImportError:
    import cv2

from ..config.board_config import CONFIG
from ..backend.vision_engine import VisionEngine
from ..utils.detect import compute_target_obs
from ..utils.plot import draw_detect_res_fast
from ..ipc.transport import JsonlInboundServer, JsonlClientSender
from ..ipc.protocol import VisionReq, TargetObs, HomeTagReq, HomeTagObs, now_ts


def setup_logger():
    logger = logging.getLogger("AppLayer")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _safe_mode_text(mode: str) -> str:
    return str(mode or "IDLE").strip().upper()


def _ctx_dict(mode: str, target_name, session_id, req_id, epoch: int):
    return {
        "mode": mode,
        "target": target_name,
        "session_id": session_id,
        "req_id": req_id,
        "epoch": int(epoch),
    }


def _task_key_for(req_mode: str, target, session_id, epoch: int):
    return (req_mode, target, session_id, int(epoch))


def _enter_hot_standby(engine: VisionEngine, log, current_mode: str, target_name, hot_until_ts: float,
                       session_id, req_id, epoch: int):
    engine.set_camera("rgb", True)
    if CONFIG.runtime.keep_model_hot_in_standby:
        engine.set_model(CONFIG.model.active_model, True)
    engine.set_inference_enabled(bool(CONFIG.runtime.enable_infer_during_hot_standby))
    engine.reset_runtime_state()
    new_until = time.time() + float(CONFIG.runtime.hot_standby_s)
    log.info(
        f"🟡 进入视觉热待机 {CONFIG.runtime.hot_standby_s:.0f}s | prev_mode={current_mode} | prev_target={target_name} | hot_until={new_until:.3f}"
    )
    return "IDLE_HOT", None, new_until, None, None, None, int(epoch)


def _enter_cold_idle(engine: VisionEngine, log, epoch: int):
    log.info("💤 热待机超时，进入冷待机并释放视觉资源")
    engine.set_inference_enabled(False)
    if CONFIG.runtime.keep_model_hot_in_standby:
        engine.set_model(CONFIG.model.active_model, False)
    engine.set_camera("rgb", False)
    engine.reset_runtime_state()
    return "IDLE", None, 0.0, None, None, None, int(epoch)


def main():
    log = setup_logger()
    log.info("=========================================")
    log.info("  VISTA 机器人视觉主控 App 启动中...")
    log.info("=========================================")

    engine = VisionEngine(CONFIG, log)

    req_server = JsonlInboundServer(
        mode=CONFIG.req_in.transport, tcp_host=CONFIG.req_in.host,
        tcp_port=CONFIG.req_in.port, uds_path=CONFIG.req_in.uds_path,
        name="req_in", logger=lambda x: log.info(f"[IPC-RX] {x['msg']}")
    )
    obs_sender = JsonlClientSender(
        mode=CONFIG.obs_out.transport, tcp_host=CONFIG.obs_out.host,
        tcp_port=CONFIG.obs_out.port, uds_path=CONFIG.obs_out.uds_path,
        name="obs_out", logger=lambda x: log.info(f"[IPC-TX] {x['msg']}")
    )

    current_mode = "IDLE"
    target_name = None
    current_session_id = None
    current_req_id = None
    current_epoch = 0
    last_send_ts = 0.0
    last_req_receive_ts = 0.0
    hot_until_ts = 0.0
    active_task_key = None

    try:
        req_server.start()
        log.info("🌐 网络监听已启动，等待业务端下发指令...")

        engine.init()
        engine.start()

        if CONFIG.debug.preview:
            cv2.namedWindow("VISTA App Dashboard")

        target_frame_time = 1.0 / max(0.5, CONFIG.runtime.loop_hz)

        while True:
            loop_start = time.time()

            for item in req_server.drain():
                payload = item["payload"]
                typ = str(payload.get("type", "vision_req")).strip()
                last_req_receive_ts = time.time()

                if typ == "vision_req":
                    req = VisionReq.from_dict(payload)
                    new_mode = _safe_mode_text(req.mode)

                    if new_mode == "FIND" and req.target:
                        next_task_key = _task_key_for(new_mode, req.target, req.session_id, req.epoch)
                        same_task = (active_task_key == next_task_key and current_mode == "FIND")
                        if same_task:
                            current_req_id = req.req_id
                            current_session_id = req.session_id
                            current_epoch = int(req.epoch)
                            log.info(f"🔄 刷新寻找目标 [{req.target}] | session={req.session_id} req={req.req_id} epoch={req.epoch}")
                        else:
                            log.info(f"🎯 开始寻找 [{req.target}] | session={req.session_id} req={req.req_id} epoch={req.epoch}")
                            engine.set_camera("rgb", True)
                            engine.set_model(CONFIG.model.active_model, True)
                            engine.set_inference_enabled(True)
                            engine.reset_runtime_state()
                            current_mode = "FIND"
                            target_name = req.target
                            current_session_id = req.session_id
                            current_req_id = req.req_id
                            current_epoch = int(req.epoch)
                            active_task_key = next_task_key
                            hot_until_ts = 0.0

                    elif new_mode in ["IDLE", "STOP", "CANCEL"]:
                        if CONFIG.runtime.keep_preview_after_stop and float(CONFIG.runtime.hot_standby_s) > 0.0:
                            current_mode, target_name, hot_until_ts, active_task_key, current_session_id, current_req_id, current_epoch = _enter_hot_standby(
                                engine, log, current_mode, target_name, hot_until_ts, current_session_id, current_req_id, current_epoch
                            )
                        else:
                            log.info(f"💤 业务指令: 进入待机状态 [{new_mode}]，销毁硬件释放功耗")
                            current_mode, target_name, hot_until_ts, active_task_key, current_session_id, current_req_id, current_epoch = _enter_cold_idle(engine, log, current_epoch)

                elif typ == "home_tag_req":
                    req = HomeTagReq.from_dict(payload)
                    log.info(f"🏠 收到返航请求 | session={req.session_id} req={req.req_id} epoch={req.epoch}")
                    engine.set_camera("rgb", True)
                    if CONFIG.runtime.keep_model_hot_in_standby:
                        engine.set_model(CONFIG.model.active_model, True)
                    engine.set_inference_enabled(False)
                    engine.reset_runtime_state()
                    current_mode = "RETURN"
                    target_name = None
                    current_session_id = req.session_id
                    current_req_id = req.req_id
                    current_epoch = int(req.epoch)
                    active_task_key = _task_key_for("RETURN", None, req.session_id, req.epoch)
                    hot_until_ts = 0.0

            frames, infer_res = engine.get_new_data()
            rgb_raw = frames.get("rgb") if frames else None
            now = time.time()

            if current_mode == "FIND" and target_name:
                if now - last_send_ts >= 1.0 / max(0.5, CONFIG.runtime.send_hz):
                    obs = None
                    if rgb_raw is not None and infer_res is not None:
                        obs = compute_target_obs(rgb_raw.shape, target_name, infer_res.get("boxes", []))
                    if obs is None:
                        out_payload = TargetObs(
                            ts=now_ts(), found=False, target=target_name,
                            session_id=current_session_id, req_id=current_req_id, epoch=current_epoch,
                        ).to_dict()
                    else:
                        out_payload = TargetObs(
                            ts=now_ts(), found=True, session_id=current_session_id,
                            req_id=current_req_id, epoch=current_epoch, **obs,
                        ).to_dict()
                    queued = obs_sender.send(out_payload)
                    if not queued:
                        log.warning("⚠️ obs_out 队列繁忙，本次观测未入队")
                    last_send_ts = now

            elif current_mode == "RETURN":
                if now - last_send_ts >= 1.0 / max(0.5, CONFIG.runtime.send_hz):
                    out_payload = HomeTagObs(
                        ts=now_ts(), found=False, session_id=current_session_id,
                        req_id=current_req_id, epoch=current_epoch,
                    ).to_dict()
                    obs_sender.send(out_payload)
                    last_send_ts = now

            # if current_mode in {"FIND", "RETURN"} and last_req_receive_ts > 0:
            #     if (now - last_req_receive_ts) > float(CONFIG.runtime.stale_req_s):
            #         if CONFIG.runtime.keep_preview_after_stop and float(CONFIG.runtime.hot_standby_s) > 0.0:
            #             current_mode, target_name, hot_until_ts, active_task_key, current_session_id, current_req_id, current_epoch = _enter_hot_standby(
            #                 engine, log, current_mode, target_name, hot_until_ts, current_session_id, current_req_id, current_epoch
            #             )
            #         else:
            #             current_mode, target_name, hot_until_ts, active_task_key, current_session_id, current_req_id, current_epoch = _enter_cold_idle(engine, log, current_epoch)

            if current_mode == "IDLE_HOT" and hot_until_ts > 0 and now >= hot_until_ts:
                current_mode, target_name, hot_until_ts, active_task_key, current_session_id, current_req_id, current_epoch = _enter_cold_idle(engine, log, current_epoch)

            if CONFIG.debug.preview:
                if current_mode in {"FIND", "RETURN", "IDLE_HOT"} and rgb_raw is not None:
                    bgr_canvas = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                    if CONFIG.debug.draw_boxes and infer_res is not None:
                        bgr_canvas = draw_detect_res_fast(bgr_canvas, infer_res.get("boxes", []), infer_res.get("masks", []))
                    cv2.putText(bgr_canvas, f"Mode: {current_mode}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    if current_mode == "IDLE_HOT":
                        remain = max(0.0, hot_until_ts - now)
                        cv2.putText(bgr_canvas, f"Hot standby: {remain:4.1f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    else:
                        cv2.putText(bgr_canvas, f"AI Sync: {'ON' if infer_res else 'OFF'}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    if current_session_id:
                        cv2.putText(bgr_canvas, f"session={current_session_id} epoch={current_epoch}", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.imshow("VISTA App Dashboard", bgr_canvas)
                else:
                    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(canvas, "SYSTEM STANDBY", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (150, 150, 150), 2)
                    cv2.putText(canvas, "Zero Power Mode Active", (170, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 1)
                    cv2.imshow("VISTA App Dashboard", canvas)

                if cv2.waitKey(1) & 0xFF == 27:
                    log.info("退出预览面板")
                    break

            dt = time.time() - loop_start
            if dt < target_frame_time:
                time.sleep(target_frame_time - dt)

    except KeyboardInterrupt:
        log.info("🛑 收到用户终端退出信号")
    except Exception as e:
        log.error(f"❌ App 主控崩溃: {e}", exc_info=True)
    finally:
        req_server.close()
        obs_sender.close()
        log.info("清理引擎实例...")
        engine.stop()
        log.info("✅ 视觉主控服务已完全关闭。")

if __name__ == "__main__":
    main()
