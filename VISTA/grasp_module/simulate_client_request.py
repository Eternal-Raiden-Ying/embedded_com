import json
import logging
import os
import time

try:
    import requests
except ImportError as exc:
    raise SystemExit("requests is required to run this script") from exc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

import sys
sys.path.append(PARENT_DIR)

from config.logging_config import configure_grasp_logger
from config.predictor_config import build_predictor_arg_parser


logger = logging.getLogger("vision.grasp")


def post_json(session, url, **kwargs):
    response = session.post(url, **kwargs)
    response.raise_for_status()
    return response.json()


def build_metadata(cfgs):
    metadata = {
        "robot_id": cfgs.robot_id,
        "cmd": cfgs.cmd,
    }
    if cfgs.class_id is not None:
        metadata["class_id"] = cfgs.class_id
    return metadata


def build_request_payload(cfgs):
    if cfgs.class_id is None:
        raise ValueError('class_id is required for the current remote contract')
    data = {
        'metadata': json.dumps(build_metadata(cfgs), ensure_ascii=False),
        'class_id': str(cfgs.class_id),
    }
    return data


def build_request_files(cfgs):
    return {
        'rgb_file': (os.path.basename(cfgs.rgb_path), open(cfgs.rgb_path, 'rb'), 'image/png'),
        'depth_file': (os.path.basename(cfgs.depth_path), open(cfgs.depth_path, 'rb'), 'image/png'),
    }


def close_request_files(files):
    for _, handle, _ in files.values():
        handle.close()


def run_predict_requests(session, base_url, cfgs):
    durations = []
    responses = []
    for idx in range(cfgs.predict_repeats):
        data = build_request_payload(cfgs)
        files = build_request_files(cfgs)
        try:
            tic = time.perf_counter()
            predict_resp = post_json(
                session,
                f'{base_url}/api/v1/predict',
                files=files,
                data=data,
                timeout=cfgs.timeout,
            )
            duration = time.perf_counter() - tic
            durations.append(duration)
            responses.append(predict_resp)
            logger.info('Predict #%s time: %.3fs, grasp_count=%s', idx + 1, duration, predict_resp.get('grasp_count'))
        finally:
            close_request_files(files)
    return durations, responses


def main():
    configure_grasp_logger(level=logging.INFO)

    default_overrides = {
        'rgb_path': os.path.join(CURRENT_DIR, 'data', 'color', 'color_00000.png'),
        'depth_path': os.path.join(CURRENT_DIR, 'data', 'depth', 'depth_raw_00000.png'),
    }
    parser = build_predictor_arg_parser(description='Simulate edge-side requests to grasp server', default_overrides=default_overrides)
    parser.add_argument('--server_url', type=str, default='http://127.0.0.1:6006', help='Base server URL')
    parser.add_argument('--robot_id', type=str, default='edge-sim', help='Robot id for metadata')
    parser.add_argument('--cmd', type=str, default='predict', help='Debug metadata override for cmd field')
    parser.add_argument('--class_id', type=int, default=46, help='Target class id for the current remote contract')
    parser.add_argument('--skip_init', action='store_true', help='Skip /init call')
    parser.add_argument('--skip_release', action='store_true', help='Skip /release call')
    parser.add_argument('--timeout', type=float, default=120.0, help='Debug HTTP timeout in seconds')
    parser.add_argument('--predict_repeats', type=int, default=3, help='Number of predict requests to send')
    cfgs = parser.parse_args()

    if cfgs.class_id is not None and cfgs.class_id < 0:
        cfgs.class_id = None

    session = requests.Session()
    base_url = cfgs.server_url.rstrip('/')

    if not cfgs.skip_init:
        logger.info('Calling /api/v1/init')
        init_resp = post_json(session, f'{base_url}/api/v1/init', timeout=cfgs.timeout)
        logger.info('Init response: %s', init_resp)

    logger.info('Calling /api/v1/predict with class_id: %s', cfgs.class_id)

    durations, responses = run_predict_requests(session, base_url, cfgs)
    if responses:
        logger.info('Last predict response: %s', json.dumps(responses[-1], ensure_ascii=False))
    if durations:
        avg = sum(durations) / len(durations)
        logger.info(
            'Predict timing summary: repeats=%s min=%.3fs max=%.3fs avg=%.3fs',
            len(durations),
            min(durations),
            max(durations),
            avg,
        )

    if not cfgs.skip_release:
        logger.info('Calling /api/v1/release')
        release_resp = post_json(session, f'{base_url}/api/v1/release', timeout=cfgs.timeout)
        logger.info('Release response: %s', release_resp)


if __name__ == '__main__':
    main()
