import os
from dataclasses import dataclass

from .predictor_config import MODULE_DIR, PredictorConfig, build_predictor_arg_parser, create_predictor_config

@dataclass
class AppConfig(PredictorConfig):
    log_path: str = os.path.join(MODULE_DIR, 'log', 'server.log')

def get_config() -> AppConfig:
    default_overrides = {
        'dump_dir': os.path.join(MODULE_DIR, 'debug_res'),
        'debug': False,
    }
    parser = build_predictor_arg_parser(default_overrides=default_overrides)
    parser.add_argument('--log_path', type=str, default=os.path.join(MODULE_DIR, 'log', 'server.log'))

    args, _ = parser.parse_known_args()
    args_dict = vars(args).copy()
    log_path = args_dict.pop('log_path')
    predictor_cfg = create_predictor_config(default_overrides=default_overrides, **args_dict)
    return AppConfig(**predictor_cfg.__dict__, log_path=log_path)

cfgs = get_config()
