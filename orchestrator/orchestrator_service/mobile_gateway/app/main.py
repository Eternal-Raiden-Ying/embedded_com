#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from orchestrator_service.mobile_gateway.config.board_config import CONFIG
from orchestrator_service.mobile_gateway.runtime.service import run_mobile_gateway_service


def main() -> None:
    run_mobile_gateway_service(CONFIG)


if __name__ == "__main__":
    main()

