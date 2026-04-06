#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from orchestrator_service.config.board_config import CONFIG
from orchestrator_service.runtime.service import run_orchestrator_service


def main() -> None:
    run_orchestrator_service(CONFIG)


if __name__ == "__main__":
    main()
