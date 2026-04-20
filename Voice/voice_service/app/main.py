#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ..config import CONFIG
from ..runtime.service import run_voice_service


def main():
    run_voice_service(CONFIG)


if __name__ == "__main__":
    main()
