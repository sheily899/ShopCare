"""读取转人工交接新流程开关；未显式开启时保持旧流程。"""

from __future__ import annotations

import os


def handoff_enabled() -> bool:
    """只有环境变量 ``HANDOFF`` 显式为真时才启用新增交接行为。"""
    return os.getenv("HANDOFF", "false").strip().lower() in {"1", "true", "yes", "on"}
