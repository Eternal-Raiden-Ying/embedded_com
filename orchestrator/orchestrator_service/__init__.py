import sys
from pathlib import Path


# Allow sibling shared packages like `common` to be imported when this service
# is launched from `/home/aidlux/embedded_com/orchestrator`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


__all__ = []
