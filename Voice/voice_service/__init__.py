"""语音服务主包。"""

import sys
from pathlib import Path


# Allow sibling shared packages like `common` to be imported when this service
# is launched from the `Voice/` directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
