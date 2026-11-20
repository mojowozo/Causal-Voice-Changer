"""Single source of truth for the app's identity."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from causalvc.update import current_version  # noqa: E402  (the VERSION file a release carries)

PRODUCT_NAME = "CausalVC Realtime"
VERSION = current_version()

APP_TITLE = f"{PRODUCT_NAME} {VERSION}"
BUILD_LABEL = "Causal streaming voice changer"
