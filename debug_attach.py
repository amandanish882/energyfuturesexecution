"""Debug attachment helper for Visual Studio / VS Code.

Enables remote debugging via debugpy when running in development.
Set COMMODITIES_DEBUG=1 to activate.

Usage
-----
    import debug_attach
    debug_attach.wait_for_client()  # blocks until debugger attaches
"""

import os
import sys
import logging

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 5678


def wait_for_client(port: int = _DEFAULT_PORT) -> None:
    """Start debugpy listener and wait for debugger attachment.

    Only activates if COMMODITIES_DEBUG environment variable is set.
    """
    if not os.environ.get("COMMODITIES_DEBUG"):
        return

    try:
        import debugpy
        debugpy.listen(("0.0.0.0", port))
        logger.info("debugpy listening on port %d — waiting for client...", port)
        print(f"[DEBUG] Waiting for debugger on port {port}...")
        debugpy.wait_for_client()
        logger.info("Debugger attached")
        print("[DEBUG] Debugger attached, continuing...")
    except ImportError:
        logger.warning("debugpy not installed; skipping debug attachment")
    except Exception as e:
        logger.warning("Debug attach failed: %s", e)


if __name__ == "__main__":
    os.environ["COMMODITIES_DEBUG"] = "1"
    wait_for_client()
