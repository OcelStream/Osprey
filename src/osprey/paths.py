"""Where the server↔client IPC sockets live.

Both halves of Osprey must agree on one directory: the server's
``nvunixfdsink`` binds ``<dir>/<stream_id>.sock`` and the client watches the
same ``<dir>`` for sockets to attach to.

The default is ``./sockets`` — relative to the working directory the app is
launched from — so nothing needs root to exist. The server forks from the
client process, so both see the same cwd.

Override with the ``OSPREY_SOCKET_DIR`` env var, ``osprey.configure(
socket_dir=…)`` on the server, or ``DeepStreamClient(watch_dir=…)`` on the
client.
"""

import os
from typing import Optional

ENV_VAR = "OSPREY_SOCKET_DIR"

#: Directory name created under the working directory when nothing is set.
DIR_NAME = "sockets"


def default_socket_dir() -> str:
    """Resolve the socket directory: ``$OSPREY_SOCKET_DIR`` or ``./sockets``."""
    override = os.environ.get(ENV_VAR)
    if override:
        return os.path.abspath(override)
    return os.path.join(os.getcwd(), DIR_NAME)


def ensure_socket_dir(path: Optional[str] = None) -> str:
    """Create the socket directory if needed and return its absolute path."""
    resolved = os.path.abspath(path) if path else default_socket_dir()
    os.makedirs(resolved, exist_ok=True)
    return resolved
