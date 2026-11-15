"""Wire format shared by the engine server and its clients.

Messages are UTF-8 JSON objects, one per line, over a local stream socket.

    client -> engine   {"id": 1, "cmd": "start", "args": {"function": "vc"}}
    engine -> client   {"id": 1, "ok": true, "result": null}
                       {"id": 1, "ok": false, "error": {"code": ..., "message": ..., "details": {...}}}
                       {"event": "meters", "data": {...}}

The socket is a Unix socket on Linux.  Windows has no Unix sockets in asyncio,
so there the engine listens on a loopback TCP port instead and writes the port
and a random token to an endpoint file only the user can read; a client sends
``{"auth": token}`` as its first line before anything else is exchanged.

An address is either a Unix socket path or ``tcp:<endpoint file>``.

See docs/engine-protocol.md for the commands and events.
"""

import json
import os
import socket

PROTOCOL_VERSION = 1
SOCKET_NAME = "causalvc-realtime.sock"
ENDPOINT_NAME = "realtime-engine.json"
TCP_PREFIX = "tcp:"


def default_socket_path():
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or f"/tmp/causalvc-realtime-{os.getuid()}"
    return os.path.join(runtime_dir, SOCKET_NAME)


def default_endpoint_file():
    """Where a TCP engine writes its port and token: per user, under %LOCALAPPDATA% on Windows."""
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".local", "state")
    return os.path.join(base, "CausalVC", ENDPOINT_NAME)


def default_address():
    """The Unix socket on Linux and macOS; the TCP endpoint file on Windows."""
    if os.name == "nt":
        return TCP_PREFIX + default_endpoint_file()
    return default_socket_path()


def is_tcp(address):
    return address.startswith(TCP_PREFIX)


def endpoint_path(address):
    return address[len(TCP_PREFIX):]


def read_endpoint(path):
    """``(port, token)`` from an endpoint file, or ``None`` if it is missing or unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        return int(data["port"]), str(data["token"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def connect(address, timeout=None):
    """A connected, authenticated socket to the engine at ``address``."""
    if not is_tcp(address):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(address)
        except OSError:
            sock.close()
            raise
        return sock
    endpoint = read_endpoint(endpoint_path(address))
    if endpoint is None:
        raise ConnectionRefusedError(f"no engine endpoint at {endpoint_path(address)}")
    port, token = endpoint
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.settimeout(timeout)
    try:
        sock.sendall(encode({"auth": token}))
    except OSError:
        sock.close()
        raise
    return sock


def encode(message):
    return (json.dumps(message, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def decode(line):
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("message must be a JSON object")
    return message
