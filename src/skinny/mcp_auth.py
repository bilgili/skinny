"""Token, bind, and request-guard helpers for the in-process MCP server.

Four independent layers, none a substitute for another (change
``mcp-scene-control``, design D10):

1. the server is off unless ``--mcp`` asks for it (handled in ``cli_common``);
2. the socket binds loopback only, asserted here at creation;
3. requests carrying an ``Origin`` header are refused, and ``Host`` is validated,
   so a page in the operator's browser cannot drive the renderer;
4. every request carries a bearer token, compared in constant time.

The token defends against other local processes and against browser reach. It
does *not* defend against a process that can read the operator's home directory,
which is exactly why the first three layers stay.
"""

from __future__ import annotations

import hmac
import os
import secrets
import socket
import stat
from pathlib import Path

from skinny.settings import SETTINGS_DIR, ensure_dirs

TOKEN_FILE = SETTINGS_DIR / "mcp_token"
LOOPBACK_HOST = "127.0.0.1"

# Only the owner may read the token. Anything broader means another local
# account could impersonate the operator's client.
_OWNER_ONLY = 0o600
_GROUP_WORLD_BITS = stat.S_IRWXG | stat.S_IRWXO


# A token shorter than this is a truncated write, not a secret.
_MIN_TOKEN_LEN = 32


def token_is_from_env() -> bool:
    """True when ``SKINNY_MCP_TOKEN`` supplies the token instead of the file."""
    return bool(os.environ.get("SKINNY_MCP_TOKEN", "").strip())


def load_or_create_token(path: Path | None = None) -> str:
    """Return the persistent bearer token, generating it on first use.

    Persistent rather than per-launch so a client's stored registration stays
    valid across restarts. ``SKINNY_MCP_TOKEN`` overrides the file for operators
    who would rather not have it on disk.
    """
    raw = os.environ.get("SKINNY_MCP_TOKEN", "")
    override = raw.strip()
    if override:
        if raw != override:
            # Authentication compares the stripped value, but a printed
            # `$SKINNY_MCP_TOKEN` expands the raw one -- say so rather than
            # letting the registration line fail mysteriously.
            print(
                "note: SKINNY_MCP_TOKEN has leading/trailing whitespace, which "
                "is stripped for authentication. The printed registration "
                "command strips it too, so it will work as-is."
            )
        return override

    path = path or TOKEN_FILE
    for _attempt in range(3):  # re-read after losing a creation race
        token = _read_token(path)
        if token is not None:
            return token
        _create_token(path)

    # Exhausted: the file exists but never reads back as usable, so it is
    # corrupt rather than contended. Say what to do about it.
    if path.exists():
        raise SystemExit(
            f"{path} exists but does not contain a usable token.\n"
            f"  Delete it and restart:  rm {path}"
        )
    raise SystemExit(f"could not establish a usable MCP token at {path}")


def _create_token(path: Path) -> None:
    """Create the token file, never overwriting an existing one.

    Publication is a single atomic exclusive step: the full payload is staged to
    a private temp file first, then linked into place. A concurrent starter that
    loses simply returns and re-reads the winner's token.

    **Nothing here ever replaces an existing file.** Two earlier attempts did —
    exclusive-create-then-write (a reader could see the empty file mid-write and
    "repair" it) and read-then-``exists()``-then-replace (the ``exists()`` check
    races the winner's publish, so the loser clobbered a token the winner was
    already serving). Both left one server holding a secret no client could
    obtain. Refusing to overwrite removes that whole class of bug; a corrupt
    token is a operator-visible error instead, which is recoverable by deleting
    one file.
    """
    ensure_dirs()
    payload = secrets.token_urlsafe(32).encode()

    # No `path.exists()` precheck: it races a peer's publish, and a starter that
    # arrived a moment late would report a corrupt-token error for a file that
    # is perfectly good. Exclusivity is established by the link below, which is
    # atomic; the caller re-reads and converges.

    # Unique per writer -- a bare pid collides between threads of one process.
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp")
    # Restrictive mode at creation, not chmod afterwards: between create and
    # chmod the token would be world-readable.
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _OWNER_ONLY)
    try:
        written = os.write(fd, payload)
        if written != len(payload):
            raise SystemExit(f"short write staging the MCP token at {tmp}")
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        tmp.unlink(missing_ok=True)
        raise
    os.close(fd)

    try:
        try:
            # Atomic exclusive publish: fails if anyone already published.
            os.link(tmp, path)
        except FileExistsError:
            return  # another starter won; the caller re-reads their token
        except (OSError, AttributeError) as exc:
            # No os.replace fallback. os.replace is atomic but *not* exclusive,
            # so two starters could both publish and one would destroy a token
            # the other is already serving -- the precise bug this function
            # exists to prevent. Refuse instead, and say why.
            raise SystemExit(
                f"cannot atomically create the MCP token at {path}: {exc}\n"
                "  The filesystem does not support hard links, which are "
                "required to publish it safely.\n"
                "  Set SKINNY_MCP_TOKEN instead to supply the token directly."
            )
    finally:
        # link() leaves the staging copy behind.
        tmp.unlink(missing_ok=True)


def _read_token(path: Path) -> str | None:
    """Read and validate the token through a single descriptor.

    On POSIX: opened with ``O_NOFOLLOW`` and validated by ``fstat`` on that same
    descriptor, so the file cannot be swapped for a symlink between the checks
    and the read. Returns ``None`` when the token is absent or too short to be
    one; raises when it exists but is unsafe to use.

    **On Windows these guarantees are weaker.** ``O_NOFOLLOW`` and ``getuid``
    do not exist there, so a reparse point is followed and the ownership/mode
    checks are skipped -- NTFS ACLs govern access instead, and this code does
    not inspect them. The token file is therefore only as protected as the
    user profile directory containing it. Recorded as a known platform gap
    rather than silently implied to be equivalent.
    """
    # O_NOFOLLOW and getuid() are POSIX-only; Windows is a supported platform,
    # so degrade to the checks that exist there rather than crashing at startup.
    # NTFS ACLs, not mode bits, govern access on that side.
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    getuid = getattr(os, "getuid", None)

    try:
        fd = os.open(path, os.O_RDONLY | no_follow)
    except FileNotFoundError:
        return None
    except OSError as exc:  # ELOOP: the path is a symlink
        raise SystemExit(f"refusing to read MCP token at {path}: {exc}")

    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"{path} is not a regular file; refusing to use it")
        if getuid is not None and info.st_uid != getuid():
            raise SystemExit(f"{path} is owned by uid {info.st_uid}, not you")
        if getuid is not None and info.st_mode & _GROUP_WORLD_BITS:
            raise SystemExit(
                f"{path} is readable beyond its owner "
                f"(mode {stat.filemode(info.st_mode)}).\n"
                f"  Fix with:  chmod 600 {path}"
            )
        token = os.read(fd, 4096).decode(errors="replace").strip()
    finally:
        os.close(fd)

    return token if len(token) >= _MIN_TOKEN_LEN else None


def bind_loopback_socket(port: int) -> socket.socket:
    """Create the listening socket, asserting it bound loopback.

    The server never binds an address the caller supplies — the host is fixed
    here. Creating the socket in-process (rather than letting the server runtime
    do it) is also what makes a port collision catchable before startup reports
    success.

    Raises ``OSError`` if the port is taken.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind((LOOPBACK_HOST, port))
        bound_host, _bound_port = sock.getsockname()
        if bound_host != LOOPBACK_HOST:
            raise RuntimeError(
                f"refusing to serve MCP on {bound_host!r}: loopback only"
            )
        sock.listen(16)
    except BaseException:
        sock.close()
        raise
    return sock


def check_request(headers, token: str, port: int) -> str | None:
    """Guard one request. Returns ``None`` to allow, or a refusal reason.

    Order matters: the cheap structural checks run before the token comparison,
    so a browser probe is refused without touching the secret at all.
    """
    origin = _header(headers, "origin")
    if origin is not None:
        # A conforming MCP client sends no Origin; a browser cannot omit one.
        # This also blocks browser-hosted MCP clients -- an accepted cost.
        return "requests carrying an Origin header are refused"

    host = _header(headers, "host")
    if not _host_is_allowed(host, port):
        # Catches DNS rebinding, which reaches us with no Origin at all. A
        # missing Host is refused too rather than waved through.
        return f"unexpected Host header {host!r}"

    authorization = _header(headers, "authorization") or ""
    expected = f"Bearer {token}"
    # compare_digest, not ==: string comparison short-circuits on the first
    # differing byte and leaks the token to anyone who can time requests.
    if not hmac.compare_digest(authorization, expected):
        return "missing or invalid bearer token"
    return None


def _header(headers, name: str) -> str | None:
    """Case-insensitive lookup across dict-like and multidict header objects."""
    getter = getattr(headers, "get", None)
    if getter is not None:
        value = getter(name)
        if value is None:
            value = getter(name.title())
        if value is not None:
            return value if isinstance(value, str) else value.decode()
    for key, value in dict(headers).items():
        key_text = key if isinstance(key, str) else key.decode()
        if key_text.lower() == name:
            return value if isinstance(value, str) else value.decode()
    return None


def _host_is_allowed(host: str | None, port: int) -> bool:
    """Require an exact loopback authority including the bound port.

    A bare hostname with no port, or a port that is not ours, is refused: both
    are shapes a rebound-DNS request can take, and neither is what a client
    talking to this server actually sends.
    """
    if not host:
        return False
    hostname, _, host_port = host.rpartition(":")
    if not hostname:  # no ":" in the header at all
        return False
    return host_port == str(port) and hostname in (
        LOOPBACK_HOST, "localhost", "[::1]",
    )


def registration_command(port: int) -> str:
    """The copy-paste client registration line printed at startup.

    Never interpolates the token itself — that would put the secret into
    terminal scrollback and any captured log. Points at whichever source is
    actually in force, since with ``SKINNY_MCP_TOKEN`` set the file may not
    exist or may hold a different value.
    """
    windows = os.name == "nt"
    if token_is_from_env():
        # Trim in the emitted command, not just in a warning: authentication
        # compares the stripped value, so a raw expansion of a whitespace-padded
        # variable would produce a command that cannot authenticate.
        # Strip leading/trailing only, exactly like Python's .strip() used for
        # authentication. `tr -d` would also delete *internal* whitespace, so a
        # token containing a space would be sent differently than it is checked.
        posix_trim = (
            '$(printf %s "$SKINNY_MCP_TOKEN" '
            "| sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
        )
        source = "$($env:SKINNY_MCP_TOKEN.Trim())" if windows else posix_trim
    elif windows:
        # -LiteralPath + single quotes: Windows profile paths routinely contain
        # spaces, and may contain PowerShell metacharacters.
        literal = str(TOKEN_FILE).replace("'", "''")
        source = f"$(Get-Content -Raw -LiteralPath '{literal}')".rstrip()
    else:
        source = f'$(cat "{TOKEN_FILE}")'
    continuation = "`" if windows else "\\"
    return (
        f"claude mcp add --transport http skinny "
        f"http://{LOOPBACK_HOST}:{port}/mcp {continuation}\n"
        f'  --header "Authorization: Bearer {source}"'
    )
