"""Signal action links — real buttons for a chat that has no inline keyboards.

Signal has no bot API and no interactive message components, so nothing can attach a button to a
Signal message. Polls are the only native widget (a list of options plus Signal's own "Vote"
button). Everything else has to leave the chat.

An "action link" is the Telegram-style escape hatch that works today: the prompt message carries a
link (Signal renders its link preview as a tappable card), and the link opens a small mobile page
whose buttons POST the choice back to the machine that sent the prompt. One tap, no typing, and
the choice comes back as a signed, replay-resistant value instead of free text.

Security model: every token is an HMAC-SHA256-signed, expiring, single-use record of the question
and its choices — the page cannot be made to return a value the sender did not offer, and a leaked
link dies with its TTL. The server binds to the tailnet address, so only the user's own devices can
reach it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import html
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterable, Optional
from urllib.parse import parse_qs, urlsplit

TOKEN_VERSION = "v1"
DEFAULT_TTL_SECONDS = 24 * 60 * 60


@dataclass
class ActionChoice:
    label: str
    value: Optional[str] = None

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = self.label


@dataclass
class ActionPrompt:
    """A question plus its choices, as rendered in the page and carried inside the token."""

    question: str
    choices: list[ActionChoice]
    prompt_id: str = field(default_factory=lambda: secrets.token_hex(5))
    expires_at: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _b64d(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def sign_token(secret: str, prompt: ActionPrompt, *, ttl: int = DEFAULT_TTL_SECONDS,
               now: Optional[float] = None) -> str:
    """Return a signed, expiring token encoding *prompt*."""
    now = time.time() if now is None else now
    payload = {
        "q": prompt.question,
        "id": prompt.prompt_id,
        "c": [{"l": c.label, "v": c.value} for c in prompt.choices],
        "exp": int(now + ttl) if not prompt.expires_at else int(prompt.expires_at),
        "m": prompt.meta or {},
    }
    body = _b64e(json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode())
    signed = f"{TOKEN_VERSION}.{body}"
    mac = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).digest()
    return f"{signed}.{_b64e(mac)}"


def verify_token(secret: str, token: str, *, now: Optional[float] = None) -> Optional[ActionPrompt]:
    """Return the prompt if *token* is authentic and unexpired, else None."""
    now = time.time() if now is None else now
    parts = (token or "").split(".")
    if len(parts) != 3 or parts[0] != TOKEN_VERSION:
        return None
    signed = f"{parts[0]}.{parts[1]}"
    expected = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).digest()
    try:
        given = _b64d(parts[2])
    except Exception:
        return None
    if not hmac.compare_digest(expected, given):
        return None
    try:
        payload = json.loads(_b64d(parts[1]))
    except Exception:
        return None
    if not isinstance(payload, dict) or not payload.get("q"):
        return None
    if int(payload.get("exp") or 0) <= int(now):
        return None
    choices = [ActionChoice(c.get("l") or "", c.get("v")) for c in payload.get("c") or []]
    if len(choices) < 2:
        return None
    return ActionPrompt(question=payload["q"], choices=choices, prompt_id=payload.get("id") or "",
                        expires_at=float(payload.get("exp") or 0), meta=payload.get("m") or {})


_PAGE_CSS = """
:root{color-scheme:light dark}
*{box-sizing:border-box}
body{margin:0;padding:24px 18px 40px;font:16px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
     background:#f6f7f9;color:#111}
@media (prefers-color-scheme:dark){body{background:#101114;color:#eceef2}}
main{max-width:520px;margin:0 auto}
.badge{font-size:12px;letter-spacing:.08em;text-transform:uppercase;opacity:.55;margin-bottom:10px}
h1{font-size:20px;line-height:1.35;margin:0 0 6px;font-weight:600}
.sub{font-size:13px;opacity:.6;margin:0 0 22px}
form{margin:0 0 10px}
button{display:block;width:100%;padding:15px 16px;font:inherit;font-weight:600;text-align:left;
       border:1px solid #d6d9df;border-radius:12px;background:#fff;color:inherit;cursor:pointer;
       transition:transform .06s ease,background .12s ease}
button:active{transform:scale(.99);background:#eef1f5}
@media (prefers-color-scheme:dark){button{background:#1a1c21;border-color:#2c3037}
  button:active{background:#23262c}}
.chosen{display:flex;gap:12px;align-items:center;padding:16px;border-radius:12px;background:#e8f5ec;
        color:#0d5b2a;font-weight:600}
@media (prefers-color-scheme:dark){.chosen{background:#12301f;color:#8ff0b4}}
.foot{margin-top:26px;font-size:12px;opacity:.5}
"""


def render_page(prompt: ActionPrompt, token: str, *, action_path: str = "", chosen: Optional[str] = None,
                now: Optional[float] = None) -> str:
    """Render the button page for *prompt*. All interpolated text is escaped."""
    now = time.time() if now is None else now
    minutes = max(0, int((prompt.expires_at - now) / 60)) if prompt.expires_at else 0
    rows = []
    for index, choice in enumerate(prompt.choices):
        rows.append(
            '<form method="post" action="{action}">'
            '<input type="hidden" name="i" value="{i}">'
            '<button type="submit">{label}</button>'
            "</form>".format(action=html.escape(action_path or f"?t={token}"), i=index,
                             label=html.escape(choice.label))
        )
    if chosen is not None:
        body = (f'<div class="chosen">✅ {html.escape(chosen)}</div>'
                '<p class="sub" style="margin-top:14px">The agent has your answer.</p>')
    else:
        body = "".join(rows)
    expiry = f"Tap one — this prompt expires in {minutes} min." if prompt.expires_at else "Tap one."
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow">
<title>{html.escape(prompt.question)}</title><style>{_PAGE_CSS}</style></head>
<body><main>
<div class="badge">Hermes · Signal action</div>
<h1>{html.escape(prompt.question)}</h1>
<p class="sub">{html.escape(expiry)}</p>
{body}
<p class="foot">Nothing here is public: the link is single-use, signed, and only reachable on your
tailnet.</p>
</main></body></html>"""


class ActionStore:
    """Records taps so the agent can read them back (append-only JSONL, one record per tap)."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    def record(self, prompt: ActionPrompt, choice: ActionChoice, *, source: str = "link") -> dict:
        entry = {"ts": time.time(), "prompt_id": prompt.prompt_id, "question": prompt.question,
                 "label": choice.label, "value": choice.value, "source": source,
                 "meta": prompt.meta or {}}
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock, open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        return entry


class ActionServer:
    """Tiny stdlib HTTP server for action pages: GET renders, POST records the tap and calls back."""

    def __init__(self, secret: str, *, host: str = "127.0.0.1", port: int = 8111,
                 store: Optional[ActionStore] = None,
                 on_choice: Optional[Callable[[ActionPrompt, ActionChoice], None]] = None):
        self.secret, self.host, self.port = secret, host, port
        self.store = store or ActionStore("/tmp/signal-actions.jsonl")
        self.on_choice = on_choice
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.taps: list[dict] = []  # in-process log, handy for tests and the demo hook

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> ActionServer:
        handler = self._handler_class()
        self._server = ThreadingHTTPServer((self.host, self.port), handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        self._server = None
        self._thread = None

    @property
    def base_url(self) -> str:
        host = self.host if self.host not in ("0.0.0.0", "") else "127.0.0.1"
        return f"http://{host}:{self.port}"

    # -- routing -----------------------------------------------------------
    def link(self, prompt: ActionPrompt, *, ttl: int = DEFAULT_TTL_SECONDS) -> str:
        token = sign_token(self.secret, prompt, ttl=ttl)
        return f"{self.base_url}/a/{token}"

    def _handler_class(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "HermesSignalActions/1.0"

            def log_message(self, format, *args):
                return

            def _send(self, status: int, body: str, content_type: str = "text/html; charset=utf-8"):
                raw = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Referrer-Policy", "no-referrer")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                parts = urlsplit(self.path)
                token = ""
                if parts.path.startswith("/a/"):
                    token = parts.path[3:]
                elif parts.path == "/a":
                    token = parse_qs(parts.query).get("t", [""])[0]
                prompt = verify_token(outer.secret, token)
                if not prompt:
                    self._send(HTTPStatus.GONE, "<h1>This prompt has expired.</h1>")
                    return
                self._send(HTTPStatus.OK, render_page(prompt, token, action_path="/a/" + token))

            def do_POST(self):
                parts = urlsplit(self.path)
                token = parts.path[3:] if parts.path.startswith("/a/") else ""
                prompt = verify_token(outer.secret, token)
                if not prompt:
                    self._send(HTTPStatus.GONE, "<h1>This prompt has expired.</h1>")
                    return
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(min(length, 4096)).decode(errors="replace")
                raw_index = parse_qs(body).get("i", ["-1"])[0]
                try:
                    index = int(raw_index)
                except ValueError:
                    index = -1
                if not 0 <= index < len(prompt.choices):
                    self._send(HTTPStatus.BAD_REQUEST, "<h1>Unknown choice.</h1>")
                    return
                choice = prompt.choices[index]
                entry = outer.store.record(prompt, choice)
                outer.taps.append(entry)
                if outer.on_choice:
                    try:
                        outer.on_choice(prompt, choice)
                    except Exception:  # the tap is already recorded; a hook failure must not 500
                        pass
                self._send(HTTPStatus.OK, render_page(prompt, token, action_path="/a/" + token,
                                                       chosen=choice.label))

        return Handler


def new_secret(path: Optional[str] = None) -> str:
    """Load a persistent action secret, creating it on first use (owner-only file if *path*)."""
    if not path:
        return secrets.token_urlsafe(32)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            secret = handle.read().strip()
        if secret:
            return secret
    except FileNotFoundError:
        pass
    secret = secrets.token_urlsafe(32)
    import os
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(secret)
    return secret


def prompt_from_rows(question: str, rows: Iterable[str]) -> ActionPrompt:
    """'Yes|yes', 'Deploy now' — ``label|value`` rows become choices (value defaults to label)."""
    choices = []
    for row in rows:
        label, _, value = str(row).partition("|")
        choices.append(ActionChoice(label.strip(), value.strip() or None))
    return ActionPrompt(question=question, choices=choices)
