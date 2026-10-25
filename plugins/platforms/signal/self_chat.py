"""Note-to-Self delivery semantics.

A "Note to Self" chat is the user's own number talking to itself — the setup Hermes uses when
signal-cli runs as a linked device on the user's phone number instead of a separate bot number.

signal-cli's default for a send addressed to your own number is a ``syncMessage``: the man page
described ``--note-to-self`` as sending "without notification" until `--notify-self` was added.
The phone files such a message under Note to Self as an outgoing message it never authored, so no
push notification fires — the agent's replies arrive silently, which is easy to mistake for the
replies not arriving at all.

``notifySelf`` (the JSON-RPC twin of ``--notify-self``) makes signal-cli send a *normal* message
instead, and the phone alerts on it like any other chat. It only applies to direct chats; group
sends keep their own semantics.
"""

from __future__ import annotations

from typing import Any


def is_self_chat(account: str, chat_id: str) -> bool:
    """True when *chat_id* is the account's own number (Note to Self), not a group or contact."""
    chat, acct = str(chat_id or "").strip(), str(account or "").strip()
    return bool(chat) and bool(acct) and not chat.startswith("group:") and chat == acct


def set_signal_notify_self(params: dict[str, Any], account: str, chat_id: str) -> None:
    """Set ``notifySelf`` on *params* for a Note-to-Self send (in place, no-op otherwise).

    Unknown params are ignored by the daemon, so typing/receipt calls may carry it harmlessly.
    """
    if is_self_chat(account, chat_id):
        params["notifySelf"] = True
