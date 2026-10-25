"""Demo runner for Signal action links: serve a button page on the tailnet and send the link to
Note to Self, so the tap happens on the phone.

    ACTION_HOST=100.111.43.43 ACTION_PORT=8111 SIGNAL_ACCOUNT=+33... \
        ~/.hermes/venv/bin/python -m plugins.platforms.signal.action_links_demo \
            --question "Pick a channel for the report:" \
            --choice "Vault note|vault" --choice "Email|email" --choice "Skip|skip" --send

Without ``--send`` it only prints the URL. Taps are appended to ``--store`` (default
``~/.hermes/cache/scratch/signal-deep/action-taps.jsonl``) and each one is confirmed back into the
Signal chat with ``notifySelf`` so the phone alerts on it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.error
import urllib.request

from plugins.platforms.signal import action_links as al


def rpc(url: str, method: str, params: dict, timeout: float = 30.0) -> dict:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    request = urllib.request.Request(f"{url.rstrip('/')}/api/v1/rpc", body,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as err:
        return {"http_error": err.code, "body": err.read().decode()[:300]}
    except Exception as err:
        return {"error": str(err)}


def send_prompt(http_url: str, account: str, question: str, link: str) -> dict:
    """Send the prompt with a link preview; notifySelf keeps the phone's notification alive."""
    return rpc(http_url, "send", {
        "account": account,
        "recipients": [account],
        "message": f"{question}\n\nTap to answer: {link}",
        "previewUrl": link,
        "previewTitle": question,
        "previewDescription": "Tap a button — no typing.",
        "notifySelf": True,
    })


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--choice", action="append", default=[], help="Label|value (repeatable)")
    parser.add_argument("--ttl", type=int, default=al.DEFAULT_TTL_SECONDS)
    parser.add_argument("--host", default=os.environ.get("ACTION_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("ACTION_PORT", "8111")))
    parser.add_argument("--store", default=os.path.expanduser(
        "~/.hermes/cache/scratch/signal-deep/action-taps.jsonl"))
    parser.add_argument("--secret-file", default=os.path.expanduser("~/.hermes/signal-actions.secret"))
    parser.add_argument("--send", action="store_true", help="send the link to Note to Self")
    args = parser.parse_args(argv)

    if len(args.choice) < 2:
        parser.error("give at least two --choice rows")

    prompt = al.prompt_from_rows(args.question, args.choice)
    secret = al.new_secret(args.secret_file)
    http_url = os.environ.get("SIGNAL_HTTP_URL", "http://127.0.0.1:8096")
    account = os.environ.get("SIGNAL_ACCOUNT", "")

    def on_choice(_prompt, choice):
        print(f"TAP {choice.value!r} ({choice.label})", flush=True)
        if account:
            rpc(http_url, "send", {"account": account, "recipients": [account],
                                   "message": f"✅ Recorded: {choice.label}", "notifySelf": True})

    server = al.ActionServer(secret, host=args.host, port=args.port,
                             store=al.ActionStore(args.store), on_choice=on_choice).start()
    link = server.link(prompt, ttl=args.ttl)
    print(f"SERVING {server.base_url}", flush=True)
    print(f"LINK {link}", flush=True)

    if args.send:
        if not account:
            print("SIGNAL_ACCOUNT not set: not sending", file=sys.stderr)
        else:
            result = send_prompt(http_url, account, args.question, link)
            print(f"SEND {json.dumps(result)[:300]}", flush=True)

    print("READY — Ctrl-C to stop", flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
