"""Tests for Signal action links (real buttons hosted outside the chat)."""

import json
import time
import urllib.error
import urllib.request

import pytest

from plugins.platforms.signal import action_links as al


SECRET = "test-secret"


def _prompt():
    return al.prompt_from_rows("Pick a channel for the report:", ["Vault note|vault", "Email|email"])


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------

class TestTokens:

    def test_roundtrip_preserves_question_and_choices(self):
        prompt = _prompt()
        token = al.sign_token(SECRET, prompt, ttl=60)
        back = al.verify_token(SECRET, token)
        assert back is not None
        assert back.question == prompt.question
        assert [(c.label, c.value) for c in back.choices] == [("Vault note", "vault"), ("Email", "email")]

    def test_expired_token_is_rejected(self):
        token = al.sign_token(SECRET, _prompt(), ttl=60, now=time.time() - 120)
        assert al.verify_token(SECRET, token) is None

    def test_tampered_payload_is_rejected(self):
        token = al.sign_token(SECRET, _prompt(), ttl=60)
        head, _body, mac = token.split(".")
        forged = al._b64e(json.dumps({"q": "x", "exp": int(time.time()) + 999,
                                      "c": [{"l": "a", "v": "a"}, {"l": "b", "v": "b"}]}).encode())
        assert al.verify_token(SECRET, ".".join([head, forged, mac])) is None

    def test_wrong_secret_is_rejected(self):
        token = al.sign_token(SECRET, _prompt(), ttl=60)
        assert al.verify_token("other-secret", token) is None

    def test_garbage_tokens_are_rejected(self):
        for bad in ("", "nope", "v1..", "v1.a.b.c", "v2.a.b"):
            assert al.verify_token(SECRET, bad) is None

    def test_single_choice_prompt_is_rejected(self):
        prompt = al.ActionPrompt(question="Only one?", choices=[al.ActionChoice("yes")])
        token = al.sign_token(SECRET, prompt, ttl=60)
        assert al.verify_token(SECRET, token) is None


# ---------------------------------------------------------------------------
# Page rendering
# ---------------------------------------------------------------------------

class TestRenderPage:

    def test_one_button_per_choice(self):
        prompt = _prompt()
        page = al.render_page(prompt, "tok", action_path="/a/tok", now=time.time())
        assert page.count("<button type=\"submit\"") == 2
        assert "Vault note" in page and "Email" in page

    def test_html_is_escaped(self):
        prompt = al.ActionPrompt(question="<script>alert(1)</script> & co",
                                 choices=[al.ActionChoice("<b>yes</b>"), al.ActionChoice("no")])
        page = al.render_page(prompt, "tok", action_path="/a/tok")
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page
        assert "&lt;b&gt;yes&lt;/b&gt;" in page

    def test_expiry_line_reflects_ttl(self):
        prompt = al.prompt_from_rows("Q", ["a", "b"])
        prompt.expires_at = time.time() + 600
        page = al.render_page(prompt, "tok", action_path="/a/tok")
        assert "expires in 9 min" in page or "expires in 10 min" in page

    def test_chosen_state_replaces_buttons(self):
        page = al.render_page(_prompt(), "tok", action_path="/a/tok", chosen="Email")
        assert "✅ Email" in page
        assert "<button type=\"submit\"" not in page


# ---------------------------------------------------------------------------
# Server round-trip
# ---------------------------------------------------------------------------

@pytest.fixture()
def server(tmp_path):
    taps = []
    store = al.ActionStore(str(tmp_path / "taps.jsonl"))
    srv = al.ActionServer(SECRET, host="127.0.0.1", port=0, store=store,
                          on_choice=lambda p, c: taps.append(c.value)).start()
    yield srv, store, taps
    srv.stop()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read().decode()


def _post(url, body):
    req = urllib.request.Request(url, body.encode(), {"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return resp.status, resp.read().decode()


class TestServer:

    def test_link_serves_the_button_page(self, server):
        srv, _store, _taps = server
        url = srv.link(_prompt(), ttl=300)
        status, body = _get(url)
        assert status == 200
        assert "<button type=\"submit\"" in body

    def test_tap_records_choice_and_calls_hook(self, server):
        srv, store, taps = server
        url = srv.link(_prompt(), ttl=300)
        status, body = _post(url, "i=1")
        assert status == 200
        assert "✅ Email" in body
        assert taps == ["email"]
        lines = [json.loads(line) for line in open(store.path, encoding="utf-8")]
        assert lines[-1]["value"] == "email"
        assert lines[-1]["question"] == "Pick a channel for the report:"

    def test_out_of_range_index_is_rejected(self, server):
        srv, _store, taps = server
        url = srv.link(_prompt(), ttl=300)
        with pytest.raises(urllib.error.HTTPError) as err:
            _post(url, "i=9")
        assert err.value.code == 400
        assert taps == []

    def test_expired_link_is_gone(self, server):
        srv, _store, _taps = server
        token = al.sign_token(SECRET, _prompt(), ttl=1, now=time.time() - 60)
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"{srv.base_url}/a/{token}")
        assert err.value.code == 410

    def test_unknown_path_is_gone(self, server):
        srv, _store, _taps = server
        with pytest.raises(urllib.error.HTTPError) as err:
            _get(f"{srv.base_url}/a/not-a-token")
        assert err.value.code == 410

    def test_hook_failure_does_not_lose_the_tap(self, tmp_path):
        store = al.ActionStore(str(tmp_path / "taps.jsonl"))

        def boom(_prompt, _choice):
            raise RuntimeError("hook down")

        srv = al.ActionServer(SECRET, host="127.0.0.1", port=0, store=store, on_choice=boom).start()
        try:
            status, _body = _post(srv.link(_prompt(), ttl=300), "i=0")
            assert status == 200
            assert json.loads(open(store.path, encoding="utf-8").read().splitlines()[-1])["value"] == "vault"
        finally:
            srv.stop()
