"""Tests for Signal interaction primitives (signal_ui) and the adapter's tap handling.

Signal has no inline buttons. These tests pin the three affordances Hermes uses instead — native
polls, numbered choice cards and emoji-reaction keys — plus the inbound translation that turns the
user's reaction / poll vote / numbered reply / edit / delete into text the agent can read.
"""
import pytest

from gateway.config import PlatformConfig
from plugins.platforms.signal import signal_ui
from plugins.platforms.signal.signal_ui import (
    KEY_EMOJIS, Choice, PendingPrompt, PromptRegistry, describe_delete, describe_edit,
    describe_poll_create, describe_poll_vote, describe_reaction, describe_sticker,
    describe_text_choice, parse_choice_reply, parse_multi_reply, poll_eligible, poll_payload,
    render_choice_card,
)

ACCOUNT = "+15555550123"
UUID = "11111111-2222-3333-4444-555555555555"


# ---------------------------------------------------------------------------
# Choice cards
# ---------------------------------------------------------------------------

class TestChoiceCard:
    def test_numbers_every_choice_and_bolds_the_marker(self):
        text, styles = render_choice_card("Deploy to prod?", [Choice("Yes"), Choice("No"), Choice("Later")])
        lines = text.split("\n")
        assert lines[0] == "Deploy to prod?"
        assert lines[2].startswith("1) Yes")
        assert lines[3].startswith("2) No")
        assert lines[4].startswith("3) Later")
        assert "Reply with the number" in text
        assert "0:15:BOLD" in styles  # the question line

    def test_styles_are_a_list_never_a_bare_string(self):
        _, styles = render_choice_card("q", [Choice("a")])
        assert isinstance(styles, list) and all(isinstance(s, str) for s in styles)
        # textStyles must be a list: signal-cli rejects a string with a ClassCastException.
        assert all(s.count(":") == 2 for s in styles)

    def test_style_offsets_are_utf16_and_land_inside_the_text(self):
        text, styles = render_choice_card("🚀 Deploy?", [Choice("Yes"), Choice("No")])
        u16_len = len(text.encode("utf-16-le")) // 2
        for style in styles:
            start, length, kind = style.split(":")
            assert int(start) + int(length) <= u16_len
            assert kind in {"BOLD", "ITALIC", "STRIKETHROUGH", "MONOSPACE", "SPOILER"}

    def test_key_emojis_and_multi_hint(self):
        text, _ = render_choice_card("Which checks?", [Choice("Tests"), Choice("Lint"), Choice("Types")], multi=True)
        assert KEY_EMOJIS[0] in text and KEY_EMOJIS[2] in text
        assert "numbers (e.g. 1 3)" in text

    def test_rejects_empty_choice_list(self):
        with pytest.raises(ValueError):
            render_choice_card("q", [])


# ---------------------------------------------------------------------------
# Free-text replies and poll eligibility
# ---------------------------------------------------------------------------

class TestParsing:
    @pytest.mark.parametrize("text,expected", [
        ("2", 2), ("2.", 2), ("2)", 2), ("#2", 2), (" option 2 ", 2), ("choice 3", 3),
        ("two", 2), ("THREE", 3), (KEY_EMOJIS[0], 1), (KEY_EMOJIS[4], 5),
    ])
    def test_accepted_number_replies(self, text, expected):
        assert parse_choice_reply(text, 5) == expected

    @pytest.mark.parametrize("text", ["0", "6", "", "yes", "2 apples", "12", None])
    def test_rejected_replies(self, text):
        assert parse_choice_reply(text, 5) is None

    def test_multi_reply(self):
        assert parse_multi_reply("1 3", 4) == [1, 3]
        assert parse_multi_reply("1,3", 4) == [1, 3]
        assert parse_multi_reply("1 and 4", 4) == [1, 4]
        assert parse_multi_reply("nonsense", 4) == []

    def test_poll_eligibility_boundaries(self):
        assert poll_eligible([Choice("a"), Choice("b")]) is True
        assert poll_eligible([Choice("a")]) is False
        assert poll_eligible([Choice(f"o{i}") for i in range(10)]) is True
        assert poll_eligible([Choice(f"o{i}") for i in range(11)]) is False
        assert poll_eligible([Choice("a" * 100), Choice("b")]) is True
        assert poll_eligible([Choice("a" * 101), Choice("b")]) is False

    def test_poll_payload_matches_signal_cli_names(self):
        assert poll_payload("Q?", [Choice("A"), Choice("B")]) == {
            "question": "Q?", "options": ["A", "B"], "multiSelect": False}
        assert poll_payload("Q?", [Choice("A"), Choice("B")], multi=True)["multiSelect"] is True


# ---------------------------------------------------------------------------
# Prompt registry
# ---------------------------------------------------------------------------

class TestPromptRegistry:
    def _prompt(self, question="Deploy?", labels=("Yes", "No", "Later")):
        return PendingPrompt(question=question, choices=[Choice(x) for x in labels])

    def test_reaction_resolves_key_emoji(self):
        reg = PromptRegistry()
        reg.register(self._prompt(), message_timestamp=1234)
        prompt, choice = reg.resolve_reaction(1234, KEY_EMOJIS[1])
        assert choice.label == "No"
        assert prompt.question == "Deploy?"

    def test_reaction_with_unmapped_emoji_and_unknown_message(self):
        reg = PromptRegistry()
        reg.register(self._prompt(), message_timestamp=1234)
        assert reg.resolve_reaction(1234, "🎉") is None
        assert reg.resolve_reaction(9999, KEY_EMOJIS[0]) is None

    def test_custom_choice_emoji_wins(self):
        reg = PromptRegistry()
        reg.register(PendingPrompt("Ship?", [Choice("Yes", emoji="✅"), Choice("No", emoji="🛑")]),
                     message_timestamp=42)
        assert reg.resolve_reaction(42, "✅")[1].label == "Yes"
        assert reg.resolve_reaction(42, KEY_EMOJIS[0]) is None

    def test_poll_vote_resolves_indexes(self):
        reg = PromptRegistry()
        reg.register(self._prompt(), poll_timestamp=777)
        prompt, chosen = reg.resolve_vote(777, [0, 2])
        assert [c.label for c in chosen] == ["Yes", "Later"]

    def test_poll_vote_ignores_out_of_range_indexes(self):
        reg = PromptRegistry()
        reg.register(self._prompt(), poll_timestamp=777)
        _, chosen = reg.resolve_vote(777, [9])
        assert chosen == []

    def test_numbered_reply_resolves_latest_prompt(self):
        reg = PromptRegistry()
        reg.register(self._prompt("Old?", ("a", "b")), message_timestamp=1)
        reg.register(self._prompt("New?", ("c", "d")), message_timestamp=2)
        prompt, chosen = reg.resolve_text("2")
        assert prompt.question == "New?" and chosen[0].label == "d"

    def test_numbered_reply_without_prompt_is_none(self):
        assert PromptRegistry().resolve_text("2") is None

    def test_expired_prompts_are_pruned(self):
        reg = PromptRegistry(ttl=0.0)
        reg.register(self._prompt(), message_timestamp=5, poll_timestamp=6)
        assert reg.resolve_reaction(5, KEY_EMOJIS[0]) is None
        assert reg.resolve_vote(6, [0]) is None
        assert reg.latest() is None

    def test_registry_is_bounded(self):
        reg = PromptRegistry(max_entries=3)
        for ts in range(10):
            reg.register(self._prompt(), message_timestamp=ts)
        assert len(reg._by_message) <= 4


# ---------------------------------------------------------------------------
# Agent-facing descriptions
# ---------------------------------------------------------------------------

class TestDescriptions:
    def test_reaction_with_choice(self):
        prompt = PendingPrompt("Deploy?", [Choice("Yes"), Choice("No")])
        text = describe_reaction(KEY_EMOJIS[1], resolved=(prompt, prompt.choices[1]))
        assert text.startswith("[Signal tap]") and "choice: No" in text and "Deploy?" in text

    def test_reaction_without_choice_and_when_removed(self):
        assert "🎉" in describe_reaction("🎉")
        assert "removed their reaction" in describe_reaction("🎉", resolved=(
            PendingPrompt("q", [Choice("a")]), Choice("a")), removed=True)

    def test_poll_vote_known_and_unknown_poll(self):
        prompt = PendingPrompt("Checks?", [Choice("Tests"), Choice("Lint")])
        known = describe_poll_vote(resolved=(prompt, [prompt.choices[0]]))
        assert "[Signal tap]" in known and "“Tests”" in known
        unknown = describe_poll_vote(option_indexes=[2], question="Other?")
        assert "option(s) [2]" in unknown

    def test_edit_delete_sticker_poll_create(self):
        assert describe_edit("new text").startswith("[Signal edit]") and "new text" in describe_edit("new text")
        assert describe_delete().startswith("[Signal delete]")
        assert describe_sticker("✨") == "[Signal sticker] The user sent a sticker (✨)."
        created = describe_poll_create("Deploy?", ["Yes", "No"], False)
        assert "2 options, one answer" in created

    def test_text_choice(self):
        prompt = PendingPrompt("Deploy?", [Choice("Yes"), Choice("No")])
        text = describe_text_choice((prompt, [prompt.choices[1]]), "2")
        assert "replied “2”" in text and "choice: “No”" in text

    def test_long_text_is_truncated(self):
        text = describe_edit("x" * 1000)
        assert "…" in text and len(text) < 600


# ---------------------------------------------------------------------------
# Markdown → Signal styles: spoiler support
# ---------------------------------------------------------------------------

class TestSpoilerMarkdown:
    def test_double_pipe_becomes_spoiler_style(self):
        from plugins.platforms.signal.signal_format import markdown_to_signal
        text, styles = markdown_to_signal("answer is ||42|| ok")
        assert text == "answer is 42 ok"
        assert styles == ["10:2:SPOILER"]

    def test_spoiler_coexists_with_bold_and_code(self):
        from plugins.platforms.signal.signal_format import markdown_to_signal
        _, styles = markdown_to_signal("**bold** then ||secret|| then `code`")
        assert [s.rsplit(":", 1)[1] for s in styles] == ["BOLD", "SPOILER", "MONOSPACE"]


# ---------------------------------------------------------------------------
# Adapter: inbound taps reach the agent
# ---------------------------------------------------------------------------

def _make_adapter(monkeypatch, account=ACCOUNT):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": account}
    return SignalAdapter(config)


def _sync_env(payload, ts, account=ACCOUNT):
    """A Note-to-Self envelope: the phone sends, the linked device only sees the sync copy."""
    sent = {"timestamp": ts, "destinationNumber": account, **payload}
    return {"envelope": {"sourceNumber": account, "sourceUuid": UUID, "sourceName": "Pierre",
                         "timestamp": ts + 1, "syncMessage": {"sentMessage": sent}}}


def _collect(adapter):
    seen = []

    async def capture(event):
        seen.append(event)

    adapter.handle_message = capture
    return seen


class TestInboundTaps:
    @pytest.mark.asyncio
    async def test_reaction_reaches_agent_and_resolves_registered_prompt(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        adapter.prompts.register(PendingPrompt("Deploy?", [Choice("Yes"), Choice("No")]),
                                 message_timestamp=555)
        await adapter._handle_envelope(_sync_env(
            {"reactionMessage": {"emoji": KEY_EMOJIS[1], "targetAuthor": ACCOUNT,
                                 "targetTimestamp": 555, "timestamp": 556}}, 556))
        assert len(seen) == 1
        assert seen[0].text.startswith("[Signal tap]") and "choice: No" in seen[0].text
        assert seen[0].raw_message["interaction"] == "reaction"

    @pytest.mark.asyncio
    async def test_reaction_without_prompt_still_reaches_agent(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope(_sync_env(
            {"reactionMessage": {"emoji": "🎉", "targetAuthor": ACCOUNT,
                                 "targetTimestamp": 1, "timestamp": 2}}, 2))
        assert len(seen) == 1 and seen[0].text == "[Signal reaction] The user reacted 🎉."

    @pytest.mark.asyncio
    async def test_poll_vote_resolves_to_option_label(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        adapter.prompts.register(PendingPrompt("Checks?", [Choice("Tests"), Choice("Lint")]),
                                 poll_timestamp=900)
        await adapter._handle_envelope(_sync_env(
            {"pollVote": {"pollAuthor": ACCOUNT, "pollTimestamp": 900, "voteCount": 1,
                          "optionIndexes": [1], "timestamp": 901}}, 901))
        assert len(seen) == 1 and "“Lint”" in seen[0].text

    @pytest.mark.asyncio
    async def test_poll_create_and_terminate(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope(_sync_env(
            {"message": "", "pollCreate": {"question": "Deploy?",
                                           "allowMultiple": False,
                                           "options": [{"optionId": 0, "optionText": "Yes"},
                                                       {"optionId": 1, "optionText": "No"}],
                                           "timestamp": 910}}, 910))
        assert "poll “Deploy?”" in seen[0].text
        # The poll is registered, so a later vote by the user resolves without extra bookkeeping.
        await adapter._handle_envelope(_sync_env(
            {"pollVote": {"pollAuthor": ACCOUNT, "pollTimestamp": 910, "voteCount": 1,
                          "optionIndexes": [0], "timestamp": 911}}, 911))
        assert "“Yes”" in seen[1].text
        await adapter._handle_envelope(_sync_env(
            {"pollTerminate": {"pollTimestamp": 910, "timestamp": 912}}, 912))
        assert "was closed" in seen[2].text

    @pytest.mark.asyncio
    async def test_edit_and_delete_reach_agent(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope(_sync_env(
            {"editMessage": {"targetSentTimestamp": 1,
                             "dataMessage": {"timestamp": 2, "message": "corrected text"}}}, 2))
        assert seen[0].text == "[Signal edit] The user edited a message to: “corrected text”"
        await adapter._handle_envelope(_sync_env({"remoteDelete": {"targetSentTimestamp": 1}}, 3))
        assert seen[1].text.startswith("[Signal delete]")

    @pytest.mark.asyncio
    async def test_sticker_only_message_reaches_agent(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope(_sync_env(
            {"message": "", "sticker": {"stickerId": 0, "packId": "abc", "emoji": "✨"}}, 4))
        assert seen[0].text == "[Signal sticker] The user sent a sticker (✨)."

    @pytest.mark.asyncio
    async def test_numbered_reply_taps_pending_card(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        adapter.prompts.register(PendingPrompt("Deploy?", [Choice("Yes"), Choice("No")]),
                                 message_timestamp=10)
        await adapter._handle_envelope(_sync_env({"message": "2"}, 20))
        assert seen[0].text.startswith("[Signal tap]") and "choice: “No”" in seen[0].text

    @pytest.mark.asyncio
    async def test_ordinary_text_is_untouched_by_the_registry(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        adapter.prompts.register(PendingPrompt("Deploy?", [Choice("Yes"), Choice("No")]),
                                 message_timestamp=10)
        await adapter._handle_envelope(_sync_env({"message": "ship it when you can, thanks"}, 20))
        assert seen[0].text == "ship it when you can, thanks"

    @pytest.mark.asyncio
    async def test_no_registry_prompt_leaves_text_alone(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope(_sync_env({"message": "2"}, 20))
        assert seen[0].text == "2"

    @pytest.mark.asyncio
    async def test_self_authored_non_sync_envelope_still_dropped(self, monkeypatch):
        """Loop protection: our own plain sends must never come back as agent turns."""
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope({"envelope": {
            "sourceNumber": ACCOUNT, "timestamp": 30,
            "reactionMessage": {"emoji": "👍", "targetTimestamp": 1, "timestamp": 30}}})
        assert seen == []

    @pytest.mark.asyncio
    async def test_group_reaction_is_gated_by_group_allowlist(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope(_sync_env(
            {"reactionMessage": {"emoji": "👍", "targetTimestamp": 1, "timestamp": 2,
                                 "groupInfo": {"groupId": "Z3JvdXA=", "groupName": "Team"}}}, 2))
        assert seen == []

    @pytest.mark.asyncio
    async def test_typing_and_receipts_stay_silent(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        seen = _collect(adapter)
        await adapter._handle_envelope(_sync_env({"typingMessage": {"action": "STARTED"}}, 40))
        await adapter._handle_envelope(_sync_env({"receiptMessage": {"type": "READ", "timestamps": [1]}}, 41))
        assert seen == []


# ---------------------------------------------------------------------------
# Adapter: outbound prompt / quote / edit params
# ---------------------------------------------------------------------------

class TestOutboundPrompts:
    @pytest.mark.asyncio
    async def test_send_prompt_uses_poll_when_options_fit(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_rpc(method, params, rpc_id=None, **kw):
            sent.append((method, dict(params)))
            return {"timestamp": 4242, "results": [{"type": "SUCCESS"}]}

        adapter._rpc = fake_rpc
        result = await adapter.send_prompt(ACCOUNT, "Deploy to prod?", ["Yes", "No"])
        assert result.success and result.message_id == "4242"
        method, params = sent[-1]
        assert method == "sendPollCreate"
        assert params["question"] == "Deploy to prod?" and params["options"] == ["Yes", "No"]
        assert adapter.prompts.for_poll(4242).question == "Deploy to prod?"

    @pytest.mark.asyncio
    async def test_send_prompt_falls_back_to_numbered_card(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_rpc(method, params, rpc_id=None, **kw):
            sent.append((method, dict(params)))
            return {"timestamp": 4343, "results": [{"type": "SUCCESS"}]}

        adapter._rpc = fake_rpc
        choices = [f"option number {i}" for i in range(1, 12)]  # 11 options exceed Signal's poll cap
        result = await adapter.send_prompt(ACCOUNT, "Pick one", choices)
        method, params = sent[-1]
        assert method == "send" and "1) option number 1" in params["message"]
        assert isinstance(params["textStyles"], list)
        assert result.success and adapter.prompts.for_message(4343).kind == "card"

    @pytest.mark.asyncio
    async def test_send_prompt_rejects_empty_choices(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        result = await adapter.send_prompt(ACCOUNT, "Q?", [])
        assert result.success is False

    @pytest.mark.asyncio
    async def test_quote_params_only_for_numeric_ids(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        params = adapter._quote_params(ACCOUNT, "1792790367553", {"reply_to_text": "hi"})
        assert params == {"quoteTimestamp": 1792790367553, "quoteAuthor": ACCOUNT, "quoteMessage": "hi"}
        assert adapter._quote_params(ACCOUNT, "not-a-timestamp", None) == {}
        assert adapter._quote_params(ACCOUNT, None, None) == {}
        group = adapter._quote_params("group:Z3JvdXA=", "5", None)
        assert group["quoteAuthor"] == ACCOUNT

    @pytest.mark.asyncio
    async def test_send_attaches_quote_params_to_every_chunk(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_rpc(method, params, rpc_id=None, **kw):
            sent.append(dict(params))
            return {"timestamp": 10 + len(sent), "results": [{"type": "SUCCESS"}]}

        adapter._rpc = fake_rpc
        adapter.MAX_MESSAGE_LENGTH = 40
        await adapter.send(ACCOUNT, "x " * 60, reply_to="777", metadata={"reply_to_text": "orig"})
        sends = [p for p in sent if "message" in p]
        assert len(sends) > 1
        assert all(p["quoteTimestamp"] == 777 for p in sends)

    @pytest.mark.asyncio
    async def test_edit_message_uses_edit_timestamp(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        sent = []

        async def fake_rpc(method, params, rpc_id=None, **kw):
            sent.append((method, dict(params)))
            return {"timestamp": 999, "results": [{"type": "SUCCESS"}]}

        adapter._rpc = fake_rpc
        result = await adapter.edit_message(ACCOUNT, "888", "new text")
        assert result.success
        assert sent[-1][1]["editTimestamp"] == 888 and sent[-1][1]["message"] == "new text"
        bad = await adapter.edit_message(ACCOUNT, "", "x")
        assert bad.success is False

    @pytest.mark.asyncio
    async def test_send_editable_returns_timestamp_as_message_id(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        async def fake_rpc(method, params, rpc_id=None, **kw):
            return {"timestamp": 5150, "results": [{"type": "SUCCESS"}]}

        adapter._rpc = fake_rpc
        result = await adapter.send_editable(ACCOUNT, "hello")
        assert result.message_id == "5150"
