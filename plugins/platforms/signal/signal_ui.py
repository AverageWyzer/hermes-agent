"""Signal interaction primitives.

Signal has no inline keyboards or bot API: there is no way to attach a tappable button to a
message. The closest native affordances are:

* **Polls** — a real multiple-choice widget rendered by Signal's own client (2–10 options, one or
  many selectable). This is the only true "button" UI Signal offers.
* **Numbered reply cards** — a plain message with a numbered list; the user replies `2` (or taps
  the matching key emoji reaction) and the number selects the option.
* **Emoji-reaction keys** — the user reacts to the prompt message with the emoji shown next to an
  option; the reaction is the "click".

This module renders those prompts and turns the user's tap back into one line of text the agent can
read, because the gateway adapter drops reaction / poll-vote / edit / delete envelopes otherwise.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

# Signal caps a poll at 10 options and 100 characters per option (measured against signal-cli 0.14.9).
MAX_POLL_OPTIONS = 10
MAX_POLL_OPTION_LENGTH = 100
MIN_POLL_OPTIONS = 2

# The key emojis shown next to each option; a reaction with one of these is a choice.
KEY_EMOJIS = ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟")

# "2", "2.", "2)", "#2", "option 2", "two" — an unambiguous reply to a numbered card.
_NUMBER_REPLY_RE = re.compile(r"^\s*(?:#|option\s+|choice\s+)?([1-9])\s*[.)]?\s*$", re.IGNORECASE)

_WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


@dataclass
class Choice:
    """One option of a prompt. ``value`` is what the agent receives; ``label`` is what the user sees."""

    label: str
    value: Optional[str] = None
    emoji: Optional[str] = None

    def __post_init__(self) -> None:
        if self.value is None:
            self.value = self.label


@dataclass
class PendingPrompt:
    """A prompt that is waiting for the user's tap."""

    question: str
    choices: list[Choice]
    kind: str = "card"  # "card" | "poll"
    message_timestamp: Optional[int] = None
    poll_timestamp: Optional[int] = None
    created_at: float = field(default_factory=time.monotonic)
    extra: dict = field(default_factory=dict)


def _utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def _styles_for(text: str, spans: Iterable[tuple[int, int, str]]) -> list[str]:
    """Convert (codepoint_start, codepoint_len, style) spans into Signal ``start:length:STYLE`` ranges."""
    out: list[str] = []
    for start, length, style in spans:
        if length <= 0:
            continue
        u16_start = _utf16_len(text[:start])
        u16_len = _utf16_len(text[start:start + length])
        out.append(f"{u16_start}:{u16_len}:{style}")
    return out


def render_choice_card(question: str, choices: list[Choice], *, multi: bool = False,
                       hint: bool = True) -> tuple[str, list[str]]:
    """Render a numbered "button card" and its native text styles.

    Returns ``(text, textStyles)``; ``textStyles`` is empty (never a bare string) so callers can pass
    it straight to ``textStyles``.
    """
    if not choices:
        raise ValueError("a choice card needs at least one choice")
    lines = [question, ""]
    spans: list[tuple[int, int, str]] = []
    cursor = len(question) + 1  # question + "\n"
    spans.append((0, len(question), "BOLD"))
    for idx, choice in enumerate(choices, start=1):
        emoji = (choice.emoji or KEY_EMOJIS[idx - 1]) if idx <= len(KEY_EMOJIS) else ""
        prefix = f"{idx}) "
        line = f"{prefix}{choice.label}" + (f"  {emoji}" if emoji else "")
        line_start = cursor
        cursor += len(line) + 1
        spans.append((line_start, len(prefix), "BOLD"))
        lines.append(line)
    if hint:
        hint_text = ("Reply with the numbers (e.g. 1 3), or react with the emoji next to an option."
                     if multi else
                     "Reply with the number, or react with the emoji next to your choice.")
        lines += ["", hint_text]
        cursor += 1
        spans.append((cursor, len(hint_text), "ITALIC"))
    text = "\n".join(lines)
    return text, _styles_for(text, spans)


def poll_eligible(choices: list[Choice]) -> bool:
    """True when the choices fit in a native Signal poll."""
    return (MIN_POLL_OPTIONS <= len(choices) <= MAX_POLL_OPTIONS
            and all(0 < len(c.label) <= MAX_POLL_OPTION_LENGTH for c in choices))


def poll_payload(question: str, choices: list[Choice], *, multi: bool = False) -> dict:
    """``sendPollCreate`` params (minus account/recipients) for a set of choices."""
    return {"question": question, "options": [c.label for c in choices], "multiSelect": multi}


def parse_choice_reply(text: str, choices: int = MAX_POLL_OPTIONS) -> Optional[int]:
    """Return the 1-based option index a free-text reply selects, or None.

    Accepts ``2``, ``2.``, ``2)``, ``#2``, ``option 2``, ``two`` and the key emojis.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped in KEY_EMOJIS:
        idx = KEY_EMOJIS.index(stripped) + 1
        return idx if idx <= choices else None
    if match := _NUMBER_REPLY_RE.match(stripped):
        idx = int(match.group(1))
        return idx if idx <= choices else None
    word = _WORD_NUMBERS.get(stripped.lower().rstrip("."))
    return word if word and word <= choices else None


def parse_multi_reply(text: str, choices: int = MAX_POLL_OPTIONS) -> list[int]:
    """Parse a multi-select reply such as ``1 3`` / ``1,3`` / ``1 and 3`` into sorted indices."""
    indices: set[int] = set()
    for token in re.split(r"[\s,;+&]+| and ", (text or "").strip()):
        idx = parse_choice_reply(token, choices)
        if idx:
            indices.add(idx)
    return sorted(indices)


class PromptRegistry:
    """Remembers which message/poll is waiting for a tap, so a reaction or vote resolves to a choice."""

    def __init__(self, ttl: float = 1800.0, max_entries: int = 64) -> None:
        self.ttl = ttl
        self.max_entries = max_entries
        self._by_message: dict[str, PendingPrompt] = {}
        self._by_poll: dict[str, PendingPrompt] = {}

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.ttl
        for store in (self._by_message, self._by_poll):
            for key in [k for k, p in store.items() if p.created_at < cutoff]:
                store.pop(key, None)
        while len(self._by_message) > self.max_entries:
            self._by_message.pop(next(iter(self._by_message)))
        while len(self._by_poll) > self.max_entries:
            self._by_poll.pop(next(iter(self._by_poll)))

    def register(self, prompt: PendingPrompt, *, message_timestamp: Optional[int] = None,
                 poll_timestamp: Optional[int] = None) -> PendingPrompt:
        self._prune()
        if message_timestamp:
            prompt.message_timestamp = int(message_timestamp)
            self._by_message[str(int(message_timestamp))] = prompt
        if poll_timestamp:
            prompt.poll_timestamp = int(poll_timestamp)
            self._by_poll[str(int(poll_timestamp))] = prompt
        return prompt

    def for_message(self, timestamp) -> Optional[PendingPrompt]:
        self._prune()
        return self._by_message.get(str(timestamp)) if timestamp is not None else None

    def for_poll(self, timestamp) -> Optional[PendingPrompt]:
        self._prune()
        return self._by_poll.get(str(timestamp)) if timestamp is not None else None

    def latest(self) -> Optional[PendingPrompt]:
        self._prune()
        prompts = [*self._by_message.values(), *self._by_poll.values()]
        return max(prompts, key=lambda p: p.created_at) if prompts else None

    def resolve_reaction(self, target_timestamp, emoji: str) -> Optional[tuple[PendingPrompt, Choice]]:
        """Resolve a reaction on a prompt message to (prompt, choice)."""
        prompt = self.for_message(target_timestamp)
        if not prompt or not emoji:
            return None
        for idx, choice in enumerate(prompt.choices, start=1):
            key = choice.emoji or (KEY_EMOJIS[idx - 1] if idx <= len(KEY_EMOJIS) else None)
            if key == emoji:
                return prompt, choice
        return None

    def resolve_vote(self, poll_timestamp, option_indexes: Iterable[int]) -> Optional[tuple[PendingPrompt, list[Choice]]]:
        """Resolve poll option indexes to (prompt, chosen choices)."""
        prompt = self.for_poll(poll_timestamp)
        if not prompt:
            return None
        chosen = [prompt.choices[i] for i in option_indexes if 0 <= i < len(prompt.choices)]
        return prompt, chosen

    def resolve_text(self, text: str) -> Optional[tuple[PendingPrompt, list[Choice]]]:
        """Resolve a free-text numbered reply against the most recent pending prompt."""
        prompt = self.latest()
        if not prompt:
            return None
        indices = (parse_multi_reply(text, len(prompt.choices)) if len(prompt.choices) > 1 and "," in text
                   else ([i] if (i := parse_choice_reply(text, len(prompt.choices))) else []))
        if not indices:
            return None
        return prompt, [prompt.choices[i - 1] for i in indices]


def _quote(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_reaction(emoji: str, *, resolved: Optional[tuple[PendingPrompt, Choice]] = None,
                      target_text: str = "", removed: bool = False) -> str:
    """Agent-facing text for an inbound reaction."""
    if resolved:
        prompt, choice = resolved
        action = "removed their reaction" if removed else f"reacted {emoji}"
        return (f"[Signal tap] The user {action} on your prompt “{_quote(prompt.question, 80)}” "
                f"→ choice: {choice.label}")
    action = "removed a reaction" if removed else f"reacted {emoji}"
    tail = f" on your message “{_quote(target_text)}”" if target_text else ""
    return f"[Signal reaction] The user {action}{tail}."


def describe_poll_vote(*, resolved: Optional[tuple[PendingPrompt, list[Choice]]] = None,
                       question: str = "", option_indexes: Optional[list[int]] = None) -> str:
    """Agent-facing text for an inbound poll vote."""
    if resolved:
        prompt, chosen = resolved
        labels = ", ".join(f"“{c.label}”" for c in chosen) or "no option"
        return f"[Signal tap] The user voted {labels} in your poll “{_quote(prompt.question, 80)}”"
    if option_indexes:
        return (f"[Signal poll vote] The user voted option(s) {option_indexes} "
                f"in “{_quote(question, 80)}” (the poll was not one of yours).")
    return "[Signal poll vote] The user withdrew their vote."


def describe_poll_terminate(question: str = "") -> str:
    return f"[Signal tap] The poll “{_quote(question, 80)}” was closed."


def describe_edit(new_text: str, target_text: str = "") -> str:
    tail = f" (was: “{_quote(target_text)}”)" if target_text else ""
    return f"[Signal edit] The user edited a message to: “{_quote(new_text, 400)}”{tail}"


def describe_delete(target_text: str = "") -> str:
    tail = f" (“{_quote(target_text)}”)" if target_text else ""
    return f"[Signal delete] The user deleted a message{tail}."


def describe_text_choice(resolved: tuple[PendingPrompt, list[Choice]], raw_text: str) -> str:
    """Agent-facing text for a numbered reply that taps a choice card."""
    prompt, chosen = resolved
    labels = ", ".join(f"“{c.label}”" for c in chosen)
    return (f"[Signal tap] The user replied “{_quote(raw_text, 20)}” to your prompt "
            f"“{_quote(prompt.question, 80)}” → choice: {labels}")


def describe_poll_create(question: str, options: list[str], multi: bool) -> str:
    kind = "multiple answers allowed" if multi else "one answer"
    return (f"[Signal poll] The user created a poll “{_quote(question, 80)}” "
            f"({len(options)} options, {kind}): " + "; ".join(_quote(o, 40) for o in options[:10]))


def describe_sticker(emoji: str = "") -> str:
    tail = f" ({emoji})" if emoji else ""
    return f"[Signal sticker] The user sent a sticker{tail}."

