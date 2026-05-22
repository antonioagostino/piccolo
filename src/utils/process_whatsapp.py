"""
WhatsApp 1-to-1 chat export pre-processor.

Parses a WhatsApp .txt export file, strips invisible Unicode characters and
media placeholders, reassembles multi-line messages, and splits the cleaned
message stream into conversation sessions separated by a configurable silence
gap.

Public API
----------
load_whatsapp_sessions(path, session_gap_minutes) -> list[str]
    Top-level helper: parse → clean → segment → format.
    Each returned string is one session formatted as "Sender: text\\n" lines.

Lower-level helpers are also importable for custom pipelines:
    parse_chat_file, clean_messages, segment_conversations, format_session
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Union

# ──────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ──────────────────────────────────────────────────────────────────────────────

# Optional leading LRM (U+200E), then [DD/MM/YY, HH:MM:SS] Sender: text
_RE_MESSAGE = re.compile(
    r"^‎?\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}:\d{2})\]\s+"
    r"([^:]+):\s*(.*)"
)

# Invisible / directional Unicode characters to strip from text
_INVISIBLE_RE = re.compile(
    r"[‎‏‪-‮⁦-⁩﻿­]"
)

# Italian WhatsApp media placeholder texts (text after stripping invisible chars)
_MEDIA_RE = re.compile(
    r"^(immagine|video|audio|sticker|gif|documento|contatto)\s+omess[oa]$",
    re.IGNORECASE,
)

# Deleted-message notifications
_DELETED_RE = re.compile(
    r"(questo|hai inviato un)\s+messaggio\s+(è stato|che è stato)\s+eliminato",
    re.IGNORECASE,
)

# System / WhatsApp notification patterns (no sender, or well-known sentence)
_SYSTEM_RE = re.compile(
    r"(crittografati end-to-end"
    r"|sicurezza del (tuo|vostro) account"
    r"|ha cambiato il (numero|nome)"
    r"|ha (aggiunto|rimosso|lasciato)"
    r"|messaggi e le chiamate sono)",
    re.IGNORECASE,
)

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Message:
    timestamp: datetime
    sender: str
    text: str


# ──────────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_datetime(date_str: str, time_str: str) -> datetime | None:
    combined = f"{date_str.strip()} {time_str.strip()}"
    for fmt in ("%d/%m/%y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
        try:
            return datetime.strptime(combined, fmt)
        except ValueError:
            continue
    return None


def parse_chat_file(path: Union[str, Path]) -> list[Message]:
    """
    Parse a WhatsApp .txt export into a list of Message objects.

    Multi-line messages (continuation lines without a timestamp header) are
    appended to the previous message's text, separated by a space.
    Lines that cannot be parsed as a timestamped message and have no preceding
    context are silently dropped.
    """
    messages: list[Message] = []
    current: Message | None = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            m = _RE_MESSAGE.match(line)
            if m:
                if current is not None:
                    messages.append(current)
                date_s, time_s, sender, text = m.groups()
                ts = _parse_datetime(date_s, time_s)
                if ts is None:
                    current = None
                    continue
                current = Message(timestamp=ts, sender=sender.strip(), text=text)
            else:
                # Continuation line — append to the current message
                stripped = line.strip()
                if current is not None and stripped:
                    current.text += " " + stripped

    if current is not None:
        messages.append(current)

    return messages


# ──────────────────────────────────────────────────────────────────────────────
# Cleaning
# ──────────────────────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Strip invisible chars and collapse whitespace."""
    text = _INVISIBLE_RE.sub("", raw)
    text = " ".join(text.split())
    return text


def clean_messages(messages: list[Message]) -> list[Message]:
    """
    Remove media placeholders, deleted-message notices, system notifications,
    and any message whose text becomes empty after cleaning.
    """
    cleaned: list[Message] = []
    for msg in messages:
        text = _clean_text(msg.text)
        if not text:
            continue
        if _MEDIA_RE.match(text):
            continue
        if _DELETED_RE.search(text):
            continue
        if _SYSTEM_RE.search(text):
            continue
        cleaned.append(Message(timestamp=msg.timestamp, sender=msg.sender, text=text))
    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Segmentation
# ──────────────────────────────────────────────────────────────────────────────

def segment_conversations(
    messages: list[Message],
    session_gap_minutes: int = 30,
) -> list[list[Message]]:
    """
    Split a flat message list into conversation sessions.

    A new session begins when the gap between consecutive messages exceeds
    *session_gap_minutes*.  Returns only non-empty sessions.
    """
    if not messages:
        return []

    sessions: list[list[Message]] = []
    current_session: list[Message] = [messages[0]]
    gap = timedelta(minutes=session_gap_minutes)

    for prev, curr in zip(messages, messages[1:]):
        if curr.timestamp - prev.timestamp > gap:
            sessions.append(current_session)
            current_session = [curr]
        else:
            current_session.append(curr)

    sessions.append(current_session)
    return [s for s in sessions if s]


# ──────────────────────────────────────────────────────────────────────────────
# Formatting
# ──────────────────────────────────────────────────────────────────────────────

def format_session(session: list[Message]) -> str:
    """
    Render a session as a plain-text string.

    Each message becomes one line: ``"Sender: text\\n"``.
    """
    return "\n".join(f"{msg.sender}: {msg.text}" for msg in session)


# ──────────────────────────────────────────────────────────────────────────────
# Top-level helper
# ──────────────────────────────────────────────────────────────────────────────

def load_whatsapp_sessions(
    path: Union[str, Path],
    session_gap_minutes: int = 30,
) -> list[str]:
    """
    Parse, clean, segment, and format a WhatsApp chat export.

    Args:
        path: Path to the WhatsApp .txt export file.
        session_gap_minutes: Minutes of silence that mark a new session.

    Returns:
        List of session strings, each formatted as ``"Sender: text\\n"`` lines.
    """
    messages = parse_chat_file(path)
    messages = clean_messages(messages)
    sessions = segment_conversations(messages, session_gap_minutes)
    return [format_session(s) for s in sessions]
