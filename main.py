"""
Ollama chat TUI — single-file, well-architected.

Usage:
    python main.py <directory_path>

Architecture:
    Symbol / TextBox / Canvas  – low-level terminal drawing primitives (unchanged)
    AppState                   – thread-safe shared state
    OllamaWorker               – streams Ollama responses on a background thread
    Renderer                   – builds each frame from AppState
    helpers                    – file-tree / prompt / notification utilities
    main()                     – event loop: render → wait for input → dispatch
"""

from __future__ import annotations

import fnmatch
import getpass
import os
import platform
import re
import sys
import termios
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, List, Optional, Tuple

import ollama
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from plyer import notification
import requests
import select
import subprocess
import tty


# ═══════════════════════════════════════════════════════════════════════════════
# Theme  — all colour definitions in one place
# ═══════════════════════════════════════════════════════════════════════════════


class Theme:
    """Central colour palette.  Change values here to restyle the entire UI."""

    # ── Chat ──────────────────────────────────────────────────────────────────
    USER_TEXT = "%color(120,180,255)"  # user bubble text
    USER_HEADER = "%color(70,70,70)"  # username / model name label
    ASST_THOUGHT = "%color(80,80,80)"  # "thought for Xs." line
    ASST_HEADER = "%color(70,70,70)"  # model name label

    # ── Think block ───────────────────────────────────────────────────────────
    THINK_GUTTER = "%color(80,80,80)"  # │ prefix
    THINK_TEXT = "%color(100,100,100)"  # plain thinking text
    THINK_HEADER = "%color(55,55,75)"  # "o · collapse thinking" label
    THINK_PENDING = "%color(160,140,60)"  # ◎ symbol colour
    THINK_PENDING_T = "%color(150,140,100)"  # pending label text
    THINK_DONE = "%color(70,160,100)"  # ✔ symbol colour
    THINK_DONE_T = "%color(100,120,100)"  # done label text
    THINK_ERROR = "%color(210,70,70)"  # ✘ symbol + error label

    # ── Status bar ────────────────────────────────────────────────────────────
    SPIN = "%color(210,120,40)"  # spinner colour
    HINT = "%color(65,65,65)"  # idle hint text
    HINT_SCROLL = "%color(70,70,90)"  # scroll-mode hint text

    # ── Diff stats (tool done line) ───────────────────────────────────────────
    DIFF_ADD = "%color(80,180,100)"  # +N added lines
    DIFF_DEL = "%color(200,80,80)"  # -N removed lines
    DIFF_NEUTRAL = "%color(90,90,90)"  # line count / neutral stat

    # ── Syntax highlighting ───────────────────────────────────────────────────
    SYN_DEFAULT = "%reset%color(185,185,185)"
    SYN_KEYWORD = "%reset%color(140,120,210)"
    SYN_STRING = "%reset%color(180,160,80)"
    SYN_NUMBER = "%reset%color(210,130,80)"
    SYN_COMMENT = "%reset%color(105,120,85)"
    SYN_GUTTER = "%color(55,55,70)"

    # ── Canvas border ─────────────────────────────────────────────────────────
    BORDER = "\u001b[38;2;55;55;65m"  # raw ANSI — used before markup engine
    BORDER_RESET = "\u001b[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# Drawing primitives
# ═══════════════════════════════════════════════════════════════════════════════


class Symbol:
    """A single drawable character plus optional ANSI escape sequence."""

    def __init__(self, character: str = "\0", escape_sequence: str = "\u001b[0m"):
        self.content: str = character
        self.escape_sequence: str = escape_sequence

    def __eq__(self, other) -> bool:
        if isinstance(other, Symbol):
            return (
                self.content == other.content
                and self.escape_sequence == other.escape_sequence
            )
        if isinstance(other, str):
            return self.content == other
        return NotImplemented

    def __repr__(self) -> str:
        return f"Symbol({self.content!r})"

    def __str__(self) -> str:
        return f"{self.escape_sequence}{self.content}"

    @staticmethod
    def from_string(text: str) -> List[Symbol]:
        """Parse a markup string into a list of Symbol objects.

        Supported markup tags (prefixed with %):
            %reset          – reset colour/style
            %color(r,g,b)   – set 24-bit foreground colour
            %bold           – bold text
            %underline      – underline text
        """
        symbols: List[Symbol] = []
        escape_sequence = "\u001b[0m"
        i = 0

        while i < len(text):
            index = i
            i += 1

            if text[index] != "%":
                symbols.append(Symbol(text[index], escape_sequence))
                continue

            def peek(tag: str) -> bool:
                end = index + 1 + len(tag)
                return text[index + 1 : end] == tag

            if peek("reset"):
                escape_sequence = "\u001b[0m"
                i = index + 6
            elif peek("color("):
                rest = text[index + 7 :]
                end = rest.find(")")
                if end == -1:
                    symbols.append(Symbol("%", escape_sequence))
                    continue
                parts = rest[:end].split(",")
                if len(parts) != 3:
                    symbols.append(Symbol("%", escape_sequence))
                    continue
                try:
                    r, g, b = int(parts[0]), int(parts[1]), int(parts[2])
                except ValueError:
                    symbols.append(Symbol("%", escape_sequence))
                    continue
                escape_sequence += f"\u001b[38;2;{r};{g};{b}m"
                i = index + 7 + end + 1
            elif peek("bold"):
                escape_sequence += "\u001b[1m"
                i = index + 5
            elif peek("underline"):
                escape_sequence += "\u001b[4m"
                i = index + 10
            else:
                symbols.append(Symbol("%", escape_sequence))

        return symbols

    @staticmethod
    def buffer(width: int, height: int) -> List[List[Symbol]]:
        return [[Symbol(" ") for _ in range(width)] for _ in range(height)]

    @staticmethod
    def null_buffer(width: int, height: int) -> List[List[Symbol]]:
        return [[Symbol("\0") for _ in range(width)] for _ in range(height)]


class Alignment(Enum):
    min = auto()
    center = auto()
    max = auto()


class TextBox:
    """Simple text-layout helper with horizontal/vertical alignment."""

    def __init__(
        self,
        width: int,
        height: int,
        text: Optional[str] = None,
        horizontal_alignment: Alignment = Alignment.min,
        vertical_alignment: Alignment = Alignment.min,
    ):
        self.width = width
        self.height = height
        self.horizontal_alignment = horizontal_alignment
        self.vertical_alignment = vertical_alignment
        self._buffer: List[List[Symbol]] = Symbol.buffer(width, height)

        if text is not None:
            self.content(text)
        else:
            self.update_alignment()

    def clear(self) -> None:
        self._buffer = Symbol.buffer(self.width, self.height)

    def get(self, x: int, y: int) -> Symbol:
        if x >= self.width or y >= self.height:
            return Symbol()
        return self._buffer[y][x]

    def content(self, text: str) -> None:
        symbols = Symbol.from_string(text)
        y, x = 0, 0
        for ch in symbols:
            if x >= self.width or ch.content == "\n":
                y += 1
                if y >= self.height:
                    break
                x = 0
                continue
            self._buffer[y][x] = ch
            x += 1
        self.update_alignment()

    def update_alignment(self) -> None:
        self.align_horizontal(self.horizontal_alignment)
        self.align_vertical(self.vertical_alignment)

    def align_horizontal(self, align: Alignment) -> None:
        if align == Alignment.min:
            return
        self.horizontal_alignment = align
        new_buffer = Symbol.buffer(self.width, self.height)
        for y in range(self.height):
            spaces = sum(
                1
                for x in range(self.width)
                if self._buffer[y][x].content in (" ", "\t", "\0")
            )
            add = spaces // 2 if align == Alignment.center else spaces
            for x in range(self.width):
                if x + add >= self.width:
                    break
                new_buffer[y][x + add] = self._buffer[y][x]
        self._buffer = new_buffer

    def align_vertical(self, align: Alignment) -> None:
        if align == Alignment.min:
            return
        self.vertical_alignment = align
        new_buffer = Symbol.buffer(self.width, self.height)
        newlines = 0
        for y in range(self.height):
            has_content = any(
                self._buffer[y][x].content not in (" ", "\t", "\0")
                for x in range(self.width)
            )
            newlines = 0 if has_content else newlines + 1
        add = newlines // 2 if align == Alignment.center else newlines
        for y in range(self.height):
            if y + add >= self.height:
                break
            for x in range(self.width):
                new_buffer[y + add][x] = self._buffer[y][x]
        self._buffer = new_buffer


class Canvas:
    """Console-based canvas. Writes only changed cells to reduce flicker."""

    _IS_WINDOWS = platform.system() == "Windows"

    def __init__(self, w: int, h: int):
        self._terminal_width: int = 0
        self._terminal_height: int = 0
        self._is_cursor_hidden: bool = False
        self._position_x: int = 0
        self._position_y: int = 2
        self.width: int = 0
        self.height: int = 0
        self._cache_buffer: List[List[Symbol]] = Symbol.buffer(0, 0)
        self._frame_buffer: List[List[Symbol]] = Symbol.buffer(0, 0)
        self.resize(w, h)

    # ── Terminal helpers ────────────────────────────────────────────────────

    def _get_terminal_size(self) -> Tuple[int, int]:
        size = os.get_terminal_size()
        return size.columns, size.lines

    # ── Cursor helpers ──────────────────────────────────────────────────────

    def hide_cursor(self) -> None:
        self._is_cursor_hidden = True
        sys.stdout.write("\u001b[?25l")
        sys.stdout.flush()

    def show_cursor(self) -> None:
        self._is_cursor_hidden = False
        sys.stdout.write("\u001b[?25h")
        sys.stdout.flush()

    def toggle_cursor(self) -> None:
        if self._is_cursor_hidden:
            self.show_cursor()
        else:
            self.hide_cursor()

    def set_cursor_position(self, left: int, top: int) -> None:
        left, top = max(left, 0), max(top, 0)
        try:
            tw, th = self._get_terminal_size()
            flag = left > self.width + self._position_x or top > self.height + 3
            if flag:
                left = min(self.width + self._position_x, tw - 1)
                top = min(self.height + 3, th - 1)
            sys.stdout.write(f"\u001b[{top + 1};{left + 1}H")
            sys.stdout.flush()
        except Exception:
            pass

    # ── Reset / resize ──────────────────────────────────────────────────────

    def reset_console(self) -> None:
        self._cache_buffer = Symbol.buffer(self.width, self.height)
        self._frame_buffer = Symbol.buffer(self.width, self.height)
        sys.stdout.write("\u001b[0m\u001b[2J")
        sys.stdout.flush()
        self.show_cursor()

    def clear(self) -> None:
        self.resize(self.width, self.height)

    def resize(self, width: int = None, height: int = None) -> None:  # type: ignore[assignment]
        if width is None or height is None:
            tw, th = self._get_terminal_size()
            width = width if width is not None else tw
            height = height if height is not None else th

        self.reset_console()
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

        tw, th = self._get_terminal_size()
        self._terminal_width = tw
        self._terminal_height = th
        self.hide_cursor()
        self.width = width
        self.height = height
        self._position_x = max(1, tw // 2 - self.width // 2)
        self._cache_buffer = Symbol.buffer(self.width, self.height)
        self._frame_buffer = Symbol.buffer(self.width, self.height)

        _B = Theme.BORDER
        _R = Theme.BORDER_RESET
        buf: List[str] = []
        for y in range(-1, self.height + 1):
            for x in range(-1, self.width + 1):
                col = self._position_x + x
                row = self._position_y + y
                buf.append(f"\u001b[{row + 1};{col + 1}H")
                if y == -1:
                    buf.append(
                        _B
                        + ("╭" if x == -1 else ("╮" if x == self.width else "─"))
                        + _R
                    )
                elif y == self.height:
                    buf.append(
                        _B
                        + ("╰" if x == -1 else ("╯" if x == self.width else "─"))
                        + _R
                    )
                elif x == -1 or x == self.width:
                    buf.append(_B + "│" + _R)
                else:
                    buf.append(" ")

        sys.stdout.write("".join(buf))
        sys.stdout.flush()

    # ── Drawing primitives ──────────────────────────────────────────────────

    def text(
        self, start_x: int, start_y: int, text: str, overflow: bool = True
    ) -> None:
        symbols = Symbol.from_string(text)
        y, x = start_y, 0
        for symbol in symbols:
            if (start_x + x >= self.width and overflow) or symbol.content == "\n":
                y += 1
                if y >= self.height:
                    break
                x = 0
                continue
            dx = x
            x += 1
            if start_x + dx < 0 or y < 0:
                continue
            if start_x + dx >= self.width and not overflow:
                continue
            self._frame_buffer[y][start_x + dx] = symbol

    def text_box(self, start_x: int, start_y: int, text_box: TextBox) -> None:
        for tb_y in range(text_box.height):
            y = start_y + tb_y
            if y >= self.height:
                break
            for tb_x in range(text_box.width):
                x = start_x + tb_x
                if x >= self.width:
                    break
                s = text_box.get(tb_x, tb_y)
                if s.content != "\0" and s.content not in (" ", "\t"):
                    self._frame_buffer[y][x] = s

    # ── Flush ───────────────────────────────────────────────────────────────

    def draw(self) -> None:
        tw, th = self._get_terminal_size()
        if tw != self._terminal_width or th != self._terminal_height:
            self._terminal_width = tw
            self._terminal_height = th
            self.resize(self.width, self.height)
            self._cache_buffer = Symbol.buffer(self.width, self.height)

        if self.height > self._terminal_height or self.width > self._terminal_width:
            midtext = "Please resize the console!"
            subtext = "(Minimum 96x36)"
            mx = min(self._terminal_width, self.width) // 2
            my = min(self._terminal_height, self.height) // 2
            self.set_cursor_position(mx - len(midtext) // 2, my)
            sys.stdout.write(midtext)
            self.set_cursor_position(mx - len(subtext) // 2, my + 1)
            sys.stdout.write(subtext)
            self.move_cursor_to_bottom(0)
            sys.stdout.flush()
            return

        out: List[str] = []
        for y in range(min(self.height, self._terminal_height)):
            for x in range(min(self.width, self._terminal_width)):
                cache_sym = self._cache_buffer[y][x]
                frame_sym = self._frame_buffer[y][x]
                if cache_sym == frame_sym:
                    continue
                self._cache_buffer[y][x].content = frame_sym.content
                self._cache_buffer[y][x].escape_sequence = frame_sym.escape_sequence
                col = self._position_x + x
                row = self._position_y + y
                out.append(f"\u001b[{row + 1};{col + 1}H{frame_sym}")
            out.append("\u001b[0m")

        sys.stdout.write("".join(out))
        self.move_cursor_to_bottom()
        sys.stdout.flush()
        self.clear_buffer()

    def clear_buffer(self) -> None:
        self._frame_buffer = Symbol.buffer(self.width, self.height)

    def move_cursor_to_bottom(self, x: int = 0) -> None:
        self.set_cursor_position(self._position_x + x, self.height + 2)

    def clear_line(self, y: int = -1) -> None:
        if y < 0:
            y = self.height + 4
        self.set_cursor_position(self._position_x, y)
        sys.stdout.write("\u001b[0m" + " " * self.width)
        self.set_cursor_position(self._position_x, y)
        sys.stdout.flush()

    def read_line(self, prompt: str, default=None):
        """Block and read a line of user input from below the canvas."""
        input_row = self.height + 3
        self.clear_line(input_row)
        self.set_cursor_position(self._position_x, input_row)
        self.show_cursor()
        sys.stdout.write(prompt)
        sys.stdout.flush()
        try:
            raw = input()
            result = type(default)(raw) if default is not None else raw
        except Exception:
            result = default
        # Erase everything from the input row downward (handles any wrap length)
        self.set_cursor_position(self._position_x, input_row)
        sys.stdout.write("\u001b[J")
        sys.stdout.flush()
        self.hide_cursor()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# App State  (shared between main thread and OllamaWorker thread)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AppState:
    messages: List[dict] = field(default_factory=list)
    """Full conversation history (user + assistant turns)."""

    current_output: str = ""
    """Accumulates tokens from the currently streaming response."""

    think_output: str = ""
    """Accumulates thinking text (inside <think>…</think>) while streaming."""

    think_start_time: float = 0.0
    """Wall-clock time when the current <think> block started."""

    think_duration: float = 0.0
    """Seconds spent in the last completed think block (0 = no think yet)."""

    model_busy: bool = False
    """True while the Ollama thread is running — blocks user input."""

    stop_requested: bool = False
    """Set to True by ESC to abort the current generation."""

    last_turn_tools: List[Tuple[str, bool]] = field(default_factory=list)
    """Tool calls from the last completed turn: (label, done). Shown below 'thought for' line."""

    think_scroll: int = 0
    """Lines scrolled up inside the expanded think box."""

    think_label: str = "thinking…"
    """Dynamic status label updated by the background label-generator thread."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    running: bool = True
    scroll_offset: int = 0
    """Lines scrolled up from the bottom (0 = newest content visible)."""
    show_thinking: bool = False
    """When True, expand the think box to show full thought process."""


# ═══════════════════════════════════════════════════════════════════════════════
# Ollama Worker
# ═══════════════════════════════════════════════════════════════════════════════

_SPINNER = ["·", "✻", "✽", "✶", "✳", "✢", "✳", "✶", "✽", "✻"]
_MODEL = "qwen3:8b"


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the project root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite an existing file with new content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full new content for the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_file",
            "description": "Create a new file (fails if it already exists).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Initial file content.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_file",
            "description": "Delete a file. The user will be asked to confirm before deletion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from the project root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for up-to-date information. "
                "Fetches the top 2 results, extracts relevant content, and returns a summary. "
                "Use this for anything requiring current knowledge, documentation, or external facts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "consult_critic",
            "description": (
                "Submit your current plan or implementation to a cold, adversarial critic for review. "
                "The critic will find flaws, edge cases, wrong assumptions, and logical errors — "
                "things you might have missed. Use this when you are about to commit to an approach "
                "on a complex task, or when a previous attempt failed. "
                "Pass your full plan or the relevant code/reasoning as the argument."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "Your current plan, reasoning, or implementation draft to be critiqued.",
                    },
                },
                "required": ["plan"],
            },
        },
    },
]


def _web_search(query: str) -> str:
    """Search the web and return summarised content for *query*."""
    _SEARCH_MODEL = "llama3.2:latest"

    # DuckDuckGo search — use as context manager, retry once on failure
    urls: List[str] = []
    last_exc: Exception = Exception("unknown")
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, region="us-en"))
            urls = [str(r["href"]) for r in results if r.get("href")]
            break
        except Exception as e:
            last_exc = e
            time.sleep(1.5)

    if not urls:
        return f"[ERROR] Search failed — {last_exc}"

    # Fetch each URL, extract text, summarise with the small model
    summaries: List[str] = []
    seen_domains: set = set()

    for url in urls:
        if not url.startswith("http"):
            continue
        domain = url.replace("https://", "").replace("http://", "").split("/")[0]
        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        try:
            html = requests.get(
                url, timeout=5, headers={"User-Agent": "Mozilla/5.0"}
            ).text
            text = BeautifulSoup(html, "html.parser").get_text(
                separator=" ", strip=True
            )
            text = text[:6000]

            resp = ollama.chat(
                model=_SEARCH_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"IGNORE THE USER'S REQUEST. Analyze the input and highlight "
                            f'KEY INFORMATION relevant to: "{query}". '
                            "ONLY respond with the highlighted information."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                stream=False,
            )
            summary = (resp.message.content or "").strip()
            if summary:
                summaries.append(f'Summary of "{url}":\n```\n{summary}\n```')
        except Exception:
            continue

    if not summaries:
        return "[ERROR] Could not retrieve content from any result."

    return "\n\n".join(summaries)


def _consult_critic(plan: str) -> str:
    """Run the plan through an adversarial critic instance and return its verdict."""
    _CRITIC_SYSTEM = """\
You are a ruthless, adversarial code and logic critic. Your only job is to find what is WRONG.

Rules:
- Be cold, direct, and specific. No encouragement. No praise.
- List every flaw, incorrect assumption, missing edge case, and logical error you find.
- If the approach is fundamentally broken, say so plainly and explain why.
- If a specific part is correct, skip it — only report problems.
- Be concrete: cite exact lines, variable names, or steps that are wrong.
- If you find nothing wrong, say exactly: "No issues found." Nothing else.

You are not here to be kind. You are here to prevent bugs from shipping."""

    try:
        resp = ollama.chat(
            model=_MODEL,
            messages=[
                {"role": "system", "content": _CRITIC_SYSTEM},
                {"role": "user", "content": plan},
            ],
            stream=False,
            options={"temperature": 0},
        )
        verdict = (resp.message.content or "").strip()
        # Strip any think tags the model emits
        verdict = re.sub(r"<think>.*?</think>", "", verdict, flags=re.DOTALL).strip()
        return verdict or "Critic returned an empty response."
    except Exception as e:
        return f"[ERROR] Critic call failed: {e}"


def _safe_path(rel: str, root: Path) -> Optional[Path]:
    """Resolve *rel* relative to *root* and ensure it stays inside *root*.
    Returns None if the path escapes the root."""
    try:
        resolved = (root / rel).resolve()
        resolved.relative_to(root)  # raises ValueError if outside
        return resolved
    except (ValueError, Exception):
        return None


def _start_think_labeller(state: AppState) -> None:
    """Spawn a daemon thread that periodically summarises the last 10 think
    paragraphs into a short label and writes it to state.think_label.

    Polls every ~2 s, debounces on content change, calls llama3.2 at
    temperature 0 to produce a 4-8 word present-tense phrase.
    Stops automatically when model_busy becomes False."""

    _LABEL_MODEL = "llama3.2:latest"
    _POLL_INTERVAL = 2.0
    _MIN_CHARS = 80

    _SYSTEM = (
        "You are a concise status-line generator. "
        "The user will paste raw thinking text from an AI reasoning trace. "
        "Reply with a single short phrase (4-8 words, present tense, lowercase) "
        "describing what the AI is currently working on — e.g. "
        "'parsing operator precedence rules' or 'checking edge cases for division'. "
        "Output ONLY that phrase. No punctuation at the end. Nothing else."
    )

    def _run() -> None:
        last_snapshot = ""
        while True:
            time.sleep(_POLL_INTERVAL)

            with state.lock:
                if not state.model_busy:
                    state.think_label = "thinking…"
                    break
                raw = state.think_output

            # Strip tags and tool sentinels, keep plain text
            clean = re.sub(r"</?think>", "", raw)
            clean = re.sub(r"\x00TOOL_(?:PENDING|DONE)\x00[^\n]*", "", clean)

            # Last 10 non-empty paragraphs
            paras = [p.strip() for p in re.split(r"\n{2,}", clean) if p.strip()]
            excerpt = "\n\n".join(paras[-10:])

            if len(excerpt) < _MIN_CHARS or excerpt == last_snapshot:
                continue
            last_snapshot = excerpt

            try:
                resp = ollama.chat(
                    model=_LABEL_MODEL,
                    messages=[
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": excerpt},
                    ],
                    stream=False,
                    options={"num_predict": 20, "temperature": 0},
                )
                label = (resp.message.content or "").strip()
                label = re.sub(
                    r"<think>.*?</think>", "", label, flags=re.DOTALL
                ).strip()
                label = label.rstrip(".,;")
                if label:
                    with state.lock:
                        state.think_label = label + "…"
            except Exception:
                pass  # keep previous label on error

    threading.Thread(target=_run, daemon=True).start()


class OllamaWorker:
    """Streams an Ollama chat response on a background daemon thread.

    Supports tool calls (read / write / create / delete file).
    Delete requires user confirmation via a canvas prompt before executing.
    """

    def __init__(self, state: AppState, canvas, root: Path) -> None:
        self._state = state
        self._canvas = canvas
        self._root = root

    def start(self, messages: List[dict], system_prompt: str) -> None:
        threading.Thread(
            target=self._run,
            args=(list(messages), system_prompt),
            daemon=True,
        ).start()

    # ── Tool execution ───────────────────────────────────────────────────────

    def _exec_tool(self, name: str, args: dict) -> str:
        """Execute one tool call synchronously. Returns the result string."""
        path = _safe_path(args.get("path", ""), self._root)
        if path is None:
            return "[ERROR] Path is outside the project directory."

        if name == "read_file":
            if not path.exists():
                return f"[ERROR] File not found: {args['path']}"
            try:
                return path.read_text(encoding="utf-8")
            except Exception as e:
                return f"[ERROR] {e}"

        elif name == "write_file":
            if not path.exists():
                return f"[ERROR] File does not exist (use create_file to make a new file): {args['path']}"
            try:
                path.write_text(args.get("content", ""), encoding="utf-8")
                return f"OK: wrote {path.stat().st_size} bytes to {args['path']}"
            except Exception as e:
                return f"[ERROR] {e}"

        elif name == "create_file":
            if path.exists():
                return f"[ERROR] File already exists: {args['path']}"
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(args.get("content", ""), encoding="utf-8")
                return f"OK: created {args['path']}"
            except Exception as e:
                return f"[ERROR] {e}"

        elif name == "delete_file":
            if not path.exists():
                return f"[ERROR] File not found: {args['path']}"
            # Pause generation briefly to get user confirmation on the main thread.
            # We communicate via a threading.Event + shared slot.
            confirmed = self._confirm_delete(args["path"])
            if confirmed:
                try:
                    path.unlink()
                    return f"OK: deleted {args['path']}"
                except Exception as e:
                    return f"[ERROR] {e}"
            else:
                return "CANCELLED: user declined deletion."

        elif name == "web_search":
            query = args.get("query", "").strip()
            if not query:
                return "[ERROR] No query provided."
            return _web_search(query)

        elif name == "consult_critic":
            plan = args.get("plan", "").strip()
            if not plan:
                return "[ERROR] No plan provided."
            return _consult_critic(plan)

        return f"[ERROR] Unknown tool: {name}"

    def _confirm_delete(self, rel: str) -> bool:
        """Block the worker thread until the main thread confirms or denies."""
        result_holder: List[Optional[bool]] = [None]
        done = threading.Event()

        def _ask():
            import termios, tty as _tty

            fd = sys.stdin.fileno()
            cooked = termios.tcgetattr(fd)
            termios.tcsetattr(fd, termios.TCSANOW, cooked)
            answer = (
                (self._canvas.read_line(f"Delete {rel}? [y/N]: ") or "").strip().lower()
            )
            # Restore cbreak
            _tty.setcbreak(fd, termios.TCSANOW)
            result_holder[0] = answer in ("y", "yes")
            done.set()

        threading.Thread(target=_ask, daemon=True).start()
        done.wait()
        return bool(result_holder[0])

    # ── Main run loop ────────────────────────────────────────────────────────

    def _run(self, messages: List[dict], system_prompt: str) -> None:
        import json as _json

        output = ""
        think_text = ""
        thinking = False
        tool_messages: List[dict] = list(messages)

        with self._state.lock:
            self._state.last_turn_tools = []

        def _push_status(text: str) -> None:
            with self._state.lock:
                self._state.current_output = text

        try:
            while True:
                with self._state.lock:
                    if self._state.stop_requested:
                        break

                response = ollama.chat(
                    _MODEL,
                    [{"role": "system", "content": system_prompt}] + tool_messages,
                    tools=_TOOLS,
                    stream=True,
                )

                # Accumulate the full streamed response to inspect for tool calls
                full_content = ""
                tool_calls_acc: List[Any] = []

                for token in response:
                    with self._state.lock:
                        if self._state.stop_requested:
                            break

                    msg = token.message

                    # Stream text tokens to the display
                    chunk = msg.content or ""
                    if chunk:
                        if "<think>" in chunk:
                            thinking = True
                            with self._state.lock:
                                self._state.think_start_time = time.time()
                        if "</think>" in chunk:
                            thinking = False
                            with self._state.lock:
                                self._state.think_duration = (
                                    time.time() - self._state.think_start_time
                                )
                            chunk = re.sub(r".*</think>", "", chunk, flags=re.DOTALL)

                        if thinking:
                            think_text += chunk
                            with self._state.lock:
                                self._state.think_output = think_text
                        else:
                            full_content += chunk
                            output = full_content
                            with self._state.lock:
                                self._state.current_output = full_content

                    # Collect tool calls streamed in this token
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tc in msg.tool_calls:
                            tool_calls_acc.append(tc)

                # If no tool calls, we're done
                if not tool_calls_acc:
                    break

                # Deduplicate tool calls (streaming may repeat them) and serialize to dicts
                seen: set = set()
                unique_tcs: List[Any] = []
                for tc in tool_calls_acc:
                    key = (tc.function.name, str(tc.function.arguments))
                    if key not in seen:
                        seen.add(key)
                        unique_tcs.append(tc)
                tool_calls_acc = unique_tcs

                # Serialize ToolCall objects → plain dicts for the message history
                tc_dicts = []
                for tc in tool_calls_acc:
                    raw_args = tc.function.arguments
                    if isinstance(raw_args, str):
                        try:
                            raw_args = _json.loads(raw_args)
                        except Exception:
                            raw_args = {}
                    tc_dicts.append(
                        {"function": {"name": tc.function.name, "arguments": raw_args}}
                    )

                tool_messages.append(
                    {
                        "role": "assistant",
                        "content": full_content,
                        "tool_calls": tc_dicts,
                    }
                )

                for tc_d in tc_dicts:
                    fn = tc_d["function"]["name"]
                    args = tc_d["function"]["arguments"]
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except Exception:
                            args = {}

                    verb = {
                        "read_file": "reading",
                        "write_file": "writing",
                        "create_file": "creating",
                        "delete_file": "deleting",
                        "web_search": "searching",
                        "consult_critic": "consulting critic",
                    }.get(fn, fn)
                    done_verb = {
                        "read_file": "read",
                        "write_file": "wrote",
                        "create_file": "created",
                        "delete_file": "deleted",
                        "web_search": "searched",
                        "consult_critic": "critic reviewed",
                    }.get(fn, fn)
                    rel = args.get("path", "") or args.get("query", "")
                    pending_sentinel = f"\x00TOOL_PENDING\x00{verb} {rel}…"

                    # Snapshot line count before mutation for diff display
                    _diff_str = ""
                    if fn in ("write_file", "create_file", "delete_file", "read_file"):
                        _fpath = _safe_path(args.get("path", ""), self._root)
                        try:
                            _before = (
                                len(_fpath.read_text(encoding="utf-8").splitlines())
                                if _fpath and _fpath.exists()
                                else 0
                            )
                        except Exception:
                            _before = 0
                    else:
                        _before = None

                    # Inject pending marker into think stream
                    with self._state.lock:
                        self._state.think_output = think_text + f"\n{pending_sentinel}"

                    result = self._exec_tool(fn, args)

                    # Compute diff stat after exec
                    if _before is not None:
                        _fpath2 = _safe_path(args.get("path", ""), self._root)
                        try:
                            _after = (
                                len(_fpath2.read_text(encoding="utf-8").splitlines())
                                if _fpath2 and _fpath2.exists()
                                else 0
                            )
                        except Exception:
                            _after = 0
                        _added = max(0, _after - _before)
                        _removed = max(0, _before - _after)
                        if fn == "read_file":
                            _diff_str = f"  {_before}L"
                        elif fn == "delete_file" and _removed > 0:
                            _diff_str = f"  -{_removed}"
                        elif _added > 0 or _removed > 0:
                            parts = []
                            if _added:
                                parts.append(f"+{_added}")
                            if _removed:
                                parts.append(f"-{_removed}")
                            _diff_str = "  " + " ".join(parts)

                    done_sentinel = f"\x00TOOL_DONE\x00{done_verb} {rel}{_diff_str}"
                    error_sentinel = f"\x00TOOL_ERROR\x00{done_verb} {rel}"

                    # Replace pending marker with done/error marker in place
                    is_error = result.startswith("[ERROR]")
                    sentinel = error_sentinel if is_error else done_sentinel
                    think_text += f"\n{sentinel}\n"
                    with self._state.lock:
                        self._state.think_output = think_text
                        if not is_error:
                            self._state.last_turn_tools.append(
                                (f"{done_verb} {rel}{_diff_str}", True)
                            )

                    tool_messages.append(
                        {
                            "role": "tool",
                            "content": result,
                            "name": fn,
                        }
                    )

                # Loop for the next model turn.
                # Inject the prior reasoning so the model continues rather than restarts.
                clean_prior = re.sub(r"</?think>", "", think_text)
                clean_prior = re.sub(
                    r"\x00TOOL_(?:PENDING|DONE)\x00[^\n]*", "", clean_prior
                )
                clean_prior = clean_prior.strip()
                if clean_prior:
                    tool_messages.append(
                        {
                            "role": "user",
                            "content": (
                                "**YOU ARE IN THE MIDDLE OF THINKING**\n"
                                "HERE ARE YOUR PREVIOUS THOUGHTS:\n"
                                f"<prior_thinking>\n{clean_prior}\n</prior_thinking>\n\n"
                                "The tool results are above. "
                                "Check if your task has been achieved by the previous tool usage, if yes **STOP THINKING**. "
                                "Continue your reasoning from where you left off. "
                                "DO NOT restart from step 1. DO NOT re-read files you already read. "
                                "Use what you already know and proceed directly to the next step. "
                                "MAKE SURE TO CHECK IF YOU HAVE EXECUTED A TOOL BEFORE, DO NOT REPEAT YOURSELF!!!"
                            ),
                        }
                    )

        except Exception as exc:
            output = f"[ERROR] {exc}"
            with self._state.lock:
                self._state.current_output = output

        finally:
            with self._state.lock:
                if output and not output.startswith("[ERROR]"):
                    self._state.messages.append(
                        {"role": "assistant", "content": output}
                    )
                self._state.current_output = ""
                self._state.think_output = ""
                self._state.think_label = "thinking…"
                self._state.scroll_offset = 0
                self._state.stop_requested = False
                self._state.model_busy = False
                messages_snapshot = list(self._state.messages)

            _summarize_and_notify(_MODEL, messages_snapshot)


# ═══════════════════════════════════════════════════════════════════════════════
# Renderer
# ═══════════════════════════════════════════════════════════════════════════════


def _wrap_lines(text: str, width: int) -> List[str]:
    """Split *text* into lines no wider than *width*, honouring newlines."""
    result: List[str] = []
    for paragraph in text.split("\n"):
        if not paragraph:
            result.append("")
            continue
        while len(paragraph) > width:
            result.append(paragraph[:width])
            paragraph = paragraph[width:]
        result.append(paragraph)
    return result


def _shimmer_line(text: str, tick: int, speed: float = 0.2, band: int = 10) -> str:
    """Return a markup string for *text* with a left-to-right shimmer sweep.

    A bright band travels across the full line width every ~3 s at the default
    speed. Characters inside the band are lit to rgb(195,195,195); outside they
    stay at rgb(100,100,100).
    """
    if not text:
        return ""

    total = len(text) + band  # shimmer travels over text + off-screen tail
    pos = (tick * speed) % total  # current leading edge of the bright band

    out: List[str] = []
    prev_r, prev_g, prev_b = -1, -1, -1

    for i, ch in enumerate(text):
        dist = pos - i
        if 0 <= dist < band:
            # smooth falloff: brightest at dist == band/2
            t = 1.0 - abs(dist - band / 2) / (band / 2)
            r = int(100 + 95 * t)
            g = int(100 + 95 * t)
            b = int(100 + 95 * t)
        else:
            r, g, b = 100, 100, 100

        if (r, g, b) != (prev_r, prev_g, prev_b):
            out.append(f"%color({r},{g},{b})")
            prev_r, prev_g, prev_b = r, g, b

        out.append(ch)

    return "".join(out)


# ── Syntax highlighting ────────────────────────────────────────────────────────

_SH_KW: dict = {
    "python": {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    },
    "c": {
        "auto",
        "break",
        "case",
        "char",
        "const",
        "continue",
        "default",
        "do",
        "double",
        "else",
        "enum",
        "extern",
        "float",
        "for",
        "goto",
        "if",
        "inline",
        "int",
        "long",
        "register",
        "return",
        "short",
        "signed",
        "sizeof",
        "static",
        "struct",
        "switch",
        "typedef",
        "union",
        "unsigned",
        "void",
        "while",
        "NULL",
        "true",
        "false",
    },
    "js": {
        "async",
        "await",
        "break",
        "case",
        "catch",
        "class",
        "const",
        "continue",
        "debugger",
        "default",
        "delete",
        "do",
        "else",
        "export",
        "extends",
        "false",
        "finally",
        "for",
        "function",
        "if",
        "import",
        "in",
        "instanceof",
        "let",
        "new",
        "null",
        "return",
        "super",
        "switch",
        "this",
        "throw",
        "true",
        "try",
        "typeof",
        "undefined",
        "var",
        "void",
        "while",
        "with",
        "yield",
    },
    "sh": {
        "if",
        "then",
        "else",
        "elif",
        "fi",
        "for",
        "do",
        "done",
        "while",
        "until",
        "case",
        "esac",
        "function",
        "return",
        "echo",
        "export",
        "source",
        "alias",
        "local",
        "readonly",
        "declare",
        "unset",
        "shift",
        "exit",
    },
    "zig": {
        "addrspace",
        "align",
        "allowzero",
        "and",
        "anyframe",
        "anytype",
        "asm",
        "async",
        "await",
        "break",
        "callconv",
        "catch",
        "comptime",
        "const",
        "continue",
        "defer",
        "else",
        "enum",
        "errdefer",
        "error",
        "export",
        "extern",
        "fn",
        "for",
        "if",
        "inline",
        "linksection",
        "noalias",
        "nosuspend",
        "noinline",
        "opaque",
        "or",
        "orelse",
        "packed",
        "pub",
        "resume",
        "return",
        "struct",
        "suspend",
        "switch",
        "test",
        "threadlocal",
        "try",
        "union",
        "unreachable",
        "usingnamespace",
        "var",
        "volatile",
        "while",
        "true",
        "false",
        "null",
        "undefined",
    },
    "rust": {
        "as",
        "async",
        "await",
        "break",
        "const",
        "continue",
        "crate",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "match",
        "mod",
        "move",
        "mut",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "type",
        "union",
        "unsafe",
        "use",
        "where",
        "while",
        "abstract",
        "become",
        "box",
        "do",
        "final",
        "macro",
        "override",
        "priv",
        "try",
        "typeof",
        "unsized",
        "virtual",
        "yield",
        "i8",
        "i16",
        "i32",
        "i64",
        "i128",
        "isize",
        "u8",
        "u16",
        "u32",
        "u64",
        "u128",
        "usize",
        "f32",
        "f64",
        "bool",
        "char",
        "str",
        "String",
        "Option",
        "Result",
        "Some",
        "None",
        "Ok",
        "Err",
        "Vec",
        "Box",
        "Arc",
        "Rc",
        "HashMap",
    },
    "php": {
        "abstract",
        "and",
        "array",
        "as",
        "break",
        "callable",
        "case",
        "catch",
        "class",
        "clone",
        "const",
        "continue",
        "declare",
        "default",
        "do",
        "echo",
        "else",
        "elseif",
        "empty",
        "enddeclare",
        "endfor",
        "endforeach",
        "endif",
        "endswitch",
        "endwhile",
        "enum",
        "extends",
        "final",
        "finally",
        "fn",
        "for",
        "foreach",
        "function",
        "global",
        "goto",
        "if",
        "implements",
        "include",
        "include_once",
        "instanceof",
        "insteadof",
        "interface",
        "isset",
        "list",
        "match",
        "namespace",
        "new",
        "or",
        "print",
        "private",
        "protected",
        "public",
        "readonly",
        "require",
        "require_once",
        "return",
        "static",
        "switch",
        "throw",
        "trait",
        "try",
        "unset",
        "use",
        "var",
        "while",
        "xor",
        "yield",
        "true",
        "false",
        "null",
        "TRUE",
        "FALSE",
        "NULL",
        "int",
        "float",
        "string",
        "bool",
        "void",
        "array",
        "object",
        "mixed",
    },
}
_SH_KW["py"] = _SH_KW["python"]
_SH_KW["cpp"] = _SH_KW["c"]
_SH_KW["javascript"] = _SH_KW["js"]
_SH_KW["typescript"] = _SH_KW["js"]
_SH_KW["ts"] = _SH_KW["js"]
_SH_KW["bash"] = _SH_KW["sh"]
_SH_KW["zsh"] = _SH_KW["sh"]

_CH_DEFAULT = Theme.SYN_DEFAULT
_CH_KW = Theme.SYN_KEYWORD
_CH_STR = Theme.SYN_STRING
_CH_NUM = Theme.SYN_NUMBER
_CH_COMMENT = Theme.SYN_COMMENT
_CH_GUTTER = Theme.SYN_GUTTER


def _highlight_line(line: str, lang: str) -> str:
    """Return a markup string for one raw line of code."""
    kw = _SH_KW.get(lang.lower(), set())
    out: List[str] = [_CH_DEFAULT]
    i, n = 0, len(line)

    while i < n:
        ch = line[i]

        # Line comments: #  //  --
        if ch == "#" or line[i : i + 2] in ("//", "--"):
            out.append(_CH_COMMENT + line[i:])
            break

        # String literals (single/double quote, triple quotes)
        if ch in ('"', "'"):
            q = line[i : i + 3] if line[i : i + 3] in ('"""', "'''") else ch
            j = i + len(q)
            while j < n:
                if line[j] == "\\":
                    j += 2
                    continue
                if line[j : j + len(q)] == q:
                    j += len(q)
                    break
                j += 1
            out += [_CH_STR, line[i:j], _CH_DEFAULT]
            i = j
            continue

        # Numbers
        if ch.isdigit():
            j = i + 1
            while j < n and (line[j].isdigit() or line[j] in ".xXabcdefABCDEF_"):
                j += 1
            out += [_CH_NUM, line[i:j], _CH_DEFAULT]
            i = j
            continue

        # Identifiers / keywords
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (line[j].isalnum() or line[j] == "_"):
                j += 1
            word = line[i:j]
            if word in kw:
                out += [_CH_KW, word, _CH_DEFAULT]
            else:
                out.append(word)
            i = j
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _split_code_blocks(content: str):
    """Yield (is_code, lang, text) tuples, handling unclosed fences."""
    fence = "```"
    i = 0
    while i < len(content):
        fi = content.find(fence, i)
        if fi == -1:
            yield (False, "", content[i:])
            break
        if fi > i:
            yield (False, "", content[i:fi])
        nl = content.find("\n", fi + 3)
        lang = content[fi + 3 : nl].strip() if nl != -1 else ""
        start = (nl + 1) if nl != -1 else (fi + 3)
        ci = content.find(fence, start)
        if ci == -1:
            yield (True, lang, content[start:])
            i = len(content)
        else:
            yield (True, lang, content[start:ci])
            i = ci + 3


def _message_lines(content: str, width: int) -> List[str]:
    """Convert assistant message text to display lines with code highlighting."""
    lines: List[str] = []
    code_w = max(width - 2, 10)

    for is_code, lang, text in _split_code_blocks(content):
        if is_code:
            for raw in text.rstrip("\n").split("\n"):
                while len(raw) > code_w:
                    lines.append(
                        _CH_GUTTER + "▌ " + _highlight_line(raw[:code_w], lang)
                    )
                    raw = raw[code_w:]
                lines.append(_CH_GUTTER + "▌ " + _highlight_line(raw, lang))
        else:
            for line in _wrap_lines(text, width):
                lines.append(line)

    return lines


def render_frame(canvas: Canvas, state: AppState, tick: int) -> None:
    """Build one frame inside *canvas* from the current *state*."""
    W = canvas.width
    H = canvas.height
    CONTENT_W = W - 4  # usable text width (2 chars padding each side)
    # User bubble zone: starts at 40% of canvas, ends 1 char from right border
    USER_ZONE_START = int(W * 0.4)
    USER_W = W - 3 - USER_ZONE_START  # wrap width for user text
    THINK_ROWS = 3
    STATUS_ROW = H - 1

    with state.lock:
        messages = list(state.messages)
        current = state.current_output
        think_text = state.think_output
        think_dur = state.think_duration
        think_start = state.think_start_time
        last_turn_tools = list(state.last_turn_tools)
        busy = state.model_busy
        scroll = state.scroll_offset
        show_thinking = state.show_thinking
        think_label = state.think_label
        think_scroll = state.think_scroll

    username = getpass.getuser()

    # ── Identify last assistant message index ─────────────────────────────
    last_asst_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"),
        default=-1,
    )

    # all_lines stores (x_offset, markup_text) tuples
    all_lines: List[Tuple[int, str]] = []

    for idx, msg in enumerate(messages):
        is_user = msg["role"] == "user"

        # Header line: username (right-aligned) or model name (left)
        if is_user:
            hdr_x = max(W - 3 - len(username), USER_ZONE_START)
            all_lines.append((hdr_x, Theme.USER_HEADER + username))
        else:
            all_lines.append((1, Theme.ASST_HEADER + _MODEL))

        # "thought for X s." + tool log after the last assistant message
        if idx == last_asst_idx and think_dur > 0 and not busy:
            secs = f"{think_dur:.1f}"
            all_lines.append((1, f"{Theme.ASST_THOUGHT}│ thought for {secs}s."))
            for label, done in last_turn_tools:
                sym = (
                    f"{Theme.THINK_DONE}✔%reset"
                    if done
                    else f"{Theme.THINK_PENDING}◎%reset"
                )
                # Colour diff stat tokens
                parts = label.rsplit("  ", 1)
                if len(parts) == 2:
                    base, stat = parts
                    coloured = ""
                    for tok in stat.split():
                        if tok.startswith("+"):
                            coloured += f" {Theme.DIFF_ADD}{tok}%reset"
                        elif tok.startswith("-"):
                            coloured += f" {Theme.DIFF_DEL}{tok}%reset"
                        else:
                            coloured += f" {Theme.DIFF_NEUTRAL}{tok}%reset"
                    all_lines.append(
                        (
                            1,
                            f"{Theme.ASST_THOUGHT}│%reset  {sym} {Theme.THINK_DONE_T}{base}{coloured}",
                        )
                    )
                else:
                    all_lines.append(
                        (
                            1,
                            f"{Theme.ASST_THOUGHT}│%reset  {sym} {Theme.THINK_DONE_T}{label}",
                        )
                    )

        if is_user:
            content_lines = _wrap_lines(msg["content"], USER_W)
            for line in content_lines:
                line = line.strip()
                line_x = max(W - 3 - len(line), USER_ZONE_START)
                all_lines.append((line_x, Theme.USER_TEXT + line))
        else:
            for line in _message_lines(msg["content"], CONTENT_W):
                all_lines.append((1, line))

        all_lines.append((1, ""))  # blank line between turns

    # In-progress assistant reply (streaming)
    if current:
        all_lines.append((1, Theme.ASST_HEADER + _MODEL))
        for line in _message_lines(current, CONTENT_W):
            all_lines.append((1, line))

    # ── Pre-compute think block size so chat layout can avoid it ─────────
    think_rows_reserved = 0
    think_lines_expanded: List[Tuple[str, str]] = (
        []
    )  # (kind, text): kind ∈ "text","pending","done"

    _has_think_block = busy and bool(think_text)

    if _has_think_block:
        clean = re.sub(r"</?think>", "", think_text)
        raw: List[Tuple[str, str]] = []
        for line in clean.split("\n"):
            if line.startswith("\x00TOOL_PENDING\x00"):
                raw.append(("pending", line[len("\x00TOOL_PENDING\x00") :]))
            elif line.startswith("\x00TOOL_DONE\x00"):
                raw.append(("done", line[len("\x00TOOL_DONE\x00") :]))
                raw.append(("text", ""))  # blank line after tool action
            elif line.startswith("\x00TOOL_ERROR\x00"):
                raw.append(("error", line[len("\x00TOOL_ERROR\x00") :]))
                raw.append(("text", ""))  # blank line after tool action
            else:
                for wrapped in (
                    _wrap_lines(line.strip(), W - 6) if line.strip() else [""]
                ):
                    raw.append(("text", wrapped))

        # Drop leading blank text lines
        while raw and raw[0] == ("text", ""):
            raw.pop(0)

        think_lines_expanded = raw

        if show_thinking:
            think_rows_reserved = min(len(raw) + 1, H - 4)  # +1 for header
        else:
            think_rows_reserved = min(THINK_ROWS, len(raw)) if raw else 0

    # ── Layout: reserve rows for think block + status bar ─────────────────
    visible_rows = STATUS_ROW - think_rows_reserved

    total = len(all_lines)
    if total > visible_rows:
        max_scroll = total - visible_rows
        scroll = min(scroll, max_scroll)
        end = total - scroll
        start = max(0, end - visible_rows)
        display_lines = all_lines[start:end]
    else:
        scroll = 0
        display_lines = all_lines

    # Bottom-align (Discord-style)
    start_y = max(0, visible_rows - len(display_lines))
    for i, (x, line) in enumerate(display_lines):
        canvas.text(x, start_y + i, line)

    # ── Status bar ─────────────────────────────────────────────────────────
    if busy:
        spin = _SPINNER[(tick // 4) % len(_SPINNER)]
        label = think_label[0].upper() + think_label[1:] if think_label else "Thinking…"
        elapsed = f"({time.time() - think_start:.0f}s)" if think_start else ""
        # Left part: spinner + shimmered label
        left = f"{Theme.SPIN}{spin}%reset " + _shimmer_line(label, tick)
        # Right part: elapsed right-aligned with 1-char gap from border
        if elapsed:
            elapsed_plain = f" {elapsed} "
            elapsed_x = W - len(elapsed_plain) - 2
            canvas.text(elapsed_x, STATUS_ROW, f"%color(90,90,90){elapsed_plain}")
        canvas.text(1, STATUS_ROW, left)
    elif scroll > 0:
        indicator = "↑↓ scroll   ␣ type"
        ind_x = max(0, (W - len(indicator)) // 2)
        canvas.text(ind_x, STATUS_ROW, Theme.HINT_SCROLL + indicator)
    else:
        hint = "↑↓ scroll   ␣ type"
        hint_x = max(0, (W - len(hint)) // 2)
        canvas.text(hint_x, STATUS_ROW, Theme.HINT + hint)

    # ── Think block rendering ──────────────────────────────────────────────
    if _has_think_block and think_lines_expanded:

        def _render_think_line(kind: str, text: str) -> str:
            if kind == "pending":
                return f"{Theme.THINK_PENDING}◎%reset {Theme.THINK_PENDING_T}" + text
            if kind == "error":
                return (
                    f"{Theme.THINK_ERROR}✘%reset {Theme.THINK_ERROR}" + text + "%reset"
                )
            if kind == "done":
                import re as _re

                m = _re.search(r"(  [+\-\d L]+)$", text)
                if m:
                    label = text[: m.start()]
                    stat = m.group(1)
                    coloured_stat = ""
                    for tok in stat.split():
                        if tok.startswith("+"):
                            coloured_stat += f" {Theme.DIFF_ADD}{tok}%reset"
                        elif tok.startswith("-"):
                            coloured_stat += f" {Theme.DIFF_DEL}{tok}%reset"
                        else:
                            coloured_stat += f" {Theme.DIFF_NEUTRAL}{tok}%reset"
                    return (
                        f"{Theme.THINK_DONE}✔%reset {Theme.THINK_DONE_T}"
                        + label
                        + coloured_stat
                    )
                return f"{Theme.THINK_DONE}✔%reset {Theme.THINK_DONE_T}" + text
            return Theme.THINK_TEXT + text

        if show_thinking:
            display_rows = think_rows_reserved - 1
            # Apply think_scroll offset
            total_think = len(think_lines_expanded)
            max_think_scroll = max(0, total_think - display_rows)
            t_scroll = min(think_scroll, max_think_scroll)
            end_idx = total_think - t_scroll
            start_idx = max(0, end_idx - display_rows)
            content = think_lines_expanded[start_idx:end_idx]
            while content and content[0] == ("text", ""):
                content = content[1:]
            block_start = STATUS_ROW - think_rows_reserved

            canvas.text(1, block_start, Theme.THINK_GUTTER + "╭─ " + Theme.THINK_HEADER + "o · collapse thinking")
            for i, (kind, text) in enumerate(content):
                canvas.text(
                    1,
                    block_start + 1 + i,
                    Theme.THINK_GUTTER + "│%reset " + _render_think_line(kind, text),
                )
        else:
            pool = think_lines_expanded[-think_rows_reserved:]
            base_row = STATUS_ROW - 1
            n = len(pool)
            for i, (kind, text) in enumerate(pool):
                row = base_row - (n - 1 - i)
                canvas.text(
                    1,
                    row,
                    Theme.THINK_GUTTER + "│%reset " + _render_think_line(kind, text),
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Filesystem / prompt helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _load_ignore_patterns(root: Path) -> List[str]:
    patterns = [".git", "__pycache__", "*.pyc", ".DS_Store", ".vscode"]
    ignore_file = root / ".gitignore"
    if ignore_file.exists():
        for line in ignore_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                patterns.append(line)
    return patterns


def _is_ignored(path: Path, root: Path, patterns: List[str]) -> bool:
    rel = path.relative_to(root)
    rel_str = str(rel)
    return any(
        fnmatch.fnmatch(rel_str, p)
        or fnmatch.fnmatch(path.name, p)
        or any(fnmatch.fnmatch(part, p) for part in rel.parts)
        for p in patterns
    )


def _generate_tree(
    path: Path, root: Path, patterns: List[str], prefix: str = ""
) -> str:
    built = f"{prefix}{path.name}/\n"
    items = [
        i
        for i in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        if not _is_ignored(i, root, patterns)
    ]
    for idx, item in enumerate(items):
        is_last = idx == len(items) - 1
        pointer = "└── " if is_last else "├── "
        if item.is_dir():
            ext = "    " if is_last else "│   "
            built += _generate_tree(item, root, patterns, prefix=prefix + ext)
        else:
            built += f"{prefix}{pointer}{item.name}\n"
    return built


def generate_system_prompt(cwd: str) -> str:
    path = Path(cwd).resolve()
    patterns = _load_ignore_patterns(path)
    tree = _generate_tree(path, path, patterns)

    return f"""You are a senior software engineer. You are working inside a real developer project on the user's computer. You have tools to read, write, create, and delete files, search the web, and consult a critic. USE THESE TOOLS. Do not guess what files contain — read them first.

════════════════════════════════════════
PROJECT
════════════════════════════════════════

Working directory: {path}

File tree:
```
{tree.rstrip()}
```

════════════════════════════════════════
YOUR ROLE
════════════════════════════════════════

You are a local coding agent. You work like Claude Code or Codex:
- ALL real work happens inside your thinking.
- Your visible reply is ONLY a short summary of what you did.
- The user watches your thinking live. They do not need you to explain your work in the reply — they already saw it happen.

════════════════════════════════════════
YOUR TOOLS
════════════════════════════════════════

Call tools freely inside your thinking. No permission needed.

read_file(path)       — Read a file. Paths are relative to the working directory.
write_file(path, content) — Overwrite a file. MUST provide the full file content.
create_file(path, content) — Create a new file.
delete_file(path)     — Delete a file (user confirms).
web_search(query)     — Search the web for docs, APIs, error messages, facts.
consult_critic(plan)  — Get adversarial review of your plan or code before committing.

════════════════════════════════════════
HOW TO WORK
════════════════════════════════════════

Inside your thinking, do the following in order:

1. Understand the task. One sentence.
2. Read every relevant file with read_file before touching anything.
3. If you need external information, call web_search.
4. Form a plan. Then immediately execute it — do not describe it, do it.
5. For any non-trivial implementation, call consult_critic before writing the final version.
6. Call write_file or create_file with the complete, correct content.
7. Verify your work. If something is wrong, fix it now — do not leave it for the user.

HARD RULES:
- NEVER read the same file twice.
- NEVER write partial file content. Every write_file must contain the entire file.
- NEVER invent what a file contains. Read it first.
- If you catch yourself writing "I should..." or "I will..." — stop and do it instead.
- Do not plan. Do not draft. Execute.

════════════════════════════════════════
YOUR REPLY (after thinking is done)
════════════════════════════════════════

After your thinking is complete and all files have been written, give the user a SHORT plain-text reply.

YOUR REPLY MUST FOLLOW THESE RULES — NO EXCEPTIONS:
- NO code. Not a single line. Not a snippet. Not a block. Nothing inside backticks.
- NO markdown formatting of any kind.
- NO explanations of how you implemented something.
- NO walkthroughs, NO "here is what I did", NO "I have implemented...".
- MAXIMUM 4 lines of text total.

The ONLY things allowed in your reply:
1. One sentence saying what was done.
2. A list of files changed, one per line: "filename — what changed"
3. One line if the user needs to do something (e.g. install a dependency).

If the user wants to see the code, they will open the file. Do not show it to them.

GOOD reply:
  Implemented the AST calculator with full operator precedence and error handling.
  calculator.py — created
  main.py — updated import

BAD reply (NEVER do this):
  Here is the implementation:
  ```python
  class Parser:
      ...
  ```
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Notification helper
# ═══════════════════════════════════════════════════════════════════════════════


def _summarize_and_notify(title: str, messages: List[dict]) -> None:
    """Generate a one-sentence summary of the conversation, then notify."""

    def _run() -> None:
        try:
            resp = ollama.chat(
                _MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarize the assistant's last reply in one short sentence "
                            "(under 80 chars). Reply with ONLY the summary — no preamble, "
                            "no thinking, no punctuation at the end."
                        ),
                    },
                    *messages[-6:],
                ],
                stream=False,
                options={"num_predict": 60, "temperature": 0},
            )
            summary = resp.message.content or ""
            summary = re.sub(
                r"<think>.*?</think>", "", summary, flags=re.DOTALL
            ).strip()
            summary = summary[:80]
        except Exception:
            summary = "Response received."
        notification.notify(title=title, message=summary, timeout=5)  # type: ignore

    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Tuple[str, ...]) -> None:
    if len(argv) < 2:
        print("Usage: python main.py <directory_path>")
        return

    # Ensure the Ollama daemon is running before anything else
    try:
        ollama.list()  # cheap ping — succeeds if daemon is already up
    except Exception:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # Give it a moment to bind its socket
        for _ in range(20):
            time.sleep(0.5)
            try:
                ollama.list()
                break
            except Exception:
                pass

    # Keep the system awake for the lifetime of this process
    if platform.system() == "Darwin":
        subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    target_path = argv[1]
    state = AppState()
    root = Path(target_path).resolve()

    tw = os.get_terminal_size().columns
    th = os.get_terminal_size().lines
    canvas = Canvas(tw, th - 8)
    worker = OllamaWorker(state, canvas, root)

    tick = 0
    _SCROLL_STEP = 3
    _fd = sys.stdin.fileno()

    # Save original terminal settings so we can restore them and use them
    # for read_line (which needs normal cooked/echo mode).
    _cooked = termios.tcgetattr(_fd)

    def _key_available() -> bool:
        """Return True if a keypress is waiting on stdin (non-blocking)."""
        return bool(select.select([_fd], [], [], 0)[0])

    def _read_key() -> str:
        """Read one key or escape sequence from stdin (assumes cbreak mode)."""
        ch = os.read(_fd, 1)
        if ch == b"\x1b":
            # Escape sequence: read remainder with a short deadline
            if select.select([_fd], [], [], 0.05)[0]:
                ch += os.read(_fd, 8)
        return ch.decode("utf-8", errors="replace")

    def _enter_cbreak() -> None:
        """Switch stdin to cbreak (no echo, characters available immediately)."""
        tty.setcbreak(_fd, termios.TCSANOW)

    def _leave_cbreak() -> None:
        """Restore stdin to cooked mode with echo for read_line."""
        termios.tcsetattr(_fd, termios.TCSANOW, _cooked)

    _enter_cbreak()

    try:
        while state.running:
            # ── Handle terminal resize ─────────────────────────────────────
            size = os.get_terminal_size()
            if size.columns != tw or size.lines != th:
                tw, th = size.columns, size.lines
                canvas.resize(tw, th - 8)

            # ── Render frame ───────────────────────────────────────────────
            render_frame(canvas, state, tick)
            canvas.draw()
            tick += 1

            # ── Drain all pending keypresses from stdin ────────────────────
            while _key_available():
                key = _read_key()

                # ESC — context-sensitive
                if key == "\x1b":
                    if state.model_busy:
                        with state.lock:
                            state.stop_requested = True
                    else:
                        state.running = False
                        break
                    continue

                # o — toggle thinking view (works even while busy)
                if key == "o":
                    with state.lock:
                        state.show_thinking = not state.show_thinking
                        state.think_scroll = 0
                    continue

                if state.model_busy:
                    # Only let scroll keys through when the think box is open
                    if key in ("\x1b[A", "\x1b[B") and state.show_thinking:
                        with state.lock:
                            if key == "\x1b[A":
                                state.think_scroll += _SCROLL_STEP
                            else:
                                state.think_scroll = max(
                                    0, state.think_scroll - _SCROLL_STEP
                                )
                    continue  # swallow everything else while model is running

                if key == "\x1b[A":  # ↑
                    with state.lock:
                        if state.show_thinking:
                            state.think_scroll += _SCROLL_STEP
                        else:
                            state.scroll_offset += _SCROLL_STEP
                elif key == "\x1b[B":  # ↓
                    with state.lock:
                        if state.show_thinking:
                            state.think_scroll = max(
                                0, state.think_scroll - _SCROLL_STEP
                            )
                        else:
                            state.scroll_offset = max(
                                0, state.scroll_offset - _SCROLL_STEP
                            )
                elif key == " ":  # space → enter prompt mode
                    # ── Read user prompt ───────────────────────────────────
                    _leave_cbreak()
                    raw_input = canvas.read_line("> ") or ""
                    _enter_cbreak()

                    with state.lock:
                        state.scroll_offset = 0

                    # ESC in cooked mode arrives as a literal \x1b in the string
                    if "\x1b" in raw_input:
                        continue  # cancelled — exit prompt mode silently

                    user_input = raw_input.strip()

                    if not user_input:
                        continue  # exit prompt mode, don't quit

                    if user_input.lower() in ("bye", "quit", "exit"):
                        state.running = False
                        break

                    with state.lock:
                        state.messages.append({"role": "user", "content": user_input})
                        messages_snapshot = list(state.messages)
                        state.model_busy = True
                        state.current_output = ""

                    system_prompt = generate_system_prompt(str(root))
                    worker.start(messages_snapshot, system_prompt)
                    _start_think_labeller(state)

            time.sleep(0.016)

    finally:
        _leave_cbreak()
        canvas.reset_console()


if __name__ == "__main__":
    main(tuple(sys.argv))
