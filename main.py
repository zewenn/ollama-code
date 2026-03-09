"""
ollama-code  —  local AI coding agent, single-file.

Usage:
    python main.py <directory_path>

Layout:
  § 1  Theme                     — colour palette
  § 2  Symbol / TextBox / Canvas — terminal drawing primitives
  § 3  AppState / Application    — shared state + persistent history
  § 4  Tool schemas              — JSON tool definitions
  § 5  Tool implementations      — file I/O, shell, web, critic
  § 6  Think labeller            — background label-generator thread
  § 7  AgentWorker               — 8-step structured pipeline
  § 8  Renderer                  — render_frame
  § 9  Filesystem / prompt       — tree builder + system prompt
  § 10 Notification              — post-turn desktop notify
  § 11 main()                    — event loop

Agent pipeline (every turn):
  1. User enters a prompt
  2. Build system prompt with current file layout
  3. Ask model for step-by-step plan          (temperature=0.5)
  4. Summarise user prompt to one sentence    (temperature=0)
  5. For each step:
       5.1  Collect context: request + past tool calls + thinking summaries
       5.2  Think + call tools until step is resolved
       5.3  Ask model whether step goal was achieved; retry if not
       5.4  Log tool usage; store thinking summary
       5.5  Advance to next step
  6. Ask model whether overall goal achieved; replan + restart if not
  7. Ask model for human-readable output summary
  8. Persist request / tool usage / summary to Application.history
"""

from __future__ import annotations

import fnmatch
import getpass
import json
import os
import platform
import re
import select
import subprocess
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ollama
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from plyer import notification


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  Theme
# ═══════════════════════════════════════════════════════════════════════════════


class Theme:
    """Central colour palette.  Change values here to restyle the entire UI."""

    # ── Chat ──────────────────────────────────────────────────────────────────
    USER_TEXT = "%color(120,180,255)"
    USER_HEADER = "%color(70,70,70)"
    ASST_THOUGHT = "%color(80,80,80)"
    ASST_HEADER = "%color(70,70,70)"

    THINK_TEXT = "%color(100,100,100)"
    THINK_GUTTER = "%color(80,80,80)"
    THINK_HEADER = "%color(55,55,75)"
    THINK_PENDING = "%color(160,140,60)"
    THINK_PENDING_T = "%color(150,140,100)"
    THINK_DONE = "%color(70,160,100)"
    THINK_DONE_T = "%color(100,120,100)"
    THINK_ERROR = "%color(210,70,70)"

    # ── Status bar ────────────────────────────────────────────────────────────
    SPIN = "%color(210,120,40)"
    HINT = "%color(65,65,65)"
    HINT_SCROLL = "%color(70,70,90)"

    # ── Diff stats ───────────────────────────────────────────────────────────
    DIFF_ADD = "%color(80,180,100)"
    DIFF_DEL = "%color(200,80,80)"
    DIFF_NEUTRAL = "%color(90,90,90)"

    # ── Syntax highlighting ───────────────────────────────────────────────────
    SYN_DEFAULT = "%reset%color(185,185,185)"
    SYN_KEYWORD = "%reset%color(140,120,210)"
    SYN_STRING = "%reset%color(180,160,80)"
    SYN_NUMBER = "%reset%color(210,130,80)"
    SYN_COMMENT = "%reset%color(105,120,85)"
    SYN_GUTTER = "%color(55,55,70)"

    # ── Canvas border ─────────────────────────────────────────────────────────
    BORDER = "\u001b[38;2;55;55;65m"
    BORDER_RESET = "\u001b[0m"


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  Drawing primitives  (Symbol / TextBox / Canvas)  — UNCHANGED
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

    def _get_terminal_size(self) -> Tuple[int, int]:
        size = os.get_terminal_size()
        return size.columns, size.lines

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

    def reset_console(self) -> None:
        self._cache_buffer = Symbol.null_buffer(self.width, self.height)
        self._frame_buffer = Symbol.null_buffer(self.width, self.height)
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
                if s.content not in ("\0", "\t"):
                    self._frame_buffer[y][x] = s

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
        self.set_cursor_position(self._position_x, input_row)
        sys.stdout.write("\u001b[J")
        sys.stdout.flush()
        self.hide_cursor()
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  Shared state  (AppState / Application / ToolEvent / TurnRecord)
# ═══════════════════════════════════════════════════════════════════════════════


class ToolStatus(Enum):
    PENDING = auto()
    DONE = auto()
    ERROR = auto()


@dataclass
class ToolEvent:
    """One tool invocation, tracked separately from raw think text."""

    name: str  # function name  e.g. "write_file"
    label: str  # display label  e.g. "writing main.py"
    status: ToolStatus = ToolStatus.PENDING
    stat: str = ""  # diff stat      e.g. "+24 -3" or "45L"


@dataclass
class ConfirmRequest:
    """Posted by the worker thread; resolved by the main event loop."""

    prompt: str
    result: Optional[bool] = None
    event: threading.Event = field(default_factory=threading.Event)


@dataclass
class TodoItem:
    """One step in the plan. Completed strictly in order."""

    text: str
    done: bool = False


@dataclass
class TurnRecord:
    """One completed user turn persisted in Application.history."""

    request: str
    tool_events: List["ToolEvent"]
    summary: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class Application:
    """Top-level application object; holds cross-turn persistent history."""

    history: List[TurnRecord] = field(default_factory=list)


@dataclass
class AppState:
    # ── Conversation history ─────────────────────────────────────────────────
    messages: List[dict] = field(default_factory=list)

    # ── Live streaming buffers ───────────────────────────────────────────────
    current_output: str = ""  # tokens that form the visible reply
    think_output: str = ""  # raw think text (no sentinels, plain text)

    # ── Tool event log for current turn ─────────────────────────────────────
    tool_events: List[ToolEvent] = field(default_factory=list)

    # ── Think timing ────────────────────────────────────────────────────────
    think_start: float = 0.0
    think_duration: float = 0.0

    # ── Dynamic think label ──────────────────────────────────────────────────
    think_label: str = "thinking…"

    # ── Scroll state ────────────────────────────────────────────────────────
    scroll_offset: int = 0  # chat scroll (lines from bottom)
    think_scroll: int = 0  # think-box scroll (lines from bottom)
    think_mode: int = 0  # 0=hidden 1=collapsed(3 lines) 2=expanded

    # ── Control ──────────────────────────────────────────────────────────────
    model_busy: bool = False
    stop_requested: bool = False
    running: bool = True
    reply_started: bool = False  # True only while final reply is streaming

    # ── Confirm request (worker → main thread) ──────────────────────────────
    confirm_request: Optional["ConfirmRequest"] = None

    # ── Plan / step tracking ─────────────────────────────────────────────────
    todo_items: List["TodoItem"] = field(default_factory=list)
    show_plan_view: bool = False  # i key: overlay showing full plan
    step_phase: str = ""  # "planning" / "step N/M" / "verifying" / …
    replan_count: int = 0

    # ── Completed-turn summary (shown in chat after busy = False) ────────────
    last_turn_tools: List[ToolEvent] = field(default_factory=list)

    lock: threading.Lock = field(default_factory=threading.Lock)


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  Tool schemas  (JSON passed to Ollama)
# ═══════════════════════════════════════════════════════════════════════════════

_MODEL = "qwen3:8b"
_SMALL_MODEL = "llama3.2:latest"

_TOOLS: List[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a file in the project directory. "
                "Small files (≤ ~16k tokens / 64k chars) are returned in full with line numbers. "
                "Large files are automatically split into parts and each part is condensed to "
                "signatures, class/function names, and docstrings only — no implementation bodies. "
                "Use start_line/end_line to read the exact content of any specific region, "
                "or search_file to locate the relevant lines first."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "1-based line to start reading from (inclusive). Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "1-based line to stop reading at (inclusive). Defaults to start_line + 299.",
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
            "description": (
                "Overwrite an existing file with new content. "
                "You MUST provide the COMPLETE file — partial writes are not allowed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Complete new content for the file.",
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
                        "description": "Relative path from project root.",
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
            "description": "Delete a file. User confirmation is required before deletion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_file",
            "description": (
                "Search for a pattern inside a file and return matching lines with context. "
                "Use this before read_file on large files to locate the region you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root.",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Case-insensitive substring or regex pattern to search for.",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Number of lines of context to show around each match (default 3).",
                    },
                },
                "required": ["path", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarise_file",
            "description": (
                "Read an entire file and return a short but detailed summary. "
                "Use instead of read_file when you only need to understand the contents, "
                "not quote or edit them. Much cheaper on context than reading the raw file."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path from project root.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the project directory and return its output. "
                "Use for: running tests, installing dependencies, building, linting, git commands, etc. "
                "User must confirm before the command executes. Timeout: 60 seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for up-to-date information: docs, APIs, error messages, stack traces, news. "
                "Returns a summary of the top results."
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
                "Submit your plan or draft implementation to an adversarial critic. "
                "The critic will find flaws, wrong assumptions, missing edge cases, and logic errors. "
                "Call this before committing to a non-trivial implementation, or after a previous attempt failed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {
                        "type": "string",
                        "description": "Your current plan, reasoning, or code draft to be critiqued.",
                    },
                },
                "required": ["plan"],
            },
        },
    },
]

# Restricted tool list for plan-mode single-task execution.
# No critic, no web search, no delete, no complete_task — just file ops.
_PLAN_TOOLS: List[dict] = [
    t
    for t in _TOOLS
    if t["function"]["name"]
    in (
        "read_file",
        "search_file",
        "summarise_file",
        "write_file",
        "create_file",
        "run_command",
    )
]


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  Tool implementations
# ═══════════════════════════════════════════════════════════════════════════════


def _safe_path(rel: str, root: Path) -> Optional[Path]:
    """Resolve *rel* inside *root*; return None if the path escapes."""
    try:
        resolved = (root / rel).resolve()
        resolved.relative_to(root)
        return resolved
    except Exception:
        return None


def _diff_stat(path: Optional[Path], before_lines: Optional[int], fn: str) -> str:
    """Compute a +N/-N stat string by comparing before/after line counts."""
    if path is None or before_lines is None:
        return ""
    try:
        after_lines = (
            len(path.read_text(encoding="utf-8").splitlines()) if path.exists() else 0
        )
    except Exception:
        return ""
    added = max(0, after_lines - before_lines)
    removed = max(0, before_lines - after_lines)
    if fn == "read_file":
        return f"  {before_lines}L"
    if fn == "delete_file" and removed > 0:
        return f"  -{removed}"
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    return ("  " + " ".join(parts)) if parts else ""


_READ_LINE_LIMIT  = 300   # max lines returned without explicit range (legacy path)
_READ_TOKEN_LIMIT = 16_000  # token budget for full-file reads  (~4 chars/token)
_READ_CHAR_LIMIT  = _READ_TOKEN_LIMIT * 4   # 64 000 chars


# ── Signature extractors (used when a file is too large for full read) ─────────

def _extract_signatures_python(content: str) -> str:
    """Return a header-file style skeleton for Python source."""
    out: List[str] = []
    lines = content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Class definition
        if stripped.startswith("class "):
            out.append(line.rstrip())
            # Grab the docstring if present
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].strip().startswith(('"""', "'''")):
                q = '"""' if lines[j].strip().startswith('"""') else "'''"
                doc_line = lines[j].strip()[3:]
                if q in doc_line:
                    out.append("    " + lines[j].strip().split(q)[0] + q + "…\"\"\"")
                else:
                    out.append("    " + lines[j].strip())
                    out.append("    …\"\"\"")
            out.append("")
            i += 1
            continue

        # Function / method definition (may span multiple lines)
        if stripped.startswith(("def ", "async def ")):
            sig_lines = [line.rstrip()]
            while not sig_lines[-1].rstrip().endswith(":") and i + len(sig_lines) < len(lines):
                sig_lines.append(lines[i + len(sig_lines)].rstrip())
            out.extend(sig_lines)
            # Grab the docstring
            j = i + len(sig_lines)
            while j < len(lines) and not lines[j].strip():
                j += 1
            # Indentation for the body stub: match the def line's indent + 4
            body_indent = " " * (len(sig_lines[0]) - len(sig_lines[0].lstrip()) + 4)
            if j < len(lines) and lines[j].strip().startswith(('"""', "'''")):
                q = '"""' if lines[j].strip().startswith('"""') else "'''"
                doc_line = lines[j].strip()[3:]
                if q in doc_line:
                    out.append(body_indent + doc_line.split(q)[0].strip())
                else:
                    # Multi-line docstring — grab first line
                    out.append(body_indent + "# " + lines[j].strip()[3:])
            out.append(body_indent + "...")
            out.append("")
            i += len(sig_lines)
            continue

        # Module-level constants / type aliases (not inside a block)
        indent_depth = len(line) - len(stripped)
        if indent_depth == 0 and stripped and not stripped.startswith("#"):
            if "=" in stripped and not stripped.startswith(("if ", "for ", "while ", "with ")):
                out.append(line.rstrip())
                i += 1
                continue

        i += 1
    return "\n".join(out)


def _extract_signatures_c_style(content: str, ext: str) -> str:
    """Return a header-file style skeleton for C / C++ / Java / C# / JS / TS."""
    out: List[str] = []
    # Match function/method definitions: optional modifiers + return type + name(…) {
    fn_re = re.compile(
        r"^[ \t]*"
        r"(?:(?:public|private|protected|static|async|override|virtual|abstract|"
        r"readonly|const|export|default|inline|extern|unsigned|signed|void|"
        r"int|long|short|char|bool|float|double|string|auto|var|let|[\w<>\[\]*&]+)\s+)+"
        r"[\w$][\w$]*\s*\([^)]*\)\s*(?::\s*[\w<>[\], ]+\s*)?\{?",
        re.MULTILINE,
    )
    class_re = re.compile(
        r"^[ \t]*(?:(?:public|private|protected|abstract|static|final|sealed)\s+)*"
        r"(?:class|interface|struct|enum)\s+\w+",
        re.MULTILINE,
    )

    lines = content.splitlines()
    i = 0
    brace_depth = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Track brace depth to detect top-level declarations
        if class_re.match(line):
            out.append(line.rstrip())
            out.append("")
        elif brace_depth == 0 and fn_re.match(line) and not stripped.startswith("//"):
            out.append(line.rstrip())
            # Grab inline comment or next-line doc comment as summary
            if i + 1 < len(lines):
                nxt = lines[i + 1].strip()
                if nxt.startswith(("//", "*", "/*")):
                    out.append("    " + nxt)
            out.append("    { … }")
            out.append("")

        brace_depth += stripped.count("{") - stripped.count("}")
        brace_depth = max(0, brace_depth)
        i += 1

    return "\n".join(out)


def _extract_signatures_rust(content: str) -> str:
    """Return a skeleton for Rust source: fn/pub fn, impl, struct, enum, trait."""
    out: List[str] = []
    lines = content.splitlines()
    i = 0
    # Collect doc-comment lines immediately above a declaration
    pending_doc: List[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Doc comments (/// or //!)
        if stripped.startswith("///") or stripped.startswith("//!"):
            pending_doc.append(line.rstrip())
            i += 1
            continue

        # Attributes (#[…])
        if stripped.startswith("#["):
            pending_doc.append(line.rstrip())
            i += 1
            continue

        # impl / struct / enum / trait / type alias / const / static
        if re.match(
            r"^\s*(?:pub(?:\(.+\))?\s+)?(?:impl|struct|enum|trait|type|const|static)\b",
            line,
        ):
            out.extend(pending_doc)
            out.append(line.rstrip())
            out.append("")
            pending_doc = []
            i += 1
            continue

        # fn / pub fn / async fn — may span multiple lines until the opening brace
        if re.match(r"^\s*(?:pub(?:\(.+\))?\s+)?(?:async\s+)?fn\b", line):
            sig = [line.rstrip()]
            j = i + 1
            while j < len(lines) and "{" not in sig[-1] and ";" not in sig[-1]:
                sig.append(lines[j].rstrip())
                j += 1
            out.extend(pending_doc)
            out.extend(sig)
            out.append("    { … }")
            out.append("")
            pending_doc = []
            i = j
            continue

        # Anything else clears the pending doc buffer
        pending_doc = []
        i += 1
    return "\n".join(out)


def _extract_signatures_zig(content: str) -> str:
    """Return a skeleton for Zig source: pub fn / fn, struct, enum, union, const."""
    out: List[str] = []
    lines = content.splitlines()
    i = 0
    pending_doc: List[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Doc comments (/// or //)
        if stripped.startswith("///") or stripped.startswith("//"):
            pending_doc.append(line.rstrip())
            i += 1
            continue

        # pub fn / fn
        if re.match(r"^\s*(?:pub\s+)?fn\b", line):
            sig = [line.rstrip()]
            j = i + 1
            # Collect continuation lines until opening brace
            while j < len(lines) and "{" not in sig[-1]:
                sig.append(lines[j].rstrip())
                j += 1
            out.extend(pending_doc)
            out.extend(sig)
            out.append("    // …")
            out.append("}")
            out.append("")
            pending_doc = []
            i = j
            continue

        # struct / enum / union / error set / comptime block
        if re.match(r"^\s*(?:pub\s+)?(?:const\b.*=\s*(?:struct|enum|union)\b|comptime\b)", line):
            out.extend(pending_doc)
            out.append(line.rstrip())
            out.append("    // …")
            out.append("};")
            out.append("")
            pending_doc = []
            i += 1
            continue

        # Top-level const / var declarations
        if re.match(r"^\s*(?:pub\s+)?(?:const|var)\b", line):
            out.extend(pending_doc)
            out.append(line.rstrip())
            out.append("")
            pending_doc = []
            i += 1
            continue

        pending_doc = []
        i += 1
    return "\n".join(out)


def _extract_signatures_php(content: str) -> str:
    """Return a skeleton for PHP source: function, class, interface, trait."""
    out: List[str] = []
    lines = content.splitlines()
    i = 0
    pending_doc: List[str] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # PHPDoc blocks (/** … */)
        if stripped.startswith("/**") or (pending_doc and stripped.startswith("*")):
            pending_doc.append(line.rstrip())
            if stripped.endswith("*/"):
                # Keep only first summary line of docblock
                summary = next(
                    (l.strip().lstrip("* ") for l in pending_doc if l.strip().startswith("*") and not l.strip().startswith("/**") and not l.strip().startswith("* @") and l.strip() not in ("*/", "/**")),
                    "",
                )
                pending_doc = ["    /** " + summary + " */"] if summary else []
            i += 1
            continue

        # class / interface / trait / abstract class
        if re.match(r"^\s*(?:abstract\s+)?(?:class|interface|trait)\b", line):
            out.extend(pending_doc)
            out.append(line.rstrip())
            out.append("{")
            out.append("    // …")
            out.append("}")
            out.append("")
            pending_doc = []
            i += 1
            continue

        # function declarations (standalone or methods)
        if re.match(
            r"^\s*(?:(?:public|protected|private|static|abstract|final|async)\s+)*function\b",
            line,
        ):
            sig = [line.rstrip()]
            j = i + 1
            while j < len(lines) and "{" not in sig[-1] and ";" not in sig[-1]:
                sig.append(lines[j].rstrip())
                j += 1
            out.extend(pending_doc)
            out.extend(sig)
            if "{" in sig[-1]:
                out.append("    // …")
                out.append("}")
            out.append("")
            pending_doc = []
            i = j
            continue

        # Class constants / properties at class scope
        if re.match(r"^\s*(?:public|protected|private|static|const)\s+", line):
            out.extend(pending_doc)
            out.append(line.rstrip())
            out.append("")
            pending_doc = []
            i += 1
            continue

        pending_doc = []
        i += 1
    return "\n".join(out)


def _extract_signatures_generic(content: str) -> str:
    """Fallback: return first 50 lines + last 20 lines with a gap marker."""
    lines = content.splitlines()
    if len(lines) <= 80:
        return content
    head = lines[:50]
    tail = lines[-20:]
    mid  = f"\n… [{len(lines) - 70} lines omitted] …\n"
    return "\n".join(head) + mid + "\n".join(tail)


def _make_file_skeleton(content: str, ext: str) -> str:
    """Convert oversized file content to a skeleton based on file type."""
    ext = ext.lower().lstrip(".")
    if ext == "py":
        return _extract_signatures_python(content)
    elif ext == "rs":
        return _extract_signatures_rust(content)
    elif ext == "zig":
        return _extract_signatures_zig(content)
    elif ext == "php":
        return _extract_signatures_php(content)
    elif ext in ("js", "ts", "jsx", "tsx", "java", "c", "cpp", "cs", "h", "hpp", "cc", "go", "kt", "swift"):
        return _extract_signatures_c_style(content, ext)
    else:
        return _extract_signatures_generic(content)


def _tool_read_file(
    path: Path, rel: str, start_line: int = 1, end_line: Optional[int] = None
) -> str:
    if not path.exists():
        return f"[ERROR] File not found: {rel}"
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[ERROR] {e}"

    all_lines = content.splitlines()
    total     = len(all_lines)
    ext       = path.suffix

    # ── Small file: fits in context — return whole thing ─────────────────────
    if len(content) <= _READ_CHAR_LIMIT and start_line == 1 and end_line is None:
        header = (
            f"[FILE READ: {rel}  ({total} lines, {len(content)} chars) — "
            "full file, this is a tool result not user input]\n"
        )
        body = "\n".join(f"{i + 1:>6}  {line}" for i, line in enumerate(all_lines))
        return header + body + f"\n[END OF FILE: {rel}]"

    # ── Explicit range requested: legacy paged read ───────────────────────────
    if start_line != 1 or end_line is not None:
        start = max(1, start_line)
        end   = min(end_line if end_line is not None else start + _READ_LINE_LIMIT - 1, total)
        chunk = all_lines[start - 1 : end]
        header = f"[FILE READ: {rel}  lines {start}-{end} of {total}  — this is a tool result, not user input]\n"
        body   = "\n".join(f"{start + i:>6}  {line}" for i, line in enumerate(chunk))
        footer = (
            f"\n[END OF CHUNK — {total - end} more lines remain. "
            f'To read the next section call read_file(path="{rel}", start_line={end + 1}). '
            f"Your task has not changed — continue working on it.]"
            if end < total
            else f"\n[END OF FILE: {rel}]"
        )
        return header + body + footer

    # ── Large file: split into parts and extract skeletons ───────────────────
    chars     = len(content)
    part_size = _READ_CHAR_LIMIT          # chars per chunk
    n_parts   = (chars + part_size - 1) // part_size
    parts: List[str] = [
        content[i * part_size : (i + 1) * part_size]
        for i in range(n_parts)
    ]

    out_parts: List[str] = [
        f"[FILE READ: {rel}  ({total} lines, {chars} chars) — "
        f"file is too large for a full read (~{chars // 4:,} tokens). "
        f"Split into {n_parts} part(s); each part is summarised to signatures/docstrings only. "
        f"Use search_file or read_file with start_line/end_line for the exact content of any section.]\n"
    ]

    for idx, part in enumerate(parts):
        part_lines = content[:sum(len(p) for p in parts[:idx + 1])].count("\n")
        skeleton   = _make_file_skeleton(part, ext)
        out_parts.append(
            f"\n{'─' * 60}\n"
            f"PART {idx + 1}/{n_parts}\n"
            f"{'─' * 60}\n"
            + skeleton
        )

    out_parts.append(f"\n[END OF SKELETON: {rel}]")
    return "\n".join(out_parts)


def _tool_search_file(
    path: Path, rel: str, pattern: str, context_lines: int = 3
) -> str:
    """Search for *pattern* (regex, case-insensitive) in *path*.
    Returns matching lines with line numbers and context."""
    if not path.exists():
        return f"[ERROR] File not found: {rel}"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return f"[ERROR] {e}"

    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"[ERROR] Invalid pattern: {e}"

    hits: List[int] = [i for i, ln in enumerate(lines) if rx.search(ln)]
    if not hits:
        return f"No matches for {pattern!r} in {rel}"

    ctx = max(0, context_lines)
    shown: set = set()
    blocks: List[str] = []
    for h in hits:
        lo = max(0, h - ctx)
        hi = min(len(lines) - 1, h + ctx)
        if lo in shown:
            continue
        block_lines = []
        for i in range(lo, hi + 1):
            shown.add(i)
            marker = ">" if i == h else " "
            block_lines.append(f"{i + 1:>6} {marker} {lines[i]}")
        blocks.append("\n".join(block_lines))

    total_matches = len(hits)
    header = f"[{rel} - {total_matches} match{'es' if total_matches != 1 else ''} for {pattern!r}]\n"
    return header + "\n---\n".join(blocks)


def _tool_summarise_file(path: Path, rel: str) -> str:
    """Read the full file and ask qwen3 for a concise but detailed summary.
    The summary is capped so it never blows the context window."""
    if not path.exists():
        return f"[ERROR] File not found: {rel}"
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return f"[ERROR] Cannot read {rel}: {e}"

    total_lines = content.count("\n") + 1
    # Truncate input to ~120 KB of text to avoid OOM on huge files
    MAX_CHARS = 120_000
    truncated = ""
    if len(content) > MAX_CHARS:
        content = content[:MAX_CHARS]
        truncated = (
            f" (first {MAX_CHARS // 1000} KB shown; file has {total_lines} lines)"
        )

    prompt = (
        f"File: {rel}{truncated}\n"
        "Summarise the file contents below. Be concise but include:\n"
        "- What the file does / its purpose\n"
        "- Key functions, classes, or data structures\n"
        "- Important constants, config values, or dependencies\n"
        "- Any non-obvious behaviour worth noting\n"
        "Keep the summary under 200 words.\n\n"
        f"```\n{content}\n```"
    )
    try:
        resp = ollama.chat(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            options={"temperature": 0},
        )
        summary = (resp.message.content or "").strip()
        summary = re.sub(r"<think>.*?</think>", "", summary, flags=re.DOTALL).strip()
        return f"[SUMMARY: {rel} ({total_lines} lines)\n{summary}]"
    except Exception as e:
        return f"[ERROR] summarise_file failed: {e}"


def _file_diff(rel: str, before: str, after: str) -> str:
    """Return a unified diff between *before* and *after* content.
    Gives the model a precise record of exactly what changed so it does not
    re-derive the content on the next thinking turn."""
    import difflib

    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
            lineterm="",
        )
    )
    after_n = len(after.splitlines())
    if not diff:
        return f"OK: {after_n} lines — no changes (content identical)"
    diff_text = "\n".join(diff)
    added = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))
    return f"OK: wrote {after_n} lines to {rel} (+{added} -{removed})\n```diff\n{diff_text}\n```"


def _tool_write_file(path: Path, rel: str, content: str) -> str:
    if not path.exists():
        return f"[ERROR] File does not exist (use create_file): {rel}"
    try:
        before = path.read_text(encoding="utf-8")
        path.write_text(content, encoding="utf-8")
        return _file_diff(rel, before, content)
    except Exception as e:
        return f"[ERROR] {e}"


def _tool_create_file(path: Path, rel: str, content: str) -> str:
    if path.exists():
        return f"[ERROR] File already exists: {rel}"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        lines = content.splitlines()
        n = len(lines)
        if n <= 10:
            preview = "\n".join(lines)
        else:
            head = "\n".join(lines[:4])
            tail = "\n".join(lines[-3:])
            preview = f"{head}\n… ({n - 7} lines) …\n{tail}"
        return f"OK: created {rel} ({n} lines)\n---\n{preview}\n---"
    except Exception as e:
        return f"[ERROR] {e}"


def _tool_delete_file(path: Path, rel: str) -> str:
    if not path.exists():
        return f"[ERROR] File not found: {rel}"
    try:
        path.unlink()
        return f"OK: deleted {rel}"
    except Exception as e:
        return f"[ERROR] {e}"


def _tool_run_command(command: str, cwd: Path) -> str:
    """Execute *command* in *cwd* with a 60-second timeout. Return combined output."""
    try:
        # Prepend an explicit `cd` so that shell=True reliably runs in the
        # right directory regardless of the Python process's own cwd.
        full_cmd = f"cd {cwd.absolute().as_posix()!r} && {command}"
        result = subprocess.run(
            full_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        parts: List[str] = []
        if result.stdout.strip():
            parts.append(result.stdout.strip())
        if result.stderr.strip():
            parts.append(f"stderr:\n{result.stderr.strip()}")
        parts.append(f"exit: {result.returncode}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return "[ERROR] Command timed out after 60 seconds."
    except Exception as e:
        return f"[ERROR] {e}"


def _tool_web_search(query: str) -> str:
    """DuckDuckGo search → fetch top pages → summarise with small model."""
    # 1. Search
    urls: List[str] = []
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3, region="us-en"))
            urls = [str(r["href"]) for r in results if r.get("href")]
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))

    if not urls:
        return "[ERROR] Search failed — no results returned."

    # 2. Fetch + summarise each unique domain
    summaries: List[str] = []
    seen_domains: set = set()

    for url in urls:
        if not url.startswith("http"):
            continue
        domain = re.sub(r"https?://", "", url).split("/")[0]
        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        try:
            html = requests.get(
                url, timeout=6, headers={"User-Agent": "Mozilla/5.0"}
            ).text
            text = BeautifulSoup(html, "html.parser").get_text(
                separator=" ", strip=True
            )
            text = text[:5000]

            resp = ollama.chat(
                model=_SMALL_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f'Extract only the information relevant to: "{query}". '
                            "Be concise and factual. Respond ONLY with the relevant content, "
                            "no preamble, no commentary."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                stream=False,
                options={"num_predict": 300, "temperature": 0},
            )
            summary = (resp.message.content or "").strip()
            if summary:
                summaries.append(f"[{url}]\n{summary}")
        except Exception:
            continue

    return (
        "\n\n".join(summaries)
        if summaries
        else "[ERROR] Could not retrieve content from any result."
    )


def _tool_consult_critic(plan: str) -> str:
    """Run *plan* through an adversarial critic instance and return its verdict."""
    _CRITIC_SYSTEM = (
        "You are a ruthless adversarial code and logic critic. "
        "Your ONLY job is to find what is WRONG.\n\n"
        "Rules:\n"
        "- Be cold, direct, specific. No praise. No encouragement.\n"
        "- List every flaw: incorrect assumptions, missing edge cases, logic errors, security issues.\n"
        "- If the approach is fundamentally broken, say so and explain why.\n"
        "- Skip anything that is correct — only report problems.\n"
        "- Be concrete: cite exact lines, variable names, or steps that are wrong.\n"
        '- If you find nothing wrong, say exactly: "No issues found." Nothing else.'
    )
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
        verdict = re.sub(r"<think>.*?</think>", "", verdict, flags=re.DOTALL).strip()
        return verdict or "Critic returned an empty response."
    except Exception as e:
        return f"[ERROR] Critic call failed: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  Think labeller  (background daemon thread)
# ═══════════════════════════════════════════════════════════════════════════════


def _start_think_labeller(state: AppState) -> None:
    """
    Daemon thread: every ~2 s, summarise the last few paragraphs of think_output
    into a short status phrase and write it to state.think_label.
    Stops automatically when model_busy becomes False.
    """
    _POLL = 2.0
    _MIN_CHARS = 60

    _SYSTEM = (
        "You are a concise status-line generator. "
        "The user will paste raw reasoning text from an AI. "
        "Reply with a single short phrase (4-8 words, present tense, lowercase) "
        "describing what the AI is doing — e.g. 'parsing function signatures' or "
        "'checking edge cases for division'. "
        "Output ONLY that phrase. No punctuation at the end. Nothing else."
    )

    def _run() -> None:
        last_snapshot = ""
        while True:
            time.sleep(_POLL)
            with state.lock:
                if not state.model_busy:
                    state.think_label = "thinking…"
                    return
                raw = state.think_output

            # Keep last 8 non-empty paragraphs
            paras = [p.strip() for p in re.split(r"\n{2,}", raw) if p.strip()]
            excerpt = "\n\n".join(paras[-8:])

            if len(excerpt) < _MIN_CHARS or excerpt == last_snapshot:
                continue
            last_snapshot = excerpt

            try:
                resp = ollama.chat(
                    model=_SMALL_MODEL,
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
                label = label.rstrip(".,;:")
                if label:
                    with state.lock:
                        state.think_label = label + "…"
            except Exception:
                pass  # keep previous label on error

    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  AgentWorker  — 8-step structured pipeline
# ═══════════════════════════════════════════════════════════════════════════════

_SPINNER = ["·", "✻", "✽", "✶", "✳", "✢", "✳", "✶", "✽", "✻"]


class AgentWorker:
    """
    Runs the structured 8-step agent pipeline on a background daemon thread.

    Pipeline per user turn:
        1. User enters a prompt
        2. System prompt is built with current file layout            (caller)
        3. Ask model to create a step-by-step plan                   (temp=0.5)
        4. Summarise user prompt into one tight sentence             (temp=0)
        5. For each step:
             5.1  Collect context: request + past tool calls + think summaries
             5.2  Think + call tools until step resolved
             5.3  Ask model whether step goal was achieved; retry if not
             5.4  Log tool usage; store thinking summary
             5.5  Advance to next step
        6. Ask model whether overall goal achieved; replan+restart if not
        7. Ask model for human-readable output summary
        8. Persist request / tool usage / summary to Application.history
    """

    _PENDING_VERB: Dict[str, str] = {
        "read_file": "reading",
        "write_file": "writing",
        "create_file": "creating",
        "delete_file": "deleting",
        "run_command": "running",
        "web_search": "searching",
        "consult_critic": "reviewing plan",
    }
    _DONE_VERB: Dict[str, str] = {
        "read_file": "read",
        "write_file": "wrote",
        "create_file": "created",
        "delete_file": "deleted",
        "run_command": "ran",
        "web_search": "searched",
        "consult_critic": "critic reviewed",
    }

    def __init__(
        self, state: AppState, canvas: Canvas, root: Path, app: "Application"
    ) -> None:
        self._state = state
        self._canvas = canvas
        self._root = root
        self._app = app
        self._task = ""

    # ── Public API ────────────────────────────────────────────────────────────

    def start(
        self,
        messages: List[dict],
        system_prompt: str,
        task: str = "",
    ) -> None:
        self._task = task.strip()
        threading.Thread(
            target=self._run,
            args=(list(messages), system_prompt),
            daemon=True,
        ).start()

    # ── Private: 8-step pipeline ─────────────────────────────────────────────

    def _run(self, messages: List[dict], system_prompt: str) -> None:
        """Orchestrates the full 8-step pipeline for one user turn."""
        with self._state.lock:
            self._state.tool_events = []
            self._state.think_output = ""
            self._state.current_output = ""
            self._state.reply_started = False
            self._state.todo_items = []
            self._state.step_phase = "planning"
            self._state.replan_count = 0

        task = self._task
        # history accumulates compressed step summaries — used only for
        # step 6 goal check and step 7 summary.  It is NOT passed to the
        # step executor; each step sees only its own context.
        history: List[dict] = list(messages)
        output_acc = ""

        try:
            # ── Step 3: build plan (temperature=0.5) ─────────────────────────
            steps = self._make_plan(system_prompt, task)

            # ── Step 4: summarise request (temperature=0) ────────────────────
            task_summary = self._summarise_request(task)

            # Accumulators carried across steps
            all_tool_events: List[ToolEvent] = []   # every tool call so far
            step_outputs:    List[str]        = []   # text output per completed step
            # Maps file path → step index that last wrote it.
            # Used to warn the executing model not to clobber prior work.
            files_written:   Dict[str, int]   = {}

            _MAX_REPLANS = 2
            for replan_attempt in range(_MAX_REPLANS + 1):
                if self._stopped():
                    break

                with self._state.lock:
                    self._state.todo_items = [TodoItem(text=s) for s in steps]
                    self._state.replan_count = replan_attempt

                for step_idx, step_text in enumerate(steps):
                    if self._stopped():
                        break

                    # ── 5.1 / 5.2 / 5.3: attempt step, retry from 5.1 on fail ─
                    with self._state.lock:
                        self._state.step_phase = f"step {step_idx + 1}/{len(steps)}"
                        self._state.think_output = ""
                        self._state.current_output = ""
                        self._state.reply_started = False
                        self._state.tool_events = []

                    _MAX_RETRIES = 2
                    step_ok = False
                    step_output = ""
                    step_history: List[dict] = []  # always bound after the loop

                    for _retry in range(_MAX_RETRIES):
                        if self._stopped():
                            break

                        # ── 5.1: build context ────────────────────────────────
                        # Model receives ONLY: current task + past tool calls
                        # + previous step outputs.  No raw user prompt, no thinking.
                        step_sys = self._build_step_system(
                            system_prompt,
                            step_text,
                            step_idx,
                            len(steps),
                            all_tool_events,
                            step_outputs,
                            files_written,
                        )

                        # Fresh history for each attempt — model never sees
                        # its own prior thinking or the outer conversation.
                        step_history: List[dict] = []

                        # ── 5.2: think + call tools ───────────────────────────
                        step_output, _think = self._execute_step(
                            step_history, step_sys, step_text
                        )

                        # ── 5.3: check step goal; loop back to 5.1 if not done
                        step_ok = self._check_step_goal(step_text, step_history)
                        if step_ok:
                            break
                        # retry: fresh step_history rebuilt at top of loop

                    # ── 5.4: log tool usage; store output for next step ───────
                    with self._state.lock:
                        last_tools = list(self._state.tool_events)
                        self._state.last_turn_tools = last_tools
                        self._state.tool_events = []

                    all_tool_events.extend(last_tools)
                    if step_output.strip():
                        step_outputs.append(step_output.strip())

                    # Record files written/created so later steps know not to clobber them
                    writing_tools = {"write_file", "create_file"}
                    for ev in last_tools:
                        if ev.status == ToolStatus.DONE and ev.name in writing_tools:
                            files_written[ev.label] = step_idx

                    # Compress step into history for goal-check / summary context
                    compressed = self._summarise_step(step_text, step_history)
                    mutating = {
                        "write_file",
                        "create_file",
                        "run_command",
                        "delete_file",
                    }
                    files_done = [
                        ev.label
                        for ev in last_tools
                        if ev.status == ToolStatus.DONE and ev.name in mutating
                    ]
                    if files_done:
                        compressed += f"\nFiles changed: {', '.join(files_done)}"

                    history.append({"role": "assistant", "content": compressed})
                    with self._state.lock:
                        self._state.messages.append(
                            {"role": "assistant", "content": compressed}
                        )
                        self._state.current_output = ""
                        self._state.reply_started = False

                    # ── 5.5: advance ──────────────────────────────────────────
                    with self._state.lock:
                        if step_idx < len(self._state.todo_items):
                            self._state.todo_items[step_idx].done = True

                if self._stopped():
                    break

                # ── Step 6: check overall goal ────────────────────────────────
                with self._state.lock:
                    self._state.step_phase = "verifying"

                overall_ok = self._check_overall_goal(task_summary, history)

                if overall_ok:
                    break

                # Tell the user what happened
                can_replan = replan_attempt < _MAX_REPLANS
                fail_msg = (
                    f"The overall goal was not fully achieved after {len(steps)} step(s). "
                    + ("Replanning and retrying…" if can_replan else "Stopping.")
                )
                with self._state.lock:
                    self._state.messages.append(
                        {"role": "assistant", "content": fail_msg}
                    )
                history.append({"role": "assistant", "content": fail_msg})

                if not can_replan:
                    break

                # Replan using previous outputs as context
                with self._state.lock:
                    self._state.step_phase = "replanning"
                prev_ctx = "\n".join(step_outputs[-4:])
                steps = self._make_plan(system_prompt, task, prior_context=prev_ctx)
                all_tool_events = []
                step_outputs    = []
                files_written   = {}

            # ── Step 7: human-readable output summary ─────────────────────────
            with self._state.lock:
                self._state.step_phase = "summarising"

            output_acc = self._generate_output_summary(task_summary, history)

            with self._state.lock:
                self._state.current_output = output_acc
                self._state.reply_started = True

            # ── Step 8: persist to Application.history ────────────────────────
            self._app.history.append(
                TurnRecord(
                    request=task,
                    tool_events=list(all_tool_events),
                    summary=output_acc,
                )
            )

            with self._state.lock:
                self._state.messages = history[:]

        except Exception as exc:
            output_acc = f"[ERROR] {exc}"
            with self._state.lock:
                self._state.current_output = output_acc

        finally:
            self._finalize(output_acc)

    # ── Step 3: create plan ───────────────────────────────────────────────────

    def _make_plan(
        self, system_prompt: str, task: str, prior_context: str = ""
    ) -> List[str]:
        """Ask the main model for a numbered step list. temperature=0.5."""
        with self._state.lock:
            self._state.step_phase = "planning"

        extra = (
            f"\n\nPrevious attempt context:\n{prior_context}" if prior_context else ""
        )
        prompt = (
            "You are a PLANNER, not an implementer. Do not implement anything.\n"
            "A separate agent will do the actual work — your only output is a list of steps.\n"
            "\n"
            f"Overall task: {task}{extra}\n\n"
            "Produce a numbered list of steps the agent must execute IN ORDER.\n"
            "\n"
            "DECOMPOSITION RULES:\n"
            "— Split the work into the SMALLEST possible steps where each step\n"
            "  touches exactly ONE file or runs exactly ONE command.\n"
            "— If a task involves multiple files, give each file its own step.\n"
            "— If a task involves both writing and running, split them: one step\n"
            "  to write, one step to run.\n"
            "— Never bundle two separate concerns into one step.\n"
            "\n"
            "UNIQUENESS RULES:\n"
            "— Every step must be UNIQUE. Do not repeat a step or rephrase the\n"
            "  same action twice. Each file appears AT MOST ONCE as a creation\n"
            "  target. Additions to an existing file count as edits, not creation.\n"
            "— Do not create a file and then immediately 'update' it — decide its\n"
            "  full contents once and write it in a single step.\n"
            "— Before writing each step, check: have I already included this file\n"
            "  or this action? If yes, merge or drop the duplicate.\n"
            "\n"
            "PRECISION RULES:\n"
            "— Each step must be so precise that there is NOTHING left for the\n"
            "  implementing agent to decide. If the agent could ask a clarifying\n"
            "  question, the step is not precise enough.\n"
            "— State the exact file path, the exact purpose of every section or\n"
            "  function, and the exact behaviour expected — all in plain English.\n"
            "— Do NOT leave implicit assumptions. If a file must have a title,\n"
            "  state it. If a function returns a value, state what. If a command\n"
            "  needs a flag, state the flag.\n"
            "\n"
            "FORMAT RULES:\n"
            "— Do NOT write code, snippets, or file content inside a step.\n"
            "— Do NOT use code fences (``` or `).\n"
            "— Do NOT include reading, searching, or planning sub-steps.\n"
            "— One tool call per step: write_file, create_file, or run_command.\n"
            "\n"
            "GOOD examples (atomic, unique, precise):\n"
            "1. Create src/config.py with a Config dataclass containing three fields:\n"
            "   host (str, default 'localhost'), port (int, default 8080), and\n"
            "   debug (bool, default False).\n"
            "2. Create src/server.py with a Server class whose constructor accepts\n"
            "   a Config instance and stores it as self.config, and a start() method\n"
            "   that prints 'Listening on host:port' using the config values.\n"
            "3. Create tests/test_server.py with two pytest test functions: one that\n"
            "   asserts Config() produces default field values, and one that asserts\n"
            "   Server(Config()).start() prints the correct listening message.\n"
            "4. Run pytest on tests/ with the -v flag.\n"
            "\n"
            "BAD examples (never do this):\n"
            "1. Create config.py and server.py with the core logic.\n"
            "   ↳ Bad: two files in one step.\n"
            "2. Create README.md in the root directory.\n"
            "   ↳ Bad: says nothing about what goes inside.\n"
            "3. Update server.py to fix the imports.\n"
            "   ↳ Bad: vague; also likely a duplicate if server.py was just created.\n"
            "\n"
            "Reply ONLY with the numbered list. No preamble, no code.\n\nSteps:"
        )
        try:
            resp = ollama.chat(
                model=_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a PLANNER. Your only job is to produce a numbered "
                            "list of steps. You do NOT implement, write content, generate "
                            "code, or produce any file contents whatsoever. "
                            "You describe WHAT needs to be done, never HOW or WHAT the "
                            "content should be. The actual implementation is done by a "
                            "separate agent after you finish. Stay in planning mode only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                stream=False,
                options={"temperature": 0.3},
            )
            raw = re.sub(
                r"<think>.*?</think>",
                "",
                resp.message.content or "",
                flags=re.DOTALL,
            ).strip()
            steps: List[str] = []
            in_fence = False
            for line in raw.splitlines():
                if line.strip().startswith("```"):
                    in_fence = not in_fence
                    continue
                if in_fence:
                    continue

                stripped = line.strip()
                if not stripped:
                    continue

                # Numbered top-level step (e.g. "1. …" / "Step 2: …")
                num_m = re.match(
                    r"^\s*(?:[Ss]tep\s*)?\d+[.):\-]\s+(.*)",
                    line,
                )
                if num_m:
                    text = num_m.group(1).strip()
                    text = re.sub(r"`[^`]*`", "", text).strip()
                    text = re.sub(r"\s*:\s*```.*", "", text, flags=re.DOTALL).strip()
                    if text:
                        steps.append(text)
                    continue

                # Bullet / dash / asterisk line — treat as continuation of the
                # previous step, NOT a new step (the model often emits sub-bullets
                # as part of a step description that the old code split into steps)
                bullet_m = re.match(r"^\s*[-*•]\s+(.*)", line)
                if bullet_m and steps:
                    extra = bullet_m.group(1).strip()
                    extra = re.sub(r"`[^`]*`", "", extra).strip()
                    if extra:
                        steps[-1] = steps[-1].rstrip(":").rstrip() + "; " + extra
                    continue

                # Indented continuation line (no bullet, no number) — also fold in
                if line.startswith(("   ", "\t")) and steps and stripped:
                    cleaned = re.sub(r"`[^`]*`", "", stripped).strip()
                    if cleaned:
                        steps[-1] = steps[-1].rstrip() + " " + cleaned

            # ── Post-parse deduplication ──────────────────────────────────────
            # Remove steps that are near-identical to an earlier step.
            # Similarity: normalise to lowercase words, check overlap ratio.
            def _word_set(s: str) -> set:
                return set(re.findall(r"\w+", s.lower()))

            deduped: List[str] = []
            for step in steps:
                ws = _word_set(step)
                is_dup = any(
                    len(ws & _word_set(prev)) / max(len(ws | _word_set(prev)), 1) > 0.75
                    for prev in deduped
                )
                if not is_dup:
                    deduped.append(step)

            return deduped if deduped else [task]
        except Exception:
            return [task]

    # ── Step 4: summarise request ─────────────────────────────────────────────

    def _summarise_request(self, task: str) -> str:
        """Compress the user prompt to one tight sentence. temperature=0."""
        try:
            resp = ollama.chat(
                model=_SMALL_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Summarise this task in ONE tight sentence (max 20 words): "
                            + task
                        ),
                    }
                ],
                stream=False,
                options={"temperature": 0, "num_predict": 40},
            )
            return (
                re.sub(
                    r"<think>.*?</think>",
                    "",
                    resp.message.content or "",
                    flags=re.DOTALL,
                )
                .strip()
                .split("\n")[0]
            )
        except Exception:
            return task[:100]

    # ── Step 5.1: build step system prompt ───────────────────────────────────
    # Context = current task + past tool calls + previous step outputs.
    # No user prompt, no raw thinking.

    def _build_step_system(
        self,
        base_sys: str,
        step_text: str,
        step_idx: int,
        total_steps: int,
        tool_events: List[ToolEvent],
        step_outputs: List[str],
        files_written: Dict[str, int],
    ) -> str:
        """Build a clean system prompt for one step execution attempt."""
        ctx_parts: List[str] = []

        # ── Past tool calls ───────────────────────────────────────────────────
        if tool_events:
            recent = tool_events[-12:]
            lines = [
                f"  {ev.status.name}: {ev.label}"
                + (f"  {ev.stat}" if ev.stat else "")
                for ev in recent
            ]
            ctx_parts.append("PAST TOOL CALLS:\n" + "\n".join(lines))

        # ── Previous step outputs ─────────────────────────────────────────────
        if step_outputs:
            recent_out = step_outputs[-4:]
            ctx_parts.append(
                "PREVIOUS STEP OUTPUTS:\n"
                + "\n".join(
                    f"  [{i + 1}] {o[:300]}" for i, o in enumerate(recent_out)
                )
            )

        ctx_block = (
            "\n\n════════════════════════════════════════\n"
            "CONTEXT FROM PREVIOUS STEPS\n"
            "════════════════════════════════════════\n"
            + "\n\n".join(ctx_parts)
            if ctx_parts
            else ""
        )

        # ── No-clobber guard ──────────────────────────────────────────────────
        # Tell the model which files already exist from prior steps so it
        # updates them instead of replacing them wholesale.
        if files_written:
            file_lines = "\n".join(
                f"  • {path}  (written in step {idx + 1})"
                for path, idx in sorted(files_written.items(), key=lambda x: x[1])
            )
            no_clobber = (
                "\n\n════════════════════════════════════════\n"
                "FILES WRITTEN BY PREVIOUS STEPS\n"
                "════════════════════════════════════════\n"
                + file_lines
                + "\n\n"
                "These files already contain work from earlier steps.\n"
                "If this step needs to modify any of them:\n"
                "  1. Call read_file first to get the current content.\n"
                "  2. Call write_file with the COMPLETE updated content,\n"
                "     preserving ALL existing content and only making the\n"
                "     changes required by this step.\n"
                "NEVER write one of these files from scratch — you will\n"
                "destroy the work done by the previous steps."
            )
        else:
            no_clobber = ""

        # ── Permitted actions (derived from the step text) ────────────────────
        allows_run, allowed_paths = self._extract_step_scope(step_text)
        permit_lines: List[str] = []
        if allowed_paths:
            permit_lines.append(
                "Files you may touch: " + ", ".join(allowed_paths)
            )
        if allows_run:
            permit_lines.append("You may call run_command for this step.")
        else:
            permit_lines.append("You may NOT call run_command for this step.")
        permitted_block = (
            "\n\n════════════════════════════════════════\n"
            "PERMITTED ACTIONS FOR THIS STEP\n"
            "════════════════════════════════════════\n"
            + "\n".join(permit_lines)
        )

        return (
            base_sys
            + "\n\n════════════════════════════════════════\n"
            + f"CURRENT STEP ({step_idx + 1}/{total_steps})\n"
            + "════════════════════════════════════════\n"
            + step_text
            + ctx_block
            + no_clobber
            + permitted_block
            + "\n\n════════════════════════════════════════\n"
            "STRICT EXECUTION CONTRACT\n"
            "════════════════════════════════════════\n"
            "The step description above is your COMPLETE and EXHAUSTIVE specification.\n"
            "You are only permitted to do what it explicitly states — nothing more.\n"
            "\n"
            "Before every tool call, ask yourself:\n"
            "  'Is this call directly required by a specific sentence in the step?'\n"
            "  If the answer is NO — do not make the call.\n"
            "\n"
            "You are FORBIDDEN from:\n"
            "— Touching any file not listed under PERMITTED ACTIONS.\n"
            "— Calling run_command unless PERMITTED ACTIONS says you may.\n"
            "— Adding imports, helpers, or extras not described in the step.\n"
            "— Running tests, linters, or checks unless the step explicitly says to.\n"
            "— Fixing unrelated bugs or making improvements you noticed.\n"
            "— Doing work that belongs to a different step.\n"
            "\n"
            "Stop the moment the step is complete. Write one sentence stating what you did."
        )

    # ── Step 5.2: execute step tool loop ─────────────────────────────────────

    def _execute_step(
        self,
        step_history: List[dict],  # fresh per attempt; modified in-place
        step_sys: str,
        step_text: str,
    ) -> Tuple[str, str]:
        """Run the tool-call loop for one step attempt.

        Returns (content_output, think_acc).
        step_history is the isolated history for this attempt only —
        no outer conversation, no prior thinking.

        After every tool call the step goal is checked immediately so the
        model cannot drift into over-achieving by continuing uninstructed.
        Tool calls outside the step's declared scope are blocked before execution.
        """
        think_acc   = ""
        content_out = ""
        _MAX_TURNS  = 12
        allows_run, allowed_paths = self._extract_step_scope(step_text)

        for _ in range(_MAX_TURNS):
            if self._stopped():
                break

            content, think_chunk, tool_calls, asst_msg = self._stream_one_turn(
                step_history,
                step_sys,
                think_acc,
                temperature=0.15,
                inject_plan=False,
                tools=_PLAN_TOOLS,
            )
            think_acc   += think_chunk
            content_out  = content
            step_history.append(asst_msg)

            if not tool_calls:
                with self._state.lock:
                    self._state.current_output = ""
                    self._state.reply_started  = False
                break

            with self._state.lock:
                self._state.reply_started  = False
                self._state.current_output = ""

            # ── Gate: block calls outside the step's declared scope ───────────
            permitted: List[Any] = []
            for tc in tool_calls:
                fn   = tc.function.name
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                reason = self._gate_tool_call(fn, args, allows_run, allowed_paths)
                if reason:
                    # Inject a synthetic tool result so the model knows it was blocked
                    step_history.append({
                        "role":      "tool",
                        "tool_name": fn,
                        "content":   reason,
                    })
                else:
                    permitted.append(tc)

            if permitted:
                think_acc = self._execute_tool_calls(permitted, step_history, think_acc)

            # Check immediately after tool use — stop as soon as the step is
            # satisfied rather than letting the model continue unnecessarily.
            if self._check_step_goal(step_text, step_history):
                break

        return content_out, think_acc

    def _extract_step_scope(self, step_text: str) -> Tuple[bool, List[str]]:
        """Parse the step description to derive what the executor is permitted to do.

        Returns:
            allows_run   — True if the step explicitly involves running a command.
            allowed_paths — list of file/directory paths mentioned in the step.
        """
        # run_command is allowed only when the step clearly intends it
        run_keywords = re.compile(
            r"\b(run|execute|invoke|call|start|launch|test|pytest|lint|build|install|"
            r"compile|make|npm|pip|cargo|zig build|go run|docker|bash|sh)\b",
            re.IGNORECASE,
        )
        allows_run = bool(run_keywords.search(step_text))

        # Extract path-like tokens: things with slashes, dots-followed-by-ext,
        # or quoted strings that look like file paths
        path_re = re.compile(
            r"(?:"
            r"['\"]([^'\"]+\.[a-zA-Z0-9_]+)['\"]"   # quoted  'foo/bar.py'
            r"|"
            r"\b([\w./\\-]+\.[\w]{1,10})\b"           # bare    src/utils.py
            r"|"
            r"\b([\w-]+/[\w./\\-]+)\b"                # dir/    src/utils
            r")"
        )
        paths: List[str] = []
        for m in path_re.finditer(step_text):
            p = m.group(1) or m.group(2) or m.group(3) or ""
            p = p.strip("'\"/")
            if p and ".." not in p:
                paths.append(p)

        # Deduplicate while preserving order
        seen: set = set()
        allowed_paths = [p for p in paths if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]
        return allows_run, allowed_paths

    def _gate_tool_call(
        self,
        fn: str,
        args: dict,
        allows_run: bool,
        allowed_paths: List[str],
    ) -> str:
        """Return a block reason string if the call is out of scope, else empty string."""
        # run_command is gated by whether the step mentions running something
        if fn == "run_command" and not allows_run:
            return (
                "[BLOCKED] run_command is not permitted for this step. "
                "The step description does not include running a command. "
                "Complete the step using file operations only."
            )

        # File-mutating tools are gated by path, but only when the step explicitly
        # names specific files.  If no paths could be extracted (the step is phrased
        # in terms of behaviour rather than exact paths), we skip the check — the
        # prompt instructions are the only guard in that case.
        mutating = {"write_file", "create_file", "delete_file"}
        if fn in mutating and allowed_paths:
            target = args.get("path", "")
            # Allow if the target matches any allowed path suffix
            # (handles "src/foo.py" matching "foo.py" mention in the step)
            matched = any(
                target == p or target.endswith("/" + p) or p.endswith("/" + target)
                for p in allowed_paths
            )
            if not matched:
                return (
                    f"[BLOCKED] {fn}({target!r}) is not permitted for this step. "
                    f"The step only authorises: {', '.join(allowed_paths)}. "
                    "Do not touch files that are not explicitly mentioned in the current step."
                )

        return ""

    # ── Step 5.3: check step goal ─────────────────────────────────────────────

    def _check_step_goal(self, step_text: str, step_history: List[dict]) -> bool:
        """Ask small model whether the step goal was achieved. YES/NO, temp=0."""
        recent = step_history[-6:]
        ctx = "\n".join(
            f"{m['role']}: {str(m.get('content', ''))[:300]}" for m in recent
        )
        try:
            resp = ollama.chat(
                model=_SMALL_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Step goal: {step_text}\n\n"
                            f"Recent actions:\n{ctx}\n\n"
                            "Was this step goal fully achieved? Reply YES or NO only."
                        ),
                    }
                ],
                stream=False,
                options={"temperature": 0, "num_predict": 5},
            )
            ans = (
                re.sub(
                    r"<think>.*?</think>",
                    "",
                    resp.message.content or "",
                    flags=re.DOTALL,
                )
                .strip()
                .upper()
            )
            return ans.startswith("YES")
        except Exception:
            return True  # assume ok on model error

    # ── Step 5.4 helper: one-line summary for history compression ────────────

    def _summarise_step(self, step_text: str, step_history: List[dict]) -> str:
        """Compress what was done for a step into one sentence."""
        recent = step_history[-6:]
        ctx = "\n".join(
            f"{m['role']}: {str(m.get('content', ''))[:300]}" for m in recent
        )
        prompt = (
            f"Task: {step_text}\n\n"
            f"Recent actions:\n{ctx}\n\n"
            "Write ONE short sentence (max 15 words) summarising what was just done. "
            "No preamble, no quotes."
        )
        try:
            resp = ollama.chat(
                model=_SMALL_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0},
            )
            return (
                re.sub(
                    r"<think>.*?</think>",
                    "",
                    resp.message.content or "",
                    flags=re.DOTALL,
                )
                .strip()
                .split("\n")[0]
            )
        except Exception:
            return f"Completed: {step_text}"

    # ── Step 6: check overall goal ────────────────────────────────────────────

    def _check_overall_goal(self, task_summary: str, history: List[dict]) -> bool:
        """Ask small model if overall goal was fully achieved. YES/NO, temp=0."""
        recent = history[-8:]
        ctx = "\n".join(
            f"{m['role']}: {str(m.get('content', ''))[:300]}" for m in recent
        )
        try:
            resp = ollama.chat(
                model=_SMALL_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            f"Overall goal: {task_summary}\n\n"
                            f"Work done:\n{ctx}\n\n"
                            "Was the overall goal fully achieved? Reply YES or NO only."
                        ),
                    }
                ],
                stream=False,
                options={"temperature": 0, "num_predict": 5},
            )
            ans = (
                re.sub(
                    r"<think>.*?</think>",
                    "",
                    resp.message.content or "",
                    flags=re.DOTALL,
                )
                .strip()
                .upper()
            )
            return ans.startswith("YES")
        except Exception:
            return True

    # ── Step 7: generate output summary ──────────────────────────────────────

    def _generate_output_summary(self, task_summary: str, history: List[dict]) -> str:
        """Ask main model for a short human-readable summary of what was done."""
        prompt = (
            f"Task completed: {task_summary}\n\n"
            "Write a SHORT plain-text summary of what was accomplished. "
            "MAX 4 lines. No markdown, no code blocks. "
            "Format: one sentence of what was done, then 'filename — what changed' "
            "for each file (if any), then any required user action on the last line."
        )
        try:
            resp = ollama.chat(
                model=_MODEL,
                messages=history[-6:] + [{"role": "user", "content": prompt}],
                stream=False,
                options={"temperature": 0},
            )
            return re.sub(
                r"<think>.*?</think>",
                "",
                resp.message.content or "",
                flags=re.DOTALL,
            ).strip()
        except Exception:
            return f"Completed: {task_summary}"

    # ── Private: stream one model turn ───────────────────────────────────────
    def _stream_one_turn(
        self,
        history: List[dict],
        system_prompt: str,
        think_acc_so_far: str,
        temperature: Optional[float] = None,
        inject_plan: bool = True,
        tools: Optional[List[dict]] = None,
    ) -> Tuple[str, str, List[Any], dict]:
        """
        Stream one ollama.chat call, parsing <think>...</think> tags manually.
        (think=True is not available in all SDK versions; tag parsing is reliable.)

        Returns:
            content          -- visible reply text accumulated this turn
            thinking_chunk   -- thinking text accumulated this turn
            tool_calls       -- deduplicated list of ToolCall objects (Ollama types)
            asst_msg         -- assembled assistant message dict for history
        """
        thinking_chunk = ""
        content = ""
        tool_calls_raw: List[Any] = []
        in_think = False

        _sys = system_prompt
        if inject_plan:
            with self._state.lock:
                _todo = list(self._state.todo_items)
            if _todo:
                next_idx = next((i for i, t in enumerate(_todo) if not t.done), None)
                plan_lines = []
                for i, item in enumerate(_todo, 1):
                    if item.done:
                        plan_lines.append(f"[x] {i}. {item.text}")
                    elif i - 1 == next_idx:
                        plan_lines.append(
                            f"[>] {i}. {item.text}  ← YOUR ONLY TASK RIGHT NOW"
                        )
                    else:
                        plan_lines.append(f"[ ] {i}. {item.text}  (locked)")
                current_task_text = (
                    _todo[next_idx].text if next_idx is not None else "all done"
                )
                _sys += (
                    "\n\n════════════════════════════════════════\n"
                    "PLAN — ONE TASK AT A TIME\n"
                    "════════════════════════════════════════\n"
                    + "\n".join(plan_lines)
                    + f"\n\nCURRENT TASK: {current_task_text}\n"
                    "Do ONLY this task. Do not start locked tasks."
                )
        _active_tools = tools if tools is not None else _TOOLS
        _opts = {"temperature": temperature} if temperature is not None else {}
        stream = ollama.chat(
            model=_MODEL,
            messages=[{"role": "system", "content": _sys}] + history,
            tools=_active_tools,
            stream=True,
            options=_opts if _opts else None,
        )

        for chunk in stream:
            if self._stopped():
                break

            msg = chunk.message
            piece = msg.content or ""

            if piece:
                # ── <think> open ──────────────────────────────────────────────
                if not in_think and "<think>" in piece:
                    in_think = True
                    with self._state.lock:
                        self._state.think_start = time.time()
                    piece = piece.split("<think>", 1)[1]

                # ── </think> close ─────────────────────────────────────────────
                if in_think and "</think>" in piece:
                    before, _, after = piece.partition("</think>")
                    thinking_chunk += before
                    in_think = False
                    with self._state.lock:
                        self._state.think_duration = (
                            time.time() - self._state.think_start
                        )
                        self._state.think_output = think_acc_so_far + thinking_chunk
                    piece = after  # remainder is normal content

                if in_think:
                    thinking_chunk += piece
                    with self._state.lock:
                        self._state.think_output = think_acc_so_far + thinking_chunk
                elif piece:
                    content += piece
                    with self._state.lock:
                        self._state.current_output = content
                        # First content token = final reply is streaming.
                        # Will be reset to False if tool calls are found.
                        self._state.reply_started = True

            # tool calls arrive as complete objects, not token-by-token
            if msg.tool_calls:
                tool_calls_raw.extend(msg.tool_calls)

        # Guard: model thought but never closed the tag (shouldn't happen normally)
        if in_think:
            with self._state.lock:
                self._state.think_duration = time.time() - self._state.think_start
                self._state.think_output = think_acc_so_far + thinking_chunk

        # Deduplicate tool calls — streaming can deliver the same call twice.
        # Use json.dumps(sort_keys=True) so dict key order never causes false misses.
        seen: set = set()
        unique: List[Any] = []
        for tc in tool_calls_raw:
            key = (tc.function.name, json.dumps(tc.function.arguments, sort_keys=True))
            if key not in seen:
                seen.add(key)
                unique.append(tc)

        # Build the history entry.
        # qwen3 was trained to see its own reasoning as <think>...</think> inside
        # the content field.  Storing it under a separate "thinking" key means the
        # model has no memory of what it already did when the next turn starts, so
        # it restarts from step 1.  Embedding it in content in the exact format the
        # model emitted preserves full context across tool-call turns.
        if thinking_chunk:
            history_content = f"<think>\n{thinking_chunk}\n</think>\n{content}"
        else:
            history_content = content
        asst_msg: dict = {"role": "assistant", "content": history_content}
        if unique:
            asst_msg["tool_calls"] = [
                {
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in unique
            ]

        return content, thinking_chunk, unique, asst_msg

    # ── Private: tool execution ───────────────────────────────────────────────

    def _execute_tool_calls(
        self,
        tool_calls: List[Any],
        history: List[dict],
        think_acc: str,
    ) -> str:
        """Execute a list of tool calls, append results to history, update UI.
        Returns the updated think_acc string."""
        for tc in tool_calls:
            fn = tc.function.name
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}

            before_lines = self._snapshot_lines(fn, args)
            event = self._push_tool_event(fn, args)

            pending_sentinel = f"\x00TOOL_PENDING\x00{event.label}…"
            think_acc += f"\n{pending_sentinel}"
            with self._state.lock:
                self._state.think_output = think_acc

            result = self._execute_tool(fn, args)

            stat = self._compute_stat(fn, args, before_lines)
            self._resolve_tool_event(event, result, stat)
            is_error = result.startswith("[ERROR]")
            done_sentinel = (
                f"\x00TOOL_ERROR\x00{event.label}"
                if is_error
                else f"\x00TOOL_DONE\x00{event.label}{stat}"
            )
            think_acc = think_acc.replace(pending_sentinel, done_sentinel) + "\n"
            with self._state.lock:
                self._state.think_output = think_acc

            history.append(
                {
                    "role": "tool",
                    "tool_name": fn,
                    "content": result,
                }
            )
        return think_acc

    def _execute_tool(self, name: str, args: dict) -> str:
        """Dispatch one tool call.  Path is resolved fresh per branch so that
        Pyright can narrow Path | None -> Path after each None-guard."""

        def _resolve(rel: str) -> "Optional[Path]":
            return _safe_path(rel, self._root) if rel else None

        if name == "read_file":
            rel = args.get("path", "")
            path = _resolve(rel)
            if path is None:
                return "[ERROR] Path is outside the project directory."
            start = int(args["start_line"]) if "start_line" in args else 1
            end = int(args["end_line"]) if "end_line" in args else None
            return _tool_read_file(path, rel, start, end)

        elif name == "search_file":
            rel = args.get("path", "")
            path = _resolve(rel)
            if path is None:
                return "[ERROR] Path is outside the project directory."
            pattern = args.get("pattern", "")
            if not pattern:
                return "[ERROR] No pattern provided."
            ctx = int(args["context_lines"]) if "context_lines" in args else 3
            return _tool_search_file(path, rel, pattern, ctx)

        elif name == "summarise_file":
            rel = args.get("path", "")
            path = _resolve(rel)
            if path is None:
                return "[ERROR] Path is outside the project directory."
            return _tool_summarise_file(path, rel)

        elif name == "write_file":
            rel = args.get("path", "")
            path = _resolve(rel)
            if path is None:
                return "[ERROR] Path is outside the project directory."
            if not str(path).startswith(str(self._root)):
                return "[ERROR] write_file is restricted to the project directory."
            return _tool_write_file(path, rel, args.get("content", ""))

        elif name == "create_file":
            rel = args.get("path", "")
            path = _resolve(rel)
            if path is None:
                return "[ERROR] Path is outside the project directory."
            if not str(path).startswith(str(self._root)):
                return "[ERROR] create_file is restricted to the project directory."
            return _tool_create_file(path, rel, args.get("content", ""))

        elif name == "delete_file":
            rel = args.get("path", "")
            path = _resolve(rel)
            if path is None:
                return "[ERROR] Path is outside the project directory."
            if not str(path).startswith(str(self._root)):
                return "[ERROR] delete_file is restricted to the project directory."
            if not self._confirm(f"Delete '{rel}'? [y/N]: "):
                return "CANCELLED: user declined deletion."
            return _tool_delete_file(path, rel)

        elif name == "run_command":
            cmd = args.get("command", "").strip()
            if not cmd:
                return "[ERROR] No command provided."
            # cwd is always the project root — the AI must not change it
            if not self._confirm(f"Run: {cmd}  [y/N]: "):
                return "CANCELLED: user declined command."
            return _tool_run_command(cmd, self._root)

        elif name == "web_search":
            query = args.get("query", "").strip()
            if not query:
                return "[ERROR] No query provided."
            return _tool_web_search(query)

        elif name == "consult_critic":
            plan = args.get("plan", "").strip()
            if not plan:
                return "[ERROR] No plan provided."
            return _tool_consult_critic(plan)

        return f"[ERROR] Unknown tool: {name}"

    # ── Private: tool state helpers ───────────────────────────────────────────

    def _push_tool_event(self, fn: str, args: dict) -> ToolEvent:
        rel = args.get("path", "") or args.get("query", "") or args.get("command", "")
        label = f"{self._DONE_VERB.get(fn, fn)} {rel}".strip()
        event = ToolEvent(name=fn, label=label, status=ToolStatus.PENDING)
        with self._state.lock:
            self._state.tool_events.append(event)
        return event

    def _resolve_tool_event(self, event: ToolEvent, result: str, stat: str) -> None:
        event.status = (
            ToolStatus.ERROR if result.startswith("[ERROR]") else ToolStatus.DONE
        )
        event.stat = stat

    def _snapshot_lines(self, fn: str, args: dict) -> Optional[int]:
        """Return the current line count of the target file BEFORE mutation."""
        if fn not in ("read_file", "write_file", "create_file", "delete_file"):
            return None
        path = _safe_path(args.get("path", ""), self._root)
        if path is None or not path.exists():
            return 0
        try:
            return len(path.read_text(encoding="utf-8").splitlines())
        except Exception:
            return 0

    def _compute_stat(self, fn: str, args: dict, before_lines: Optional[int]) -> str:
        """Compute diff stat using the pre-captured before_lines snapshot."""
        if before_lines is None:
            return ""
        path = _safe_path(args.get("path", ""), self._root)
        return _diff_stat(path, before_lines, fn)

    # ── Private: finalisation ─────────────────────────────────────────────────

    def _finalize(self, output: str) -> None:
        with self._state.lock:
            if output and not output.startswith("[ERROR]"):
                self._state.messages.append({"role": "assistant", "content": output})
            self._state.last_turn_tools = list(self._state.tool_events)
            self._state.current_output = ""
            self._state.think_output = ""
            self._state.think_label = "thinking…"
            self._state.scroll_offset = 0
            self._state.stop_requested = False
            self._state.model_busy = False
            messages_snapshot = list(self._state.messages)

        _notify_async(_MODEL, messages_snapshot)

    # ── Private: utilities ────────────────────────────────────────────────────

    def _stopped(self) -> bool:
        with self._state.lock:
            return self._state.stop_requested

    def _confirm(self, prompt: str) -> bool:
        """Post a ConfirmRequest to AppState and block until the main
        event loop resolves it.  The main thread owns the terminal, so
        all input handling happens there — no side-thread tty juggling."""
        req = ConfirmRequest(prompt=prompt)
        with self._state.lock:
            self._state.confirm_request = req
        req.event.wait()  # main loop sets req.result and signals this
        return bool(req.result)


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  Renderer
# ═══════════════════════════════════════════════════════════════════════════════


def _wrap_lines(text: str, width: int) -> List[str]:
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
    """Left-to-right shimmer sweep over *text*."""
    if not text:
        return ""
    total = len(text) + band
    pos = (tick * speed) % total
    out: List[str] = []
    prev = (-1, -1, -1)
    for i, ch in enumerate(text):
        dist = pos - i
        if 0 <= dist < band:
            t = 1.0 - abs(dist - band / 2) / (band / 2)
            r = g = b = int(100 + 95 * t)
        else:
            r = g = b = 100
        if (r, g, b) != prev:
            out.append(f"%color({r},{g},{b})")
            prev = (r, g, b)
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
}
_SH_KW["py"] = _SH_KW["python"]
_SH_KW["cpp"] = _SH_KW["c"]
_SH_KW["javascript"] = _SH_KW["ts"] = _SH_KW["typescript"] = _SH_KW["js"]
_SH_KW["bash"] = _SH_KW["zsh"] = _SH_KW["sh"]

_CH_DEFAULT = Theme.SYN_DEFAULT
_CH_KW = Theme.SYN_KEYWORD
_CH_STR = Theme.SYN_STRING
_CH_NUM = Theme.SYN_NUMBER
_CH_COMMENT = Theme.SYN_COMMENT
_CH_GUTTER = Theme.SYN_GUTTER


def _highlight_line(line: str, lang: str) -> str:
    kw = _SH_KW.get(lang.lower(), set())
    out: List[str] = [_CH_DEFAULT]
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch == "#" or line[i : i + 2] in ("//", "--"):
            out.append(_CH_COMMENT + line[i:])
            break
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
        if ch.isdigit():
            j = i + 1
            while j < n and (line[j].isdigit() or line[j] in ".xXabcdefABCDEF_"):
                j += 1
            out += [_CH_NUM, line[i:j], _CH_DEFAULT]
            i = j
            continue
        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (line[j].isalnum() or line[j] == "_"):
                j += 1
            word = line[i:j]
            out += [_CH_KW, word, _CH_DEFAULT] if word in kw else [word]
            i = j
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _split_code_blocks(content: str):
    fence, i = "```", 0
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


def _colour_stat(stat: str) -> str:
    """Colour +N/-N stat tokens for display."""
    if not stat.strip():
        return ""
    out: List[str] = []
    for tok in stat.split():
        if tok.startswith("+"):
            out.append(f" {Theme.DIFF_ADD}{tok}%reset")
        elif tok.startswith("-"):
            out.append(f" {Theme.DIFF_DEL}{tok}%reset")
        else:
            out.append(f" {Theme.DIFF_NEUTRAL}{tok}%reset")
    return "".join(out)


def _render_think_event(kind: str, text: str) -> str:
    """Format one line in the think block."""
    if kind == "pending":
        return f"{Theme.THINK_PENDING}◎%reset {Theme.THINK_PENDING_T}{text}"
    if kind == "error":
        return f"{Theme.THINK_ERROR}✘%reset {Theme.THINK_ERROR}{text}%reset"
    if kind == "done":
        m = re.search(r"(  [+\-\d L]+)$", text)
        if m:
            base, stat = text[: m.start()], m.group(1)
            return f"{Theme.THINK_DONE}✔%reset {Theme.THINK_DONE_T}{base}{_colour_stat(stat)}"
        return f"{Theme.THINK_DONE}✔%reset {Theme.THINK_DONE_T}{text}"
    return Theme.THINK_TEXT + text


def _parse_think_display_lines(
    think_text: str, wrap_width: int
) -> List[Tuple[str, str]]:
    """
    Convert raw think_output (with sentinel markers) into a list of
    (kind, text) tuples ready for rendering.
    kind ∈ {"text", "pending", "done", "error"}
    """
    clean = re.sub(r"</?think>", "", think_text)
    rows: List[Tuple[str, str]] = []
    for line in clean.split("\n"):
        if line.startswith("\x00TOOL_PENDING\x00"):
            rows.append(("pending", line[len("\x00TOOL_PENDING\x00") :]))
        elif line.startswith("\x00TOOL_DONE\x00"):
            rows.append(("done", line[len("\x00TOOL_DONE\x00") :]))
            rows.append(("text", ""))
        elif line.startswith("\x00TOOL_ERROR\x00"):
            rows.append(("error", line[len("\x00TOOL_ERROR\x00") :]))
            rows.append(("text", ""))
        else:
            for wrapped in (
                _wrap_lines(line.strip(), wrap_width) if line.strip() else [""]
            ):
                rows.append(("text", wrapped))
    # Drop leading blank lines
    while rows and rows[0] == ("text", ""):
        rows.pop(0)
    return rows


def render_frame(canvas: Canvas, state: AppState, tick: int) -> None:
    W = canvas.width
    H = canvas.height
    CONTENT_W = W - 4
    USER_ZONE_X = int(W * 0.4)
    USER_W = W - 3 - USER_ZONE_X
    THINK_ROWS = 3

    with state.lock:
        messages = list(state.messages)
        current = state.current_output
        think_text = state.think_output
        think_dur = state.think_duration
        think_start = state.think_start
        last_turn_tools = list(state.last_turn_tools)
        busy = state.model_busy
        scroll = state.scroll_offset
        think_mode = state.think_mode
        think_label = state.think_label
        think_scroll = state.think_scroll
        reply_started = state.reply_started
        todo_items = list(state.todo_items)
        show_plan_view = state.show_plan_view
        step_phase = state.step_phase
        replan_count = state.replan_count

    PLAN_ROWS = 1 if (todo_items or (busy and step_phase and step_phase != "planning")) else 0
    STATUS_ROW = H - 1 - PLAN_ROWS

    username = getpass.getuser()

    # ── Find last assistant message index ─────────────────────────────────
    last_asst_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"),
        default=-1,
    )

    # ── Build all chat lines ───────────────────────────────────────────────
    # Each entry: (x_offset, markup_text)
    all_lines: List[Tuple[int, str]] = []

    for idx, msg in enumerate(messages):
        is_user = msg["role"] == "user"

        # Header
        if is_user:
            hdr_x = max(W - 3 - len(username), USER_ZONE_X)
            all_lines.append((hdr_x, Theme.USER_HEADER + username))
        else:
            all_lines.append((1, Theme.ASST_HEADER + _MODEL))

        # "thought for Xs." + tool log (shown after the last assistant reply)
        if idx == last_asst_idx and think_dur > 0 and not busy:
            secs = f"{think_dur:.1f}"
            all_lines.append((1, f"{Theme.ASST_THOUGHT}│ thought for {secs}s."))
            for ev in last_turn_tools:
                sym = (
                    f"{Theme.THINK_DONE}✔%reset"
                    if ev.status == ToolStatus.DONE
                    else f"{Theme.THINK_ERROR}✘%reset"
                )
                coloured_stat = _colour_stat(ev.stat)
                all_lines.append(
                    (
                        1,
                        f"{Theme.ASST_THOUGHT}│%reset  {sym} {Theme.THINK_DONE_T}{ev.label}{coloured_stat}",
                    )
                )

        # Message body
        if is_user:
            for line in _wrap_lines(msg["content"], USER_W):
                line = line.strip()
                line_x = max(W - 3 - len(line), USER_ZONE_X)
                all_lines.append((line_x, Theme.USER_TEXT + line))
        else:
            for line in _message_lines(msg["content"], CONTENT_W):
                all_lines.append((1, line))

        all_lines.append((1, ""))  # blank separator

    # In-progress streaming reply.
    # reply_started is only True once we know there are no tool calls,
    # so the model name never appears during think blocks or tool execution.
    if current:
        if reply_started:
            all_lines.append((1, Theme.ASST_HEADER + _MODEL))
        for line in _message_lines(current, CONTENT_W):
            all_lines.append((1, line))

    # Auto-hide: only when reply_started, meaning the final reply is streaming.
    if reply_started and think_mode > 0:
        think_mode = 0

    # ── Pre-compute think block dimensions ────────────────────────────────
    _has_think = busy and bool(think_text)
    think_display_lines: List[Tuple[str, str]] = []
    think_rows_reserved = 0

    if _has_think:
        think_display_lines = _parse_think_display_lines(think_text, W - 6)
        if think_mode == 2:  # expanded
            if show_plan_view and todo_items:
                # Plan overlay occupies rows 0..(len(items)+5) inclusive:
                #   row 0          → top border  ╭─…─╮
                #   rows 1..n+4    → _ov_row calls (PLAN, sep, items, sep, "i·close")
                #   row n+5        → bottom border ╰─…─╯
                plan_ov_last_row = len(todo_items) + 5
                think_start_y    = plan_ov_last_row + 2
                think_rows_reserved = max(2, STATUS_ROW - think_start_y)
            else:
                think_rows_reserved = min(len(think_display_lines) + 1, H - 4)
        elif think_mode == 1:  # collapsed (3 lines + header)
            content_rows = min(THINK_ROWS, len(think_display_lines))
            think_rows_reserved = content_rows + 1 if content_rows else 1
        else:  # hidden (header only)
            think_rows_reserved = 1

    # ── Chat scrolling / layout ───────────────────────────────────────────
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
        if step_phase == "planning":
            label = "Planning…"
        else:
            label = (think_label[0].upper() + think_label[1:]) if think_label else "Thinking…"
        elapsed = f"({time.time() - think_start:.0f}s)" if think_start else ""
        left = f"{Theme.SPIN}{spin}%reset " + _shimmer_line(label, tick)
        if elapsed:
            ep_x = W - len(f" {elapsed} ") - 2
            canvas.text(ep_x, STATUS_ROW, f"%color(90,90,90) {elapsed} ")
        canvas.text(1, STATUS_ROW, left)
    elif scroll > 0:
        hint = "↑↓ scroll   ␣ type"
        canvas.text(max(0, (W - len(hint)) // 2), STATUS_ROW, Theme.HINT_SCROLL + hint)
    else:
        hint = "↑↓ scroll   ␣ type"
        canvas.text(max(0, (W - len(hint)) // 2), STATUS_ROW, Theme.HINT + hint)

    # ── Plan strip ─────────────────────────────────────────────────────────
    plan_row = STATUS_ROW + 1
    if PLAN_ROWS:
        if todo_items:
            done_count = sum(1 for t in todo_items if t.done)
            total_count = len(todo_items)
            active = next((t for t in todo_items if not t.done), None)
            active_text = active.text if active else "all done"
            replan_tag = f" [replan {replan_count}]" if replan_count else ""
            phase_tag = f" · {step_phase}" if step_phase else ""
            plan_str = (
                f"{done_count}/{total_count}{replan_tag}{phase_tag}: {active_text}"
            )
            if len(plan_str) > canvas.width - 7:
                plan_str = plan_str[: canvas.width - 8] + "…"

            capped_len = min(len(plan_str), canvas.width - 6)

            canvas.text(1, plan_row, f"%color(80,80,80)╰ {plan_str[:capped_len]} ")
        elif busy and step_phase:
            capped_len = min(len(step_phase), canvas.width - 6)
            canvas.text(1, plan_row, f"%color(55,55,80)╰ {step_phase[:capped_len]} ")

    # ── Plan view overlay (i key) — anchored top-left ─────────────────────
    if show_plan_view and todo_items:
        ov_w = canvas.width - 2  # full canvas width
        inner_w = ov_w - 6  # text width between "│ " and " │"
        border = "%color(55,55,80)"
        done_c = "%color(70,160,100)"
        todo_c = "%color(160,160,160)"
        cur_c = "%color(210,180,80)"
        head_c = "%color(140,140,200)"
        sep_c = "%color(55,55,80)"
        next_i = next((i for i, t in enumerate(todo_items) if not t.done), None)

        _ov_row_y = [1]  # mutable so the nested fn can increment it

        def _ov_row(text: str, colour: str = "%reset") -> None:
            clipped = text[:inner_w - 1] + "…" if len(text) > inner_w else text
            padded = clipped.ljust(inner_w)
            canvas.text(
                1,
                _ov_row_y[0],
                border + "│ " + colour + padded + "%reset" + border + " │",
            )
            _ov_row_y[0] += 1

        canvas.text(1, 0, border + "╭" + "─" * (ov_w - 4) + "╮")
        _ov_row("PLAN", head_c)
        _ov_row("─" * inner_w, sep_c)
        for idx, item in enumerate(todo_items, 1):
            if item.done:
                mark, colour = "x", done_c
            elif idx - 1 == next_i:
                mark, colour = ">", cur_c
            else:
                mark, colour = " ", todo_c
            _ov_row(f"[{mark}] {idx}. {item.text}", colour)
        _ov_row("─" * inner_w, sep_c)
        _ov_row("i · close", "%color(70,70,90)")
        canvas.text(0, _ov_row_y[0], border + "╰" + "─" * (ov_w - 3) + "╯")

    # ── Think block ────────────────────────────────────────────────────────
    if _has_think:
        block_start = STATUS_ROW - think_rows_reserved
        if think_mode == 0:
            # Hidden: just the header tooltip, no content rows
            canvas.text(
                1,
                block_start,
                Theme.THINK_GUTTER + "╭─ " + Theme.THINK_HEADER + "o · show thinking",
            )
        elif think_mode == 1:
            # Collapsed: header + last 3 lines of thought
            canvas.text(
                1,
                block_start,
                Theme.THINK_GUTTER + "╭─ " + Theme.THINK_HEADER + "o · expand thinking",
            )
            content_rows = think_rows_reserved - 1
            pool = think_display_lines[-content_rows:] if content_rows else []
            for i, (kind, text) in enumerate(pool):
                canvas.text(
                    1,
                    block_start + 1 + i,
                    Theme.THINK_GUTTER + "│%reset " + _render_think_event(kind, text),
                )
        else:
            # Expanded: header + scrollable full content
            canvas.text(
                1,
                block_start,
                Theme.THINK_GUTTER
                + "╭─ "
                + Theme.THINK_HEADER
                + "o · collapse thinking",
            )
            display_rows = think_rows_reserved - 1
            total_think = len(think_display_lines)
            max_tscroll = max(0, total_think - display_rows)
            t_scroll = min(think_scroll, max_tscroll)
            end_idx = total_think - t_scroll
            start_idx = max(0, end_idx - display_rows)
            pool = think_display_lines[start_idx:end_idx]
            while pool and pool[0] == ("text", ""):
                pool = pool[1:]
            # Pad to display_rows so the gutter is always fully drawn.
            # Without this, rows below the content are blank and the box
            # appears to jump/shrink when pool is shorter than the reserved space.
            while len(pool) < display_rows:
                pool.append(("text", ""))
            for i, (kind, text) in enumerate(pool):
                canvas.text(
                    1,
                    block_start + 1 + i,
                    Theme.THINK_GUTTER + "│%reset " + _render_think_event(kind, text),
                )


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  Filesystem helpers + system prompt
# ═══════════════════════════════════════════════════════════════════════════════


def _load_ignore_patterns(root: Path) -> List[str]:
    patterns = [
        ".git",
        "__pycache__",
        "*.pyc",
        ".DS_Store",
        ".vscode",
        "node_modules",
        "*.egg-info",
    ]
    for f in (".gitignore", ".ignore"):
        ignore_file = root / f
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

    return f"""\
You are a senior software engineer working inside a real project on the user's machine.
You have direct access to the filesystem and shell. USE YOUR TOOLS — do not guess, read.

════════════════════════════════════════
PROJECT
════════════════════════════════════════
Working directory: {path}

File tree:
```
{tree.rstrip()}
```

════════════════════════════════════════
YOUR TOOLS
════════════════════════════════════════
read_file(path, [start_line], [end_line])
                              — Read a file.
                                Small files (≤ ~16k tokens): returned in full with line numbers.
                                Large files: automatically split and condensed to signatures,
                                class/function names, and docstrings — no implementation bodies.
                                Use start_line/end_line to read the exact bytes of any region,
                                or search_file to locate the right lines first.
summarise_file(path)          — Ask the model to read and summarise an entire file in
                                one shot. Use when you need to understand a file without
                                quoting or editing it. Cheaper on context than read_file.
search_file(path, pattern, [context_lines])
                              — Grep-style search inside a file. Returns matching lines
                                with line numbers and context. Use BEFORE read_file on
                                large files to find the region you need.
write_file(path, content)     — Overwrite a file. MUST contain the COMPLETE file content.
create_file(path, content)    — Create a new file (fails if it already exists).
delete_file(path)             — Delete a file (requires user confirmation).
run_command(command)          — Run a shell command (requires user confirmation).
                                Use for: tests, linting, builds, installs, git, etc.
web_search(query)             — Search the web for docs, APIs, error messages, or facts.
consult_critic(plan)          — Adversarial review of your plan or code before committing.

════════════════════════════════════════
HOW TO WORK (inside your thinking)
════════════════════════════════════════
1. Understand the task in one sentence.
2. read_file behaviour:
   — Small files (≤ ~16k tokens): returned in full. No need to paginate.
   — Large files: returned as a skeleton (signatures + docstrings only).
     Use search_file to locate a specific region, then read_file with
     start_line/end_line to get the exact implementation lines you need.
   — To understand a large file without editing: use summarise_file.
   — When you need to edit a large file, always read the relevant section
     with start_line/end_line before writing — never guess at the content.
3. For external information (APIs, error codes, versions), call web_search.
4. Form a concrete plan. Then execute it immediately — do not describe it.
5. For non-trivial code, call consult_critic before the final write.
6. Write complete files with write_file / create_file.
7. Verify: run tests with run_command if a test suite exists.
8. If something is wrong, fix it before replying.

HARD RULES — violations waste time and produce wrong answers:
— NEVER read the same file twice. You already have its contents.
— NEVER write partial file content. Every write_file must contain the entire file.
— NEVER invent what a file contains. Read it first.
— NEVER re-examine a decision you have already made. Move forward.
— If you find yourself writing "I should..." or "I will..." — stop and do it instead.
— NEVER list, enumerate, or guess values from memory (imports, functions, symbols,
  dependencies, etc.). If you need to know what is in a file, READ IT. Any list you
  produce from memory will be wrong or incomplete. Use search_file or read_file and
  work from the actual content, not from what you expect the file to contain.

THINKING DISCIPLINE — this is critical:
— Think through each step ONCE. Do not loop back and re-check conclusions.
— After a tool call succeeds, read the result and continue to the NEXT step.
  Do not re-analyse what the tool just did — the result is the confirmation.
— Do not re-read files you already read. The content is in your context.
— Do not re-derive code you already wrote. The write receipt confirms it was saved.
— If you catch yourself re-examining something already resolved, STOP and move on.

════════════════════════════════════════
YOUR REPLY  (after all work is done)
════════════════════════════════════════
Write a SHORT plain-text summary. NO code, NO markdown, MAXIMUM 4 lines.

Allowed:
  1. One sentence describing what was done.
  2. Files changed, one per line: "filename — what changed"
  3. One line for any required user action (e.g. "Run: pip install httpx").

If the user wants to see the code they will open the file.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# § 10  Notification helper
# ═══════════════════════════════════════════════════════════════════════════════


def _notify_async(title: str, messages: List[dict]) -> None:
    """Generate a one-sentence summary and send a desktop notification."""

    def _run() -> None:
        try:
            resp = ollama.chat(
                _SMALL_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Summarise the assistant's last reply in one short sentence "
                            "(under 80 chars). Reply with ONLY the summary — "
                            "no preamble, no thinking, no punctuation at the end."
                        ),
                    },
                    *messages[-6:],
                ],
                stream=False,
                options={"num_predict": 60, "temperature": 0},
            )
            summary = (resp.message.content or "").strip()
            summary = re.sub(
                r"<think>.*?</think>", "", summary, flags=re.DOTALL
            ).strip()
            summary = summary[:80]
        except Exception:
            summary = "Response received."
        notify_fn = getattr(notification, "notify", None)
        if callable(notify_fn):
            notify_fn(title=title, message=summary, timeout=5)

    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
# § 11  main()
# ═══════════════════════════════════════════════════════════════════════════════


def main(argv: Tuple[str, ...]) -> None:
    if len(argv) < 2:
        print("Usage: python main.py <directory_path>")
        return

    # Ensure the Ollama daemon is up
    try:
        ollama.list()
    except Exception:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(20):
            time.sleep(0.5)
            try:
                ollama.list()
                break
            except Exception:
                pass

    # Keep display awake on macOS
    if platform.system() == "Darwin":
        subprocess.Popen(
            ["caffeinate", "-i", "-w", str(os.getpid())],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    root = Path(argv[1]).resolve()
    state = AppState()
    app = Application()

    tw = os.get_terminal_size().columns
    th = os.get_terminal_size().lines
    canvas = Canvas(tw, th - 8)
    worker = AgentWorker(state, canvas, root, app)

    tick = 0
    _SCROLL_STEP = 3
    _fd = sys.stdin.fileno()
    _cooked = termios.tcgetattr(_fd)  # saved for read_line restore

    def _key_available() -> bool:
        return bool(select.select([_fd], [], [], 0)[0])

    def _read_key() -> str:
        ch = os.read(_fd, 1)
        if ch == b"\x1b" and select.select([_fd], [], [], 0.05)[0]:
            ch += os.read(_fd, 8)
        return ch.decode("utf-8", errors="replace")

    def _enter_cbreak() -> None:
        tty.setcbreak(_fd, termios.TCSANOW)

    def _leave_cbreak() -> None:
        termios.tcsetattr(_fd, termios.TCSANOW, _cooked)

    _enter_cbreak()

    try:
        while state.running:
            # ── Resize ────────────────────────────────────────────────────
            size = os.get_terminal_size()
            if size.columns != tw or size.lines != th:
                tw, th = size.columns, size.lines
                canvas.resize(tw, th - 8)

            # ── Render ────────────────────────────────────────────────────
            render_frame(canvas, state, tick)
            canvas.draw()
            tick += 1

            # ── Confirm request from worker thread ────────────────────────
            with state.lock:
                _creq = state.confirm_request
            if _creq is not None and _creq.result is None:
                _leave_cbreak()
                _answer = (canvas.read_line(_creq.prompt) or "").strip().lower()
                _enter_cbreak()
                _creq.result = _answer in ("y", "yes")
                with state.lock:
                    state.confirm_request = None
                _creq.event.set()

            # ── Input ─────────────────────────────────────────────────────
            while _key_available():
                key = _read_key()

                # ESC — abort generation or quit
                if key == "\x1b":
                    if state.model_busy:
                        with state.lock:
                            state.stop_requested = True
                    else:
                        state.running = False
                    break

                # o — toggle think box  (works while busy)
                if key == "o":
                    with state.lock:
                        state.think_mode = (state.think_mode + 1) % 3
                        state.think_scroll = 0
                    continue

                # i — toggle plan view overlay  (works while busy)
                if key == "i":
                    with state.lock:
                        state.show_plan_view = not state.show_plan_view
                    continue

                # Scroll keys while model is busy: only inside expanded think box
                if state.model_busy:
                    if key in ("\x1b[A", "\x1b[B") and state.think_mode == 2:
                        with state.lock:
                            if key == "\x1b[A":
                                state.think_scroll += _SCROLL_STEP
                            else:
                                state.think_scroll = max(
                                    0, state.think_scroll - _SCROLL_STEP
                                )
                    continue  # swallow everything else while busy

                # Arrow keys — chat / think scroll
                if key == "\x1b[A":
                    with state.lock:
                        if state.think_mode == 2:
                            state.think_scroll += _SCROLL_STEP
                        else:
                            state.scroll_offset += _SCROLL_STEP
                elif key == "\x1b[B":
                    with state.lock:
                        if state.think_mode == 2:
                            state.think_scroll = max(
                                0, state.think_scroll - _SCROLL_STEP
                            )
                        else:
                            state.scroll_offset = max(
                                0, state.scroll_offset - _SCROLL_STEP
                            )

                # Space — enter prompt mode
                elif key == " ":
                    _leave_cbreak()
                    raw_input = canvas.read_line("> ") or ""
                    _enter_cbreak()

                    with state.lock:
                        state.scroll_offset = 0

                    if "\x1b" in raw_input:
                        continue  # cancelled

                    user_input = raw_input.strip()
                    if not user_input:
                        continue

                    if user_input.lower() in ("bye", "quit", "exit"):
                        state.running = False
                        break

                    with state.lock:
                        state.messages.append({"role": "user", "content": user_input})
                        messages_snapshot = list(state.messages)
                        state.model_busy = True
                        state.current_output = ""
                        state.think_duration = 0.0

                    system_prompt = generate_system_prompt(str(root))
                    worker.start(messages_snapshot, system_prompt, user_input)
                    _start_think_labeller(state)

            time.sleep(0.016)

    finally:
        _leave_cbreak()
        canvas.reset_console()


if __name__ == "__main__":
    main(tuple(sys.argv))