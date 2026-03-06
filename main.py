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
from typing import List, Optional, Tuple

import ollama
from plyer import notification
import select
import tty


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

        _B = "\u001b[38;2;55;55;65m"  # muted blue-gray
        _R = "\u001b[0m"
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

    pending_updates: dict = field(default_factory=dict)
    """File paths → content proposed by the model via %update% blocks."""

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


class OllamaWorker:
    """Streams an Ollama chat response on a background daemon thread,
    writing tokens into AppState.current_output as they arrive."""

    def __init__(self, state: AppState) -> None:
        self._state = state

    def start(self, messages: List[dict], system_prompt: str) -> None:
        """Spawn a new daemon thread for the current request."""
        threading.Thread(
            target=self._run,
            args=(list(messages), system_prompt),
            daemon=True,
        ).start()

    def _run(self, messages: List[dict], system_prompt: str) -> None:
        output = ""
        think_text = ""
        thinking = False

        try:
            response = ollama.chat(
                _MODEL,
                [{"role": "system", "content": system_prompt}] + messages,
                stream=True,
            )

            for token in response:
                with self._state.lock:
                    if self._state.stop_requested:
                        break
                content: str = token.message.content or ""
                if not content:
                    continue

                if "<think>" in content:
                    thinking = True
                    with self._state.lock:
                        self._state.think_start_time = time.time()
                if "</think>" in content:
                    thinking = False
                    with self._state.lock:
                        self._state.think_duration = (
                            time.time() - self._state.think_start_time
                        )
                    continue

                if thinking:
                    # Accumulate think text and push last 3 lines to state
                    think_text += content
                    with self._state.lock:
                        self._state.think_output = think_text
                    continue

                output += content
                with self._state.lock:
                    self._state.current_output = output

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
                self._state.pending_updates = extract_files(output)
                self._state.current_output = ""
                self._state.think_output = ""
                self._state.scroll_offset = 0
                self._state.stop_requested = False
                # think_duration is intentionally kept so the renderer can show "thought for X s"
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

_CH_DEFAULT = "%reset%color(185,185,185)"
_CH_KW = "%reset%color(140,120,210)"
_CH_STR = "%reset%color(180,160,80)"
_CH_NUM = "%reset%color(210,130,80)"
_CH_COMMENT = "%reset%color(105,120,85)"
_CH_GUTTER = "%color(55,55,70)"


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
        busy = state.model_busy
        scroll = state.scroll_offset
        show_thinking = state.show_thinking

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
            all_lines.append((hdr_x, "%color(70,70,70)" + username))
        else:
            all_lines.append((1, "%color(70,70,70)" + _MODEL))

        # "thought for X s." before the last assistant message
        if idx == last_asst_idx and think_dur > 0 and not busy:
            secs = f"{think_dur:.1f}"
            all_lines.append((1, f"%color(80,80,80)│ thought for {secs}s."))

        if is_user:
            content_lines = _wrap_lines(msg["content"], USER_W)
            for line in content_lines:
                line = line.strip()
                line_x = max(W - 3 - len(line), USER_ZONE_START)
                all_lines.append((line_x, "%color(120,180,255)" + line))
        else:
            for line in _message_lines(msg["content"], CONTENT_W):
                all_lines.append((1, line))

        all_lines.append((1, ""))  # blank line between turns

    # In-progress assistant reply (streaming)
    if current:
        all_lines.append((1, "%color(70,70,70)" + _MODEL))
        for line in _message_lines(current, CONTENT_W):
            all_lines.append((1, line))

    # ── Pre-compute think block size so chat layout can avoid it ─────────
    think_rows_reserved = 0
    think_lines_expanded: List[str] = []

    if busy and think_text:
        clean = re.sub(r"</?think>", "", think_text)
        if show_thinking:
            raw_lines: List[str] = []
            for raw in clean.split("\n"):
                for wrapped in _wrap_lines(raw.strip(), W - 6) if raw.strip() else [""]:
                    raw_lines.append(wrapped)
            # Drop leading blank lines so no gap appears between label and content
            while raw_lines and raw_lines[0].strip() == "":
                raw_lines.pop(0)
            # +1 for the label line at the top of the block
            think_rows_reserved = min(len(raw_lines) + 1, H - 4)
            think_lines_expanded = raw_lines
        else:
            think_rows_reserved = THINK_ROWS

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
        status = f"%color(210,120,40){spin}%reset " + _shimmer_line("thinking…", tick)
        canvas.text(1, STATUS_ROW, status)
    elif scroll > 0:
        indicator = "↑↓ scroll   ␣ type"
        ind_x = max(0, (W - len(indicator)) // 2)
        canvas.text(ind_x, STATUS_ROW, "%color(70,70,90)" + indicator)
    else:
        hint = "↑↓ scroll   ␣ type"
        hint_x = max(0, (W - len(hint)) // 2)
        canvas.text(hint_x, STATUS_ROW, "%color(65,65,65)" + hint)

    # ── Think block rendering ──────────────────────────────────────────────
    if busy and think_text:
        clean = re.sub(r"</?think>", "", think_text)
        if show_thinking:
            # Label occupies the first row of the reserved block, lines fill the rest
            display_rows = think_rows_reserved - 1  # rows available for content
            content = think_lines_expanded[-display_rows:] if display_rows > 0 else []
            # Strip any leading blank lines from what we're about to render
            while content and content[0].strip() == "":
                content = content[1:]
            block_start = STATUS_ROW - think_rows_reserved

            # Label row — no gap between label and lines
            canvas.text(1, block_start, "%color(55,55,75)  o · collapse thinking")

            for i, tl in enumerate(content):
                canvas.text(
                    1,
                    block_start + 1 + i,
                    "%color(80,80,80)│%reset %color(100,100,100)" + tl.strip(),
                )
        else:
            flat = " ".join(clean.split())
            think_display = _wrap_lines(flat, W - 6)[-THINK_ROWS:]
            base_row = STATUS_ROW - 1
            n = len(think_display)
            for i, tl in enumerate(think_display):
                row = base_row - (n - 1 - i)
                canvas.text(
                    1, row, "%color(80,80,80)│%reset %color(100,100,100)" + tl.strip()
                )


# ═══════════════════════════════════════════════════════════════════════════════
# Filesystem / prompt helpers
# ═══════════════════════════════════════════════════════════════════════════════


def extract_files(markdown: str) -> dict:
    pattern = r"%update%\s+(\S+)\s*\n```[^\n]*\n(.*?)```"
    return {path: content for path, content in re.findall(pattern, markdown, re.DOTALL)}


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


def _gather_file_contents(root: Path, patterns: List[str]) -> str:
    parts = ["## File Contents:"]
    for fp in root.rglob("*"):
        if fp.is_file() and not _is_ignored(fp, root, patterns):
            try:
                content = fp.read_text(encoding="utf-8")
                parts.append(f"Contents of `{fp.relative_to(root)}`:")
                parts.append(f"```\n{content.strip()}\n```\n")
            except (UnicodeDecodeError, PermissionError):
                continue
    return "\n".join(parts)


def generate_system_prompt(cwd: str) -> str:
    path = Path(cwd).resolve()
    patterns = _load_ignore_patterns(path)

    prompt = "# Project Information\n"
    prompt += "> Refer to the following when answering the user's prompt.\n\n"
    prompt += "## Directory File Structure:\n```\n"
    prompt += _generate_tree(path, path, patterns)
    prompt += "```\n\n"
    prompt += _gather_file_contents(path, patterns)
    prompt += "\n# Guidelines\n"
    prompt += "- Look up language specs for the correct version when writing code.\n"
    prompt += "- Triple-check gathered information before responding.\n"
    prompt += "- Follow the style of the user's existing code.\n"
    prompt += "- Cite sources and be accurate.\n"
    prompt += "\n## File Changes\n"
    prompt += (
        "To modify a file use **exactly** this syntax (full file contents only):\n"
    )
    prompt += "%update% <filepath>\n```<code>```\n"
    prompt += "Always rewrite the complete file — no partial changes.\n"
    prompt += "\n## Response Style\n"
    prompt += "Keep replies concise. Only point out non-obvious issues.\n"
    prompt += "Do not overthink simple tasks.\n"
    return prompt


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

    target_path = argv[1]
    state = AppState()
    worker = OllamaWorker(state)

    tw = os.get_terminal_size().columns
    th = os.get_terminal_size().lines
    canvas = Canvas(tw, th - 8)

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
                    continue

                if state.model_busy:
                    continue  # swallow all other keys while model is running

                if key == "\x1b[A":  # ↑
                    with state.lock:
                        state.scroll_offset += _SCROLL_STEP
                elif key == "\x1b[B":  # ↓
                    with state.lock:
                        state.scroll_offset = max(0, state.scroll_offset - _SCROLL_STEP)
                elif key == " ":  # space → enter prompt mode
                    # ── Apply any pending file updates ─────────────────────
                    with state.lock:
                        pending = dict(state.pending_updates)
                        state.pending_updates.clear()

                    for rel_path, content in pending.items():
                        abs_path = os.path.join(os.path.realpath(target_path), rel_path)
                        if not os.path.exists(abs_path):
                            canvas.text(
                                0,
                                canvas.height - 2,
                                f"%color(255,100,100)Path not found: {abs_path}",
                            )
                            canvas.draw()
                            time.sleep(1.5)
                            continue
                        _leave_cbreak()
                        answer = (
                            canvas.read_line(f"Write to ./{rel_path}? [Y/n]: ") or ""
                        ).strip()
                        _enter_cbreak()
                        if answer.lower() not in ("n", "no"):
                            with open(abs_path, "w", encoding="utf-8") as fh:
                                fh.write(content)

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

                    system_prompt = generate_system_prompt(target_path)
                    worker.start(messages_snapshot, system_prompt)

            time.sleep(0.016)

    finally:
        _leave_cbreak()
        canvas.reset_console()


if __name__ == "__main__":
    main(tuple(sys.argv))
