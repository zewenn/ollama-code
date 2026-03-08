import platform
import sys
import os
from typing import List, Tuple, Optional
from .symbol import Symbol
from .textbox import TextBox
from .alignment import Alignment

_BORDER_COLOR = "\u001b[38;2;55;55;65m"
_BORDER_RESET = "\u001b[0m"


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
            current_terminal_width, current_terminal_height = self._get_terminal_size()
            width = width if width is not None else current_terminal_width
            height = height if height is not None else current_terminal_height

        self.reset_console()
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

        current_terminal_width, current_terminal_height = self._get_terminal_size()
        self._terminal_width = current_terminal_width
        self._terminal_height = current_terminal_height

        self.hide_cursor()

        self.width = width
        self.height = height

        self._position_x = max(1, current_terminal_width // 2 - self.width // 2)
        self._cache_buffer = Symbol.buffer(self.width, self.height)
        self._frame_buffer = Symbol.buffer(self.width, self.height)

        _B = _BORDER_COLOR
        _R = _BORDER_RESET

        border_buffer: List[str] = []
        for y in range(-1, self.height + 1):
            for x in range(-1, self.width + 1):
                col = self._position_x + x
                row = self._position_y + y

                border_buffer.append(f"\u001b[{row + 1};{col + 1}H")
                if y == -1:
                    border_buffer.append(
                        _B
                        + ("╭" if x == -1 else ("╮" if x == self.width else "─"))
                        + _R
                    )
                elif y == self.height:
                    border_buffer.append(
                        _B
                        + ("╰" if x == -1 else ("╯" if x == self.width else "─"))
                        + _R
                    )
                elif x == -1 or x == self.width:
                    border_buffer.append(_B + "│" + _R)
                else:
                    border_buffer.append(" ")

        sys.stdout.write("".join(border_buffer))
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

            current_x = x
            x += 1
            if start_x + current_x < 0 or y < 0:
                continue
            if start_x + current_x >= self.width and not overflow:
                continue

            self._frame_buffer[y][start_x + current_x] = symbol

    def text_box(self, start_x: int, start_y: int, text_box: TextBox) -> None:
        for text_box_y in range(text_box.height):
            y = start_y + text_box_y

            if y >= self.height:
                break

            for text_box_x in range(text_box.width):
                x = start_x + text_box_x

                if x >= self.width:
                    break

                s = text_box.get(text_box_x, text_box_y)
                if s.content not in ["\t", "\0"]:
                    self._frame_buffer[y][x] = s

    def window(
        self,
        pos: Tuple[int, int],
        size: Tuple[int, int],
        title: str,
        content: str,
        footer: str,
    ) -> None:
        content_height: int = size[1] - 6
        content_width: int = size[0] - 2

        content_box = TextBox(
            content_width,
            content_height,
            content,
            Alignment.min,
            Alignment.min,
        )

        top_line = "╭" + ("─" * content_width) + "╮"
        separator = "├" + ("─" * content_width) + "┤"
        bottom_line = "╰" + ("─" * content_width) + "╯"

        title_len = min(content_width, len(title))
        footer_len = min(content_width, len(footer))

        title_line = "│" + title[:title_len] + (" " * (content_width - title_len)) + "│"
        footer_line = (
            "│" + footer[:footer_len] + (" " * (content_width - footer_len)) + "│"
        )

        content_lines = ["│" + (" " * content_width) + "│"] * content_height

        window_content = "\n".join(
            [top_line, title_line, separator]
            + content_lines
            + [separator, footer_line, bottom_line]
        )

        window_box = TextBox(
            size[0],
            size[1],
            window_content,
            Alignment.min,
            Alignment.min,
        )

        self.text_box(pos[0], pos[1], window_box)
        self.text_box(pos[0] + 1, pos[1] + 3, content_box)

    def draw(self) -> None:
        current_terminal_width, current_terminal_height = self._get_terminal_size()
        if (
            current_terminal_width != self._terminal_width
            or current_terminal_height != self._terminal_height
        ):
            self._terminal_width = current_terminal_width
            self._terminal_height = current_terminal_height

            self.resize(self.width, self.height)

            self._cache_buffer = Symbol.buffer(self.width, self.height)

        if self.height > self._terminal_height or self.width > self._terminal_width:
            midtext = "Please resize the console!"
            subtext = f"(Minimum {self.width}x{self.height})"

            middle_x = min(self._terminal_width, self.width) // 2
            middle_y = min(self._terminal_height, self.height) // 2

            self.set_cursor_position(middle_x - len(midtext) // 2, middle_y)
            sys.stdout.write(midtext)

            self.set_cursor_position(middle_x - len(subtext) // 2, middle_y + 1)
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
        self._frame_buffer = Symbol.null_buffer(self.width, self.height)

    def move_cursor_to_bottom(self, x: int = 0) -> None:
        self.set_cursor_position(self._position_x + x, self.height + 2)

    def clear_line(self, y: int = -1) -> None:
        if y < 0:
            y = self.height + 4

        self.set_cursor_position(self._position_x, y)
        sys.stdout.write("\u001b[0m" + " " * self.width)
        self.set_cursor_position(self._position_x, y)
        sys.stdout.flush()

    def read_line(self, prompt: str, default: Optional[str] = None) -> Optional[str]:
        """Block and read a line of user input from below the canvas."""

        input_row = self.height + 3

        self.clear_line(input_row)
        self.set_cursor_position(self._position_x, input_row)
        self.show_cursor()

        sys.stdout.write(prompt)
        sys.stdout.flush()

        result = default

        try:
            result = input()
        except Exception:
            pass  # TODO: handle error case, but shouldn't happen

        self.set_cursor_position(self._position_x, input_row)

        sys.stdout.write("\u001b[J")
        sys.stdout.flush()

        self.hide_cursor()

        return result
