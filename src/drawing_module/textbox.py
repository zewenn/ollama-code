from typing import Optional, List
from .symbol import Symbol
from .alignment import Alignment

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
