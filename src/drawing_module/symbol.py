from typing import List


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
    def from_string(text: str) -> List["Symbol"]:
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
    def buffer(width: int, height: int) -> List[List["Symbol"]]:
        return [[Symbol(" ") for _ in range(width)] for _ in range(height)]

    @staticmethod
    def null_buffer(width: int, height: int) -> List[List["Symbol"]]:
        return [[Symbol("\0") for _ in range(width)] for _ in range(height)]
