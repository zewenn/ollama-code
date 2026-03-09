# 1.  (caller)  User enters a prompt
# 2.  Build a system prompt that includes the current file layout
# 3.  Ask the model to create a step-by-step plan          (temperature=0.5)
# 4.  Summarise the user prompt into one tight sentence    (temperature=0)
# 5.  For each step
#     5.1  Collect context: user request + past tool calls + thinking summaries
#     5.2  Think + call tools until the step is resolved
#     5.3  Ask the model whether the step goal was achieved; retry if not
#     5.4  Log tool usage and store a thinking summary
#     5.5  Advance to the next step
# 6.  Ask the model whether the overall goal was achieved; replan + restart if not
# 7.  Ask the model for a human-readable output summary
# 8.  Persist user request / tool usage / summary to Application.history


import ollama
import threading
from enum import Enum, auto
from typing import Optional, List
from dataclasses import dataclass


class Role(Enum):
    system = auto()
    assistant = auto()
    user = auto()
    tool = auto()


class Status(Enum):
    todo = auto()
    in_progress = auto()
    done = auto()


class Message:
    role: Role
    content: str

    def __init__(self, role: Role, content: str) -> None:
        self.role = role
        self.content = content

    def to_ollama_message(self) -> dict[str, str]:
        return {
            "role": self.role.name,
            "content": self.content,
        }


@dataclass
class Step:
    content: str
    status: Status


class OllamaWorker:
    MODEL = "qwen3:8b"
    AGENDA_GENERATE_PROMPT = Message(
        Role.user,
        "\n".join(
            [
                "You are a task planner for a coding agent."
                f'Overall task: "%user_prompt%"\n'
                "Break this into sequential, atomic steps. Each step must:"
                "— Be a single concrete action (write_file, create_file, or run_command)."
                "— Name the exact file and describe exactly what it should contain."
                "— NOT include reading, searching, or planning sub-steps."
                "\n"
                "Reply ONLY with a numbered list. No preamble, no commentary.\n\nSteps:"
            ]
        ),
    )

    history: List[Message] = []

    steps: List[Step] = []
    current_step: int = 0

    user_prompt: Optional[str] = None
    lock: threading.Lock

    def __init__(self) -> None:
        self.lock = threading.Lock()

    def start(self, msg: str) -> None:
        self.user_prompt = msg

        new_agenda_prompt = Message(
            self.AGENDA_GENERATE_PROMPT.role,
            self.AGENDA_GENERATE_PROMPT.content.replace(
                "%user_prompt%", self.user_prompt
            ),
        )

        response = ollama.chat(
            self.MODEL,
            [new_agenda_prompt.to_ollama_message()],
            think=False,
            stream=True,
        )
        
        for part in response:
            print(part.message.content, end="", flush=True)
