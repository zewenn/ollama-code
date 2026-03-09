# 1.    (caller)  User enters a prompt
# 2.    Build a system prompt that includes the current file layout
# 3.    Ask the model to create a step-by-step plan, 
#       make sure to split the plan into simple tasks          
#       (temperature=0.5)
# 4.    Summarise the user prompt into one tight sentence    (temperature=0)
# 5.    For each step
#       5.1     Collect context: current task + past tool calls + previous output
#               Note: the model should only recieve these, not the user prompt, and not the previous thinking
#       5.2     Think + call tools until the step is resolved
#       5.3     Ask the model whether the step goal was achieved; retry from 5.1 if not
#       5.4     Log tool usage and store the output so that the next step may see what happened
#       5.5     Advance to the next step
# 6.    Ask the model whether the overall goal was achieved; tell the user and replan + restart if not
# 7.    Ask the model for a human-readable output summary
# 8.    Persist user request / tool usage / summary to Application.history

import drawing
from ollama_worker import OllamaWorker
import sys
from typing import *  # type: ignore


def main(args: Tuple[str, ...]) -> None:
    worker = OllamaWorker()

    worker.start(args[1])


if __name__ == "__main__":
    main(tuple(sys.argv))
