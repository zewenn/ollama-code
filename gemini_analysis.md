This is a fantastic project. Building a single-file, local "vibe-coding" agent with a custom terminal UI and streaming thoughts is no small feat. I can clearly see your intentions: you want a lightweight, highly capable assistant that bypasses heavy frameworks (like LangChain or AutoGen) in favor of a clean, transparent event loop, direct Ollama integration, and a sophisticated "plan mode" for autonomous, multi-step execution.

However, getting the agentic loop—especially the history management and tool-call constraints—right in a streaming context is notoriously difficult. I have analyzed `main.py` and found a few structural and logical issues that will cause erratic behavior, particularly when executing plans.

Here is a candid breakdown of the issues, rated by importance.

### 1. History Desynchronization (Context Amnesia)

**Severity: 10/10**

Your `_run_plan_mode` creates a fundamental disconnect between the model's active working memory and the global application state.

- **The Issue:** Inside `_run_plan_mode`, you pass a cloned `history` list to `_run_one_task`. The agent loops, calling tools, and appending raw `assistant` and `tool` messages to this `history` list. However, when the task finishes, you only append the **llama3.2 generated summary** to `state.messages`.
- **The Result:** When the plan finishes and control returns to the user, the next time the user types a prompt, `worker.start()` is called with `state.messages`. The agent will have _completely forgotten_ the actual tool outputs, file diffs, and code it just wrote. It only sees a one-line summary (e.g., "Created main.py with helper functions"). If the user asks, "Can you modify that function you just wrote?", the model will hallucinate because the actual code isn't in its context history.
- **The Fix:** You must merge the actual `history` from `_run_plan_mode` back into `state.messages` after the plan completes, or ensure that file reads/writes are preserved in the conversation history, not just the high-level summaries.

### 2. Brittle Plan Step Completion Mechanics

**Severity: 9/10**

Your mechanism for advancing to the next plan step forces the LLM into a corner that it will inevitably fail to escape.

- **The Issue:** In `_run_one_task`, a task is only considered complete if `self._state.task_work_done` is `True`. This flag is _only_ set by `write_file`, `create_file`, or `run_command`. Furthermore, you have a hard cap of `max_turns = 10`.
- **The Result:** If the task is "Write src/utils.py", but the agent needs to read 3 different files first to understand the required imports, it might use 4 or 5 turns just reading and searching. If it takes more than 10 turns, or if the task inherently didn't require writing (e.g., the model realizes the code already exists), the loop hits the cap, returns `False`, and the entire plan aborts with an error.
- **The Fix:** Allow the model to explicitly call a `complete_task` tool (which you already defined but restricted from `_PLAN_TOOLS`), or use a more forgiving heuristic. The model is the best judge of when it has satisfied the prompt.

### 3. Context Window Bloat During Plan Execution

**Severity: 8/10**

This is the inverse of Issue #1, occurring _during_ the plan's execution.

- **The Issue:** In `_run_plan_mode`, `history` is shared across all tasks. Every time the agent reads a file (which can return up to 300 lines) or writes a file (which returns a unified diff), that block of text is permanently added to `history` for the remainder of the plan.
- **The Result:** If a plan has 6 steps, by step 5, the prompt includes all the raw file reads and diffs from steps 1-4. For an 8B model like `qwen3:8b`, this will rapidly degrade reasoning quality, increase time-to-first-token (TTFT), and eventually blow the context window, causing an Ollama API crash.
- **The Fix:** Implement a "context compression" step between tasks. Keep the system prompt and user intent, but compress previous `tool` interactions into summaries _within_ the running `history` list, rather than just appending summaries at the end.

### 4. Tool Call Deduplication via Stringification

**Severity: 6/10**

The way you deduplicate streaming tool calls is vulnerable to non-deterministic JSON serialization.

- **The Issue:** In `_stream_one_turn`, you deduplicate tool calls using:
  `key = (tc.function.name, str(tc.function.arguments))`
- **The Result:** `tc.function.arguments` is a dictionary. In Python, calling `str()` on a dict relies on key insertion order. While modern Python preserves order, the way the Ollama streaming API reconstructs JSON chunks might result in the same arguments arriving with differently ordered keys. If this happens, `str(dict)` differs, the deduplication fails, and the agent executes the exact same tool twice in a row.
- **The Fix:** Use deterministic JSON serialization for the deduplication key:
  `key = (tc.function.name, json.dumps(tc.function.arguments, sort_keys=True))`

### 5. Fragile Planner Output Parsing

**Severity: 5/10**

LLMs are notoriously bad at strictly following formatting instructions without preambles.

- **The Issue:** `_run_planner` parses the plan using `re.match(r"^(?:\d+[.)\-]\s*|[-*]\s+)(.*)", line)`. It silently ignores anything that doesn't match.
- **The Result:** If the model outputs:
  `Here is the step-by-step plan:`
  `Step 1. create README.md`
  Your regex fails to capture "Step 1" because of the word "Step". `items` will be empty, and plan mode will silently fail to start.
- **The Fix:** Relax the regex to search for numbers followed by list markers anywhere near the start of the line (e.g., `r"^\s*(?:Step\s*)?\d+[.)\-]\s*(.*)"`), or force the planner to output pure JSON.

---

Would you like me to rewrite the `_run_plan_mode` and `_run_one_task` methods to fix the history synchronization and context bloat issues while keeping it entirely within your single-file architecture?
