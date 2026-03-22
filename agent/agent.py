import json
import os
from datetime import datetime
from openai import OpenAI
from tools import TOOLS, run_tool
client = OpenAI()

# ── System prompt ─────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    tool_descriptions = "\n".join(
        f"- {name}: {meta['description']}"
        for name, meta in TOOLS.items()
    )
    return f"""You are a ReAct agent. You solve tasks by alternating between thinking and acting.

You have access to these tools:
{tool_descriptions}

You MUST follow this exact format on every turn — no exceptions:

Thought: <your reasoning about what to do next>
Action: <tool_name>
Action Input: <input to the tool>

When you have enough information to answer, use:

Thought: <your final reasoning>
Answer: <your final answer to the user>

Rules:
- Always start with a Thought
- Never skip the Action/Action Input if you need more information
- Never make up information — use tools to verify
- write_file input format is always: filename.txt|content here
"""

# ── Response parser ───────────────────────────────────────────────────────────
def parse_response(text: str) -> dict:
    """Parse LLM output into structured dict."""
    result = {"thought": "", "action": None, "action_input": None, "answer": None}

    lines = text.strip().splitlines()
    current_key = None
    buffer = []

    for line in lines:
        if line.startswith("Thought:"):
            current_key = "thought"
            buffer = [line[len("Thought:"):].strip()]
        elif line.startswith("Action:"):
            if current_key == "thought":
                result["thought"] = " ".join(buffer).strip()
            current_key = "action"
            result["action"] = line[len("Action:"):].strip()
            buffer = []
        elif line.startswith("Action Input:"):
            current_key = "action_input"
            buffer = [line[len("Action Input:"):].strip()]
        elif line.startswith("Answer:"):
            if current_key == "action_input":
                result["action_input"] = " ".join(buffer).strip()
            current_key = "answer"
            buffer = [line[len("Answer:"):].strip()]
        else:
            buffer.append(line.strip())

    # flush last buffer
    if current_key == "thought":
        result["thought"] = " ".join(buffer).strip()
    elif current_key == "action_input":
        result["action_input"] = " ".join(buffer).strip()
    elif current_key == "answer":
        result["answer"] = " ".join(buffer).strip()

    return result

# ── Trace logger ──────────────────────────────────────────────────────────────
def save_trace(trace: list, task: str):
    """Save the full agent trace to a JSON file."""
    os.makedirs("traces", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"traces/trace_{timestamp}.json"
    payload = {"task": task, "steps": trace}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n📝 Trace saved to {filename}")

# ── Main agent loop ───────────────────────────────────────────────────────────
def run_agent(task: str, max_steps: int = 8) -> str:
    print(f"\n{'='*60}")
    print(f"TASK: {task}")
    print(f"{'='*60}")

    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user",   "content": task},
    ]

    trace = []

    for step in range(1, max_steps + 1):
        print(f"\n── Step {step} {'─'*40}")

        # ── LLM call ──────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0,
        )
        raw = response.choices[0].message.content
        parsed = parse_response(raw)

        print(f"Thought : {parsed['thought']}")

        # ── Build trace step ──────────────────────────────────────────────
        step_log = {"step": step, "thought": parsed["thought"]}

        # ── Final answer ──────────────────────────────────────────────────
        if parsed["answer"]:
            print(f"Answer  : {parsed['answer']}")
            step_log["answer"] = parsed["answer"]
            trace.append(step_log)
            save_trace(trace, task)
            return parsed["answer"]

        # ── Tool call ─────────────────────────────────────────────────────
        if not parsed["action"]:
            print("⚠️  No action or answer found — stopping.")
            break

        action       = parsed["action"]
        action_input = parsed["action_input"] or ""

        print(f"Action  : {action}")
        print(f"Input   : {action_input}")

        observation = run_tool(action, action_input)
        print(f"Observe : {observation[:300]}{'...' if len(observation) > 300 else ''}")

        step_log.update({
            "action": action,
            "action_input": action_input,
            "observation": observation,
        })
        trace.append(step_log)

        # ── Feed observation back into conversation ────────────────────────
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user",      "content": f"Observation: {observation}"})

    save_trace(trace, task)
    return "Agent reached max steps without a final answer."