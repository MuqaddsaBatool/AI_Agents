# memory.py
# Short-term memory: stores critiques from previous runs
# and injects them into the next run's system prompt.

_critiques: list[str] = []

def add_critique(critique: str):
    """Store a critique from a completed run."""
    _critiques.append(critique)
    # Keep only the last 3 — avoid bloating the context window
    if len(_critiques) > 3:
        _critiques.pop(0)

def get_critiques() -> list[str]:
    return list(_critiques)

def clear():
    _critiques.clear()