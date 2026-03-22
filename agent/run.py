from agent import run_agent

if __name__ == "__main__":
    task = (
        "Find the abstract of the 'Attention is All You Need' paper "
        "and save it to abstract.txt"
    )
    result = run_agent(task)
    print(f"\nFinal Answer: {result}")