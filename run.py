from agent import run_agent

# TASK = (
#     "Find the abstract of the 'Attention is All You Need' paper "
#     "and save it to abstract.txt"
# )

# if __name__ == "__main__":
#     scores = []

#     for i in range(1, 4):
#         print(f"\n{'#'*60}")
#         print(f"# RUN {i}")
#         print(f"{'#'*60}")
#         result = run_agent(TASK)

#     print("\n All 3 runs complete. Check traces/ for full logs.")

# TASK = (
#     "Using the research papers in the knowledge base, "
#     "explain how Reflexion improves on the basic ReAct approach. "
#     "Save the explanation to comparison.txt"
# )

TASK = (
    "What is the difference between the observation space "
    "in ReAct vs Reflexion? Save the answer to observation_diff.txt"
)

if __name__ == "__main__":
    result = run_agent(TASK)
    print(f"\nFinal Answer: {result}")