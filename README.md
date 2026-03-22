# ReAct Agent

A ReAct agent built from scratch using the OpenAI API — no LangChain.

## What it does
- Searches the web (Serper API)
- Reads URLs
- Writes files
- Logs every step as a JSON trace

## Setup
```
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install openai python-dotenv requests beautifulsoup4
```

Add a \.env\ file:
```
with your credentials
```

## Run
```Bash
python run.py
```

