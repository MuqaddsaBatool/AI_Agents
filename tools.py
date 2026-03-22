import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from retriever import retrieve
load_dotenv()

def web_search(query: str) -> str:
    """Search the web using Serper API and return top results."""
    import json
    
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "Error: SERPER_API_KEY not set in .env"
    
    headers = {
        "X-API-KEY": api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query, "num": 5}
    
    try:
        response = requests.post(
            "https://google.serper.dev/search",
            headers=headers,
            json=payload,
            timeout=10
        )
        data = response.json()
        
        output = []
        for i, r in enumerate(data.get("organic", [])[:5]):
            title   = r.get("title", "")
            snippet = r.get("snippet", "")
            link    = r.get("link", "")
            output.append(f"[{i+1}] {title}\n    {snippet}\n    URL: {link}")
        
        return "\n\n".join(output) if output else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


# ── Tool 2: Read URL ──────────────────────────────────────────────────────────
def read_url(url: str) -> str:
    """Fetch a webpage and return its main text content (truncated to 2000 chars)."""
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [l for l in text.splitlines() if l.strip()]
        content = "\n".join(lines)[:2000]
        
        return content if content else "Could not extract content."
    except Exception as e:
        return f"Read error: {str(e)}"


# ── Tool 3: Write File ────────────────────────────────────────────────────────
def write_file(filename: str, content: str) -> str:
    """Write content to a file in the current directory."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to '{filename}'."
    except Exception as e:
        return f"Write error: {str(e)}"


# ── Tool registry ─────────────────────────────────────────────────────────────
TOOLS = {
    "web_search": {
        "fn": web_search,
        "description": "Search the web for current information. Input: a search query string.",
    },
    "read_url": {
        "fn": read_url,
        "description": "Read and extract text from a webpage. Input: a full URL string.",
    },
    "write_file": {
        "fn": write_file,
        "description": "Write text content to a file. Input: 'filename.txt|content here'  (pipe-separated).",
    },
    "search_papers": {
    "fn": retrieve,
    "description": (
        "Search the local research paper knowledge base for relevant content. "
        "Returns excerpts with source citations. "
        "Use this when the task involves concepts from research papers. "
        "Input: a specific question or concept to search for."
    ),
    },
}

def run_tool(tool_name: str, tool_input: str) -> str:
    """Execute a tool by name and return its output."""
    if tool_name not in TOOLS:
        return f"Unknown tool: '{tool_name}'. Available: {list(TOOLS.keys())}"
    
    tool = TOOLS[tool_name]["fn"]
    
    # write_file needs two args — split on first pipe
    if tool_name == "write_file":
        parts = tool_input.split("|", 1)
        if len(parts) != 2:
            return "write_file input must be: 'filename|content'"
        return tool(parts[0].strip(), parts[1].strip())
    
    return tool(tool_input)
#Testing
# if __name__ == "__main__":
#     print(web_search("ReAct language model agent"))
#     print("---")
#     print(write_file("test.txt", "hello from agent"))