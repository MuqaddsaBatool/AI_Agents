from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Reply with: setup works"}]
)

print(response.choices[0].message.content)