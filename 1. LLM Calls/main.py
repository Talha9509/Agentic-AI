import os
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is req")

client = Groq(api_key=my_api_key)

model = "openai/gpt-oss-120b"
role = "user"
prompt = "Hi, I want to know who is India's First PM?"
message = {
    "role": role,
    "content": prompt
}
messages = [message]

response = client.chat.completions.create(model=model, messages=messages)

print(response.choices[0].message.content)