import os
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is req")

client = Groq(api_key=my_api_key)

message_system = {
    "role": "system",
    "content": "You are a Brand Manager who suggests name for the food company. Name should be in 1 word"
}

model = "openai/gpt-oss-120b"
role = "user"
prompt = "Suggest a name for my food company"
message = {
    "role": role,
    "content": prompt
}
messages = [message_system, message]

# Temperature is between 0 and 2
response = client.chat.completions.create(model=model, messages=messages, temperature=2)

print(response.choices[0].message.content)