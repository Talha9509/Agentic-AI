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
prompt1 = "Hi!"
prompt2 = "Explain time travel in detail"
prompt3 = "Write a 1000 word essay on Machine Learning"
prompts = [prompt1, prompt2, prompt3]

for prompt in prompts:
    message = {
        "role": role,
        "content": prompt
    }
    messages = [message]
    response = client.chat.completions.create(model=model, messages=messages, max_tokens=4, max_completion_tokens=1000)
    # print(response.choices[0].message.content)
    usage = response.usage
    print(f"Prompt: {prompt} --> your tokens: {usage.prompt_tokens}, completion tokens: {usage.completion_tokens}, finish reason: {response.choices[0].finish_reason}")