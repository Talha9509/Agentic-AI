import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is req")

client = Groq(api_key=my_api_key)
groqModel = "openai/gpt-oss-120b"
model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Employees receive 24 days of paid leave per year.",
   
    "Employees work from the office on Tuesday, Wednesday and Thursday. "
    "Monday and Friday are optional work-from-home days.",
   
    "Employees receive Rs 3000 per month for gym reimbursement.",
   
    "Employees can claim Rs 2000 per month for home internet.",
   
    "Employees have a 90 day notice period."
]

doc_embeds = model.encode(documents)
# doc_embed gives array of each documents[i] embedding array
print(doc_embeds)

def cosine_sim(a, b):
  return np.dot(a, b) / (
    np.linalg.norm(a) * np.linalg.norm(b)
  )
  
def retrieve(query_embed):
    scores = []
    for i, doc_embed in enumerate(doc_embeds):
        score = cosine_sim(query_embed, doc_embed)
        scores.append((score, documents[i]))
    scores.sort(reverse=True)
    return scores[0]

def ask_llm(query, context):

    sys_prompt=f"""answer in one line only. Answer only based on this context. do not hallucinate. Context: {context}"""
    system_message={
        "role": "system",
        "content": sys_prompt

    }
    message={
        "role": "user",
        "content": query
    }
    messages=[system_message, message]
    response=client.chat.completions.create(model=groqModel, messages=messages)
    answer=response.choices[0].message.content
    return answer



query = "How much vacation do I get?"
query_embed = model.encode(query)
print()
print()
print("-------------------------------------------------------")
print()
print()
score, context = retrieve(query_embed)
print(score)
print(context)

answer = ask_llm(query, context)
print(answer)
