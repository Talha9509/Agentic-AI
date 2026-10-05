import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
text = "Machine learning is good"

embedding = model.encode(text)
print(embedding.shape)
print(embedding[:10])

def cosine_sim(a, b):
  return np.dot(a, b) / (
    np.linalg.norm(a) * np.linalg.norm(b)
  )

t1 = "There are 24 holidays"
t2 = "There are 24 vacations"

v1 = model.encode(t1)
v2 = model.encode(t2)

print(cosine_sim(v1, v2))