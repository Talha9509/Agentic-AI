# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from groq import Groq

load_dotenv()

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
print("Connected to Qdrant")


# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "knowledge"
EMBEDDING_SIZE = 384

if client.collection_exists(COLLECTION_NAME):
    print("deleting collection")
    client.delete_collection(COLLECTION_NAME)
    
client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)

print(f"Created collection: {COLLECTION_NAME}")
print(f"Vector size: {EMBEDDING_SIZE}")
print("Distance: COSINE")


# ============================================================
# PART 4 — LOAD OUR KNOWLEDGE
# ============================================================

with open("knowledge.txt", "r", encoding="utf-8") as file:
    documents = []
    content = file.readlines()
    for line in content:
        documents.append(line.strip())
    print(documents)

print("loaded documents")


# ============================================================
# PART 5 — CREATE EMBEDDINGS
# ============================================================

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(documents)


# ============================================================
# PART 6 — CREATE QDRANT POINTS
# ============================================================

points = []
for i, embed in enumerate(embeddings):
    point = PointStruct(
        id=i+1,
        vector=embed.tolist(),
        payload={
            "text": documents[i]
        }
    )
    points.append(point)
    

# ============================================================
# PART 7 — UPLOAD TO QDRANT
# ============================================================

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print("uploaded points to qdrant")


# ============================================================
# PART 8 — SEARCH QDRANT
# ============================================================

def search(query, top_k=3):
    query_vector = model.encode(query)
    
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points
    
    return results


# ============================================================
# PART 9 — TEST SEARCH
# ============================================================

query = "How many vacation days do i get?"
results = search(query)
print(results)

print("Search results: \n")

for result in results:
    print(f"Score: {result.score: .3f}")
    print(result.payload["text"])
    print()
    

# ============================================================
# PART 10 — CONNECT TO GROQ
# ============================================================
    
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("API key is req")

groqClient = Groq(api_key=my_api_key)
groqModel = "openai/gpt-oss-120b"


# ============================================================
# PART 11 — ASK THE LLM
# ============================================================
    
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
    response = groqClient.chat.completions.create(model=groqModel, messages=messages)
    answer = response.choices[0].message.content
    return answer


# ============================================================
# PART 12 — COMPLETE RAG PIPELINE
# ============================================================

context = ""
for result in results:
    context = context + result.payload["text"] + "\n"
    
print("context")
print(context)

answer = ask_llm(query, context)
print("Final answer:")
print(answer)
    

    
    
    
    