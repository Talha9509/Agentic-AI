# ============================================================
# PART 1 — IMPORTS AND ENVIRONMENT
# ============================================================

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue, MatchAny, PayloadSchemaType
from sentence_transformers import SentenceTransformer
from groq import Groq
import json

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


# ============================================================
# PART 2 — CONNECT TO QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

print("connected to qdrant")


# ============================================================
# PART 3 — CREATE QDRANT COLLECTION
# ============================================================

COLLECTION_NAME = "knowledge_filter"
EMBEDDING_SIZE = 384

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=EMBEDDING_SIZE,
        distance=Distance.COSINE
    )
)

client.create_payload_index(
    collection_name=COLLECTION_NAME,
    field_name="category",
    field_schema=PayloadSchemaType.KEYWORD
)


# ============================================================
# PART 4 — LOAD OUR KNOWLEDGE
# ============================================================

with open("knowledge.json", "r", encoding="utf-8") as f:
    documents = json.load(f)
    
print(documents)


# ============================================================
# PART 5 — CREATE EMBEDDINGS
# ============================================================

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [document["text"] for document in documents]
print(texts)

embeddings = model.encode(texts)


# ============================================================
# PART 6 — CREATE QDRANT POINTS
# ============================================================

points = []

for i in range(len(documents)):
    point = PointStruct(
        id=i+1,
        vector=embeddings[i].tolist(),
        payload=documents[i]
    )
    points.append(point)
    

# ============================================================
# PART 7 — UPLOAD TO QDRANT
# ============================================================
    
client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)


# ============================================================
# PART 8 — SEARCH QDRANT
# ============================================================

def search(query, top_k=3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True
    ).points
    
    return results

def search_with_filter(query, query_filter=None, top_k=3):
    query_vector = model.encode(query).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter
    ).points
    
    return results

reimbursment_filter = Filter(
    must=[
        FieldCondition(
            key="category",
            match=MatchValue(value="reimbursement")
        )
    ]
)


# ============================================================
# PART 9 — TEST SEARCH
# ============================================================

query = "How many vacation days do I get?"

results = search_with_filter(query, reimbursment_filter, top_k=3)
for result in results:
    print(f"Score: {result.score}")
    print(result.payload["text"])
    print()
    
    
    
groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# PART 10 — ASK THE LLM
# ============================================================

def ask_llm(query, context):
    prompt = f"""
    Answer the question using only the information provided below.

Context:
{context}

Question:
{query}

If the answer is not present in the context, say:
"I don't know based on the provided information."
    """
    
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    return response.choices[0].message.content



# ============================================================
# PART 11 — COMPLETE RAG PIPELINE
# ============================================================

question = "How many vacation days do I get?"

results = search(question, top_k=3)


# Extract text from the search results
context = "\n".join(
    result.payload["text"]
    for result in results
)


answer = ask_llm(question, context)


print("\nFinal Answer:")
print(answer)