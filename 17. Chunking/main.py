
import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text = """
Artificial intelligence is a branch of computer science that allows machines to learn, reason, and solve problems like humans.

It uses massive amounts of data and special software rules called algorithms to spot patterns, recognize speech, and make choices.

You use it every day when your phone unlocks with your face, when an online store suggests a product, or when a digital assistant answers a question. 

While these systems help us work faster and automate routine tasks, people must guide them carefully to ensure safety, fairness, and privacy for everyone
"""

fixed = CharacterTextSplitter(
    separator="",
    chunk_size=100,
    chunk_overlap=0
)

print("-------FIXED-----------")

for i, chunk in enumerate(fixed.split_text(text), 1):
    print(f"Chunk {i}: ")
    print(chunk)
    print("-------------------")
    
    
paragraph = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=0
)
    
print("-------PARAGRAPH-----------")

for i, chunk in enumerate(paragraph.split_text(text), 1):
    print(f"Chunk {i}: ")
    print(chunk)
    print("-------------------")


recursive = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)
    
print("-------RECURSIVE-----------")

for i, chunk in enumerate(recursive.split_text(text), 1):
    print(f"Chunk {i}: ")
    print(chunk)
    print("-------------------")
