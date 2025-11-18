import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load your Q&A file
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract questions for embedding
questions = [item["question"] for item in data]

# Load sentence transformer model
print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(questions)

# Convert embeddings to float32 (FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]  # usually 384
index = faiss.IndexFlatL2(dimension)

print("Adding vectors to FAISS index...")
index.add(embeddings)

# Save index
faiss.write_index(index, "faiss_index/my_index.faiss")

print("FAISS index successfully created!")

# Also save data.json as mapping
with open("faiss_index/mapping.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4)

print("Mapping file saved.")
