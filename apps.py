from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# -------------------------------
# 1. Load the model (same used in index.py)
# -------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# 2. Load FAISS index
# -------------------------------
index = faiss.read_index("faiss_index/my_index.faiss")

# -------------------------------
# 3. Load mapping file (Q&A data)
# -------------------------------
with open("faiss_index/mapping.json", "r", encoding="utf-8") as f:
    mapping = json.load(f)   # list of {question, answer}


@app.route("/chat", methods=["POST"])
def chat():
    user_msg = request.json.get("message")

    # 1. Convert user message to embedding
    user_embedding = model.encode([user_msg]).astype("float32")

    # 2. Search FAISS (find top-1 closest question)
    top_k = 1
    distances, indices = index.search(user_embedding, top_k)

    best_index = int(indices[0][0])
    best_distance = float(distances[0][0])

    # Distance threshold — lower = better match
    THRESHOLD = 1.5

    if best_distance > THRESHOLD:
        return jsonify({"response": "Sorry, I don't understand. Try rephrasing!"})

    # 3. Return answer from mapping file
    best_answer = mapping[best_index]["answer"]
    return jsonify({"response": best_answer})


if __name__ == "__main__":
    app.run(debug=True)
