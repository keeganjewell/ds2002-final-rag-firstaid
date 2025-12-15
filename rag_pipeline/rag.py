import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_PATH = "vector_store/first_aid_index.faiss"
DOCS_PATH = "vector_store/first_aid_docs.pkl"

print("Loading model and vector store...")
model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)

with open(DOCS_PATH, "rb") as f:
    documents = pickle.load(f)

def answer_question(question):
    """
    Given a question string:
    - embed it
    - search FAISS index
    - retrieve top documents
    - RETURN (answer_text, sources_list)
    """

    query_vector = model.encode([question])
    
    distances, ids = index.search(np.array(query_vector).astype("float32"), 3)

    retrieved_docs = [documents[i] for i in ids[0]]

    answer_text = retrieved_docs[0] if retrieved_docs else "No information available."

    return answer_text, retrieved_docs
