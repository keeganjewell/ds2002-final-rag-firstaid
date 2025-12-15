from flask import Flask, request, jsonify
from rag_pipeline.rag import answer_question

app = Flask(__name__)

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    # Call your RAG pipeline
    answer, sources = answer_question(question)

    return jsonify({
        "answer": answer,
        "sources": sources
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
