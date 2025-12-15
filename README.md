# First Aid Micro-Guide RAG Project

This project is a small Retrieval-Augmented Generation (RAG) system that answers basic **first-aid questions** (for example: minor cuts, mild burns, simple nosebleeds, fainting, sprains). The goal is not medical diagnosis, but to demonstrate how to build a full data → embeddings → vector search → API pipeline.

> **Disclaimer:** This system provides simple first-aid information only and is **not** a substitute for professional medical advice, diagnosis, or emergency care.

---

## 1. Project Overview

- Domain: basic first aid scenarios (cuts, burns, nosebleeds, fainting, sprains, etc.).
- Goal: given a user question like *“What should I do for a minor cut?”*, retrieve the most relevant first-aid instructions from a small, curated dataset.
- Focus: showing the full RAG stack:
  - ETL of custom data
  - Document creation
  - Embeddings
  - FAISS vector store
  - Flask API for querying

---

## 2. Data

All data lives under `data/`:

- `data/etl_first_aid_microguide - Sheet1.csv`
  - A small table with rows like:
    - `CUT_MINOR` – minor cut
    - `BURN_MILD` – mild burn (no blisters)
    - `NOSEBLEED_SIMPLE` – simple nosebleed
  - Columns include:
    - `condition_id`
    - `condition_name`
    - `category`
    - `severity_tag`
    - `common_signs`
    - `immediate_steps`
    - `do_not_do`
    - `when_to_seek_help`
    - `follow_up_advice`
    - `source`

- `data/additional_documents/First Aid Quick Guide.pdf`
  - A short, student-written “quick guide” to first aid: cuts, burns, nosebleeds, fainting, sprains.
  - Used as an extra text source for retrieval.

- `data/additional_documents/first_aid_notes.txt`
  - Simple notes including:
    - Red flag symptoms (e.g., uncontrolled bleeding, trouble breathing)
    - General first-aid principles
    - Acronym R.I.C.E. (Rest, Ice, Compression, Elevation)

---

## 3. RAG Pipeline

All RAG setup code is in **`src/ingestion.ipynb`**. The main steps are:

1. **Load CSV**
   - Read `etl_first_aid_microguide - Sheet1.csv` into a Pandas DataFrame.
   - For each row, build a structured text “document” that includes:
     - condition name
     - common signs
     - immediate first-aid steps
     - what NOT to do
     - when to seek professional help
     - follow-up advice

2. **Load PDF & TXT**
   - Use `pypdf` to extract text from `First Aid Quick Guide.pdf`.
   - Read `first_aid_notes.txt` as plain text.
   - Wrap each into a text document with simple source tags.

3. **Combine Documents**
   - Combine:
     - CSV-based documents
     - PDF-based document(s)
     - Notes document
   - Store them in a Python list called `all_documents`.

4. **Embeddings**
   - Model: `BAAI/bge-small-en-v1.5` from `sentence-transformers`.
   - Create a 384-dimensional embedding for each document.
   - Normalize embeddings and store in a NumPy array.

5. **Vector Store**
   - Use FAISS `IndexFlatIP` (inner product on normalized vectors ≈ cosine similarity).
   - Add all document embeddings to the FAISS index.
   - Save:
     - `vector_store/first_aid_index.faiss`
     - `vector_store/first_aid_docs.pkl`

This gives a reusable semantic search index for the first-aid domain.

---

## 4. Project Structure

```text
firstaid_rag_project/
│
├── data/
│   ├── etl_first_aid_microguide - Sheet1.csv
│   └── additional_documents/
│        ├── First Aid Quick Guide.pdf
│        └── first_aid_notes.txt
│
├── src/
│   └── ingestion.ipynb          # ETL + embeddings + FAISS index creation
│
├── vector_store/
│   ├── first_aid_index.faiss    # FAISS index
│   └── first_aid_docs.pkl       # List of document texts
│
├── api/
│   ├── app.py                   # Flask API that uses the index
│   └── requirements.txt         # Backend dependencies
│
└── README.md
