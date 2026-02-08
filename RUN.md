# Run locally

## Setup

```bash
cd d:\PROJECT\offline-rag-bot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
```

## Run

```bash
python app.py
```

Open http://localhost:5000. Put PDF, TXT, MD, DOCX, or CSV files in `data/documents/` (or upload via the UI). Ask questions; answers are grounded only in those documents. If the answer is not in the documents, the bot responds with:

"The provided documents do not contain enough information to answer this question."

## Optional: rebuild index after adding files on disk

```bash
python rebuild_vectorstore.py
```

Then restart `python app.py`.
