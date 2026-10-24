# NLP Insight Studio — Topic Modeling & Text Summarization

A student-friendly NLP web application for the Week 5 Advanced NLP Techniques task.

## Features
- Upload TXT, PDF, and DOCX files or paste text
- Text preprocessing preview
- LDA topic modeling, topic keywords, average topic weights, optional c_v coherence
- TextRank-style extractive summarization
- Optional Hugging Face transformer abstractive summarization
- Keyword extraction module
- Multiple-document comparison and combined summaries
- Evaluation with compression statistics and optional ROUGE
- Optional BERTopic and BERTScore extensions
- Export summary as TXT/PDF and topics as JSON
- SQLite-based account registration, password hashing, saved history, and admin view
- Responsive Streamlit dashboard

## Requirements
Python 3.10–3.12 recommended. Python 3.13 may have dependency compatibility differences.

## Setup (Windows)
```bash
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If `py` is unavailable, use `python -m venv .venv`.

## Optional transformer and BERTopic models
```bash
pip install -r requirements-optional.txt
```
The first transformer/BERTopic run may download model weights and can require substantial RAM/disk space. If you have limited memory or no internet, use TextRank and LDA.

## Create an admin account
1. Register an account in the app.
2. Stop the Streamlit server.
3. Run this command from the project folder, replacing `YOUR_USERNAME`:
```bash
python -c "from nlp_core.storage import init_db,promote_admin; init_db(); print(promote_admin('YOUR_USERNAME'))"
```
4. Restart the app and sign in again.
Do not expose this local database or admin setup to the public internet. This is a learning-project authentication layer, not a production-ready identity system.

## GitHub upload
1. Create a new public repository named `advanced-nlp-topic-modeling-summarization`.
2. Do not upload `.venv`, the SQLite database, uploaded documents, model weights, or secrets.
3. Add the project files and commit:
```bash
git init
git add .
git commit -m "Build NLP topic modeling and summarization app"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/advanced-nlp-topic-modeling-summarization.git
git push -u origin main
```
Replace `YOUR_USERNAME` with your GitHub username. Copy the real repository URL into your internship submission only after pushing successfully.

## Project structure
```text
advanced_nlp_project/
├── app.py
├── nlp_core/
│   ├── documents.py
│   ├── preprocessing.py
│   ├── topics.py
│   ├── summarization.py
│   ├── keywords.py
│   ├── evaluation.py
│   ├── storage.py
│   └── reporting.py
├── data/
├── tests/
├── requirements.txt
├── requirements-optional.txt
└── README.md
```

## Method notes and limitations
- LDA is a bag-of-words probabilistic model; it ignores word order and works best with enough documents and meaningful token frequency.
- TextRank is extractive: selected sentences come from the input, so it may sound less fluent or repeat context.
- Transformer summarization is abstractive and may paraphrase or hallucinate. Check facts before relying on output.
- ROUGE needs a human/reference summary and measures overlap, not truth.
- Topic coherence is a proxy for interpretability, not proof that topics are useful.
- The PDF parser extracts selectable text; scanned PDFs may need OCR.
- Basic sentence splitting and stop-word lists are intentionally lightweight.
- For robust real-world use, add tests, rate limits, file-size limits, secure deployment, and privacy controls.

## Suggested testing
- Run on a short article, a multi-page report, and at least five related documents.
- Try 3, 5, and 8 LDA topics; record coherence and inspect keywords.
- Compare TextRank and transformer summaries against a human-written reference using ROUGE-1/2/L.
- Ask a reviewer to rate factual consistency, coverage, readability, and usefulness.
- Document hardware, runtime, library versions, and failed cases honestly.
