# PaperBrain

PaperBrain is a Streamlit app that lets you upload PDF documents, extract text, create embeddings, and ask questions using retrieval-augmented generation.

## Features

- Upload one or more PDF files
- Extract text using PyPDF2
- Split document text into smaller chunks
- Create embeddings with sentence-transformers
- Store vectors in FAISS for fast similarity search
- Answer questions using OpenAI chat completions grounded in retrieved text
- Display chat history and source excerpts

---

## 📷 Preview

> ![Image Alt](https://github.com/EnukaSathmina/PaperBrain/blob/main/PBimg.png?raw=true)

---

## Setup

1. Create a Python environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:

```bash
setx OPENAI_API_KEY "your_api_key_here"
```

4. Run the app:

```bash
streamlit run app.py
```

## Usage

1. Upload PDF files using the upload widget.
2. Click `Process PDFs` to extract text and build the vector index.
3. Enter a question in the chat box and click `Ask PaperBrain`.

The app retrieves relevant chunks from the PDF and uses the LLM to answer only from that content.
