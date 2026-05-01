import os
import io
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai

# ----------------------------
# LOAD ENV
# ----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

# ----------------------------
# CONFIG
# ----------------------------
st.set_page_config(page_title="PaperBrain", page_icon="🧠", layout="wide")

MAX_FILE_SIZE = 5_000_000        # 5MB
MAX_TEXT_LENGTH = 200_000
MAX_CHUNKS = 200

# ----------------------------
# LOAD MODEL (CACHED)
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-MiniLM-L3-v2")  # lighter model

# ----------------------------
# EXTRACT TEXT
# ----------------------------
def extract_text(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""

    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"

    return text

# ----------------------------
# SPLIT TEXT (INTELLIGENT)
# ----------------------------
def split_text(text, chunk_size=800, overlap=150):
    """Split text by sentences when possible to avoid breaking mid-sentence."""
    import re
    
    # Split by sentence boundaries (. ! ?) while preserving them
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += (" " if current_chunk else "") + sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
            
            # Fallback: if single sentence exceeds chunk_size, split it
            if len(sentence) > chunk_size:
                words = sentence.split()
                temp_chunk = ""
                for word in words:
                    if len(temp_chunk) + len(word) < chunk_size:
                        temp_chunk += (" " if temp_chunk else "") + word
                    else:
                        if temp_chunk:
                            chunks.append(temp_chunk)
                        temp_chunk = word
                if temp_chunk:
                    current_chunk = temp_chunk
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Add overlap by including parts of previous chunks
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_chunk = chunks[i-1]
            # Add last `overlap` chars of previous chunk to beginning
            overlap_text = prev_chunk[-overlap:] if len(prev_chunk) > overlap else prev_chunk
            chunk = overlap_text + " " + chunk
        overlapped_chunks.append(chunk)
    
    return overlapped_chunks[:MAX_CHUNKS]

# ----------------------------
# BUILD INDEX
# ----------------------------
def build_index(chunks):
    model = load_model()

    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        batch_size=16
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index

# ----------------------------
# SEARCH
# ----------------------------
def search(query, chunks, index):
    model = load_model()

    # Encode main query
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    # Get top results with scores
    scores, idx = index.search(q, k=15)
    
    # Filter by relevance threshold (min similarity: 0.3) and deduplicate
    results = []
    seen_text = set()
    
    for i, score in zip(idx[0], scores[0]):
        if score < 0.3:  # Skip low-relevance results
            continue
        if i >= len(chunks):
            continue
        
        chunk = chunks[i]
        # Deduplicate: skip if text is >90% similar to already included chunks
        is_duplicate = False
        for seen in seen_text:
            overlap = len(set(chunk.split()) & set(seen.split())) / max(len(set(chunk.split())), len(set(seen.split())))
            if overlap > 0.9:
                is_duplicate = True
                break
        
        if not is_duplicate:
            results.append(chunk)
            seen_text.add(chunk)
            if len(results) >= 8:  # Limit to 8 unique results
                break
    
    return results if results else [chunks[i] for i in idx[0][:5] if i < len(chunks)]

# ----------------------------
# GEMINI RESPONSE
# ----------------------------
def get_answer(question, context):
    if not API_KEY:
        return "❌ Missing GOOGLE_API_KEY in .env"

    genai.configure(api_key=API_KEY)
    
    # Deduplicate and limit context to control token usage
    unique_context = []
    seen = set()
    for chunk in context:
        chunk_key = chunk[:100]  # Use first 100 chars as key
        if chunk_key not in seen:
            unique_context.append(chunk)
            seen.add(chunk_key)
    
    context_text = "\n\n---\n\n".join(unique_context[:8])  # Max 8 chunks

    prompt = f"""You are a precise factual AI assistant. Your task is to answer the user's question using ONLY the provided PDF content.

PDF CONTENT:
{context_text}

USER QUESTION:
{question}

ANSWER RULES:
1. Extract the answer directly from the PDF content above
2. DO NOT infer, guess, or add outside knowledge
3. If the exact answer is not in the PDF, respond with "I don't know"
4. Quote or cite the relevant PDF text when possible
5. Be concise and factual

ANSWER:"""

    try:
        available_models = list(genai.list_models())
    except Exception as e:
        return f"❌ Failed to list models: {e}"

    if not available_models:
        return "❌ No models available from Google API"

    for model_info in available_models:
        try:
            model_name = model_info.name
            if "generate" not in str(model_info).lower():
                continue

            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(
                    prompt, 
                    safety_settings={},
                    generation_config={"temperature": 0.1, "top_p": 0.8}  # Lower temperature for factuality
                )
                answer = getattr(response, "text", str(response)).strip()
                return answer if answer else "❌ Model returned empty response"
            except Exception as model_error:
                continue

        except Exception:
            continue

    return (
        f"❌ No working model found. Available: {', '.join(m.name for m in available_models[:5])}"
    )

# ----------------------------
# SESSION STATE
# ----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "index" not in st.session_state:
    st.session_state.index = None
if "chat" not in st.session_state:
    st.session_state.chat = []

# ----------------------------
# UI
# ----------------------------
st.title("🧠 PaperBrain")

# RESET BUTTON
if st.button("🔄 Reset"):
    st.session_state.clear()
    st.rerun()

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ----------------------------
# PROCESS PDF
# ----------------------------
if uploaded_file:

    # FILE SIZE CHECK
    if len(uploaded_file.getvalue()) > MAX_FILE_SIZE:
        st.error("❌ File too large (Max 5MB)")
        st.stop()

    if st.button("Process PDF"):

        with st.spinner("Processing..."):

            text = extract_text(uploaded_file.read())

            if not text.strip():
                st.error("❌ No readable text in PDF")
                st.stop()

            # LIMIT TEXT SIZE
            if len(text) > MAX_TEXT_LENGTH:
                st.warning("Large PDF detected, truncating...")
                text = text[:MAX_TEXT_LENGTH]

            chunks = split_text(text)

            index = build_index(chunks)

            st.session_state.chunks = chunks
            st.session_state.index = index

            st.success("✅ PDF Ready!")

# ----------------------------
# CHAT
# ----------------------------
if st.session_state.chunks:

    question = st.text_input("Ask something about the PDF")

    if st.button("Ask") and question:

        with st.spinner("Thinking..."):

            context = search(
                question,
                st.session_state.chunks,
                st.session_state.index
            )

            answer = get_answer(question, context)

            st.session_state.chat.append((question, answer))

# ----------------------------
# CHAT HISTORY
# ----------------------------
for q, a in reversed(st.session_state.chat):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**PaperBrain:** {a}")
    st.markdown("---")
