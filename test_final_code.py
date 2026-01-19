import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import timedelta

# For embeddings:
from sentence_transformers import SentenceTransformer

# -----------------------------
# App Config (must be FIRST)
# -----------------------------
st.set_page_config(
    page_title="Video Transcript Semantic Search (Dynamic Embeddings)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Helper: ms → HH:MM:SS
# -----------------------------
def ms_to_hms(ms):
    td = timedelta(milliseconds=int(ms))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return (
        f"{hours:02}:{minutes:02}:{seconds:02}"
        if hours > 0
        else f"{minutes:02}:{seconds:02}"
    )

# -----------------------------
# Load Data from CSV and generate embeddings dynamically
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_data_and_embed(csv_path="enhanced_transcript_llm_style.csv"):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Load embedding model (you can replace with any other model)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Generate embeddings for the text column
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    embeddings = np.array(embeddings)
    
    return df, embeddings

# -----------------------------
# Semantic Search Function
# -----------------------------
@st.cache_data(show_spinner=False)
def semantic_search(embeddings, query_embedding, top_k=5):
    scores = cosine_similarity(embeddings, [query_embedding]).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    return top_idx, scores[top_idx]

# -----------------------------
# UI Header
# -----------------------------
st.title("🎥 Video Transcript Semantic Search (Dynamic Embeddings)")
st.caption("Offline, fast, no LLM, no precomputed embeddings required")

# -----------------------------
# Load CSV Data & Embeddings
# -----------------------------
with st.spinner("Loading transcript data and generating embeddings..."):
    df, embeddings = load_data_and_embed()

st.success(f"Loaded {len(df)} transcript chunks")

# -----------------------------
# User Input (debounced)
# -----------------------------
query = st.text_input(
    "Ask a question about the video:",
    placeholder="Example: Explain relational operators in Java",
)

if not query.strip():
    st.info("Enter a question to retrieve relevant transcript sections.")
    st.stop()

# -----------------------------
# Load embedding model for query
# -----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

query_embedding = model.encode([query])[0]

# -----------------------------
# Search
# -----------------------------
with st.spinner("Searching transcript..."):
    top_idx, top_scores = semantic_search(embeddings, query_embedding, top_k=5)

top_df = df.iloc[top_idx].copy()
top_df["similarity"] = top_scores

# -----------------------------
# Combine top results into one answer
# -----------------------------
# Assuming start_time and end_time columns are in seconds (adjust if milliseconds)
combined_start = ms_to_hms(top_df["start_time"].min() * 1000)
combined_end = ms_to_hms(top_df["end_time"].max() * 1000)

combined_text = "\n\n".join(top_df["text"].tolist())

# Display combined answer
st.subheader(f"🔎 Combined Relevant Transcript Sections ({combined_start} – {combined_end})")
st.write(combined_text)

# -----------------------------
# Download combined text
# -----------------------------
download_text = f"[{combined_start} - {combined_end}]\n{combined_text}"

st.download_button(
    "📥 Download Combined Transcript",
    download_text,
    file_name="combined_retrieved_transcript.txt",
    use_container_width=True
)
