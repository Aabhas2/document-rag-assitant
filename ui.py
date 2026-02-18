import streamlit as st
import requests

API = "http://127.0.0.1:8000"

st.title("Mini Document RAG Assistant")

# ── Namespace selector (shared across all sections) ─────────
namespace = st.text_input("Workspace / Namespace", value="movies")

# ── 1) Ingest ────────────────────────────────────────────────
st.subheader("1) Ingest a document")
uploaded = st.file_uploader("Upload a .txt file", type=["txt"])
if uploaded is not None:
    if st.button("Ingest"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        r = requests.post(f"{API}/ingest", files=files, params={"namespace": namespace})
        st.write(r.json())

# ── 2) Ask ───────────────────────────────────────────────────
st.subheader("2) Ask a question")
q = st.text_input("Question")
top_k = st.slider("Top-k", 1, 10, 5)
if st.button("Ask") and q.strip():
    r = requests.post(
        f"{API}/ask",
        json={"question": q, "namespace": namespace, "top_k": top_k},
    )
    out = r.json()
    st.markdown("### Answer")
    st.write(out.get("answer"))
    st.markdown("### Citations")
    st.write(out.get("citations"))

# ── 3) Reset workspace ──────────────────────────────────────
st.subheader("3) Reset workspace")
if st.button(f"Reset '{namespace}' workspace"):
    r = requests.post(f"{API}/reset", params={"namespace": namespace})
    st.write(r.json())

# ──────────────────────────────────────────────────────────────
# MANUAL TEST PLAN
# 1. Start API:  uvicorn app.main:app --reload
# 2. Start UI:   streamlit run ui.py
#
# Test A – movies namespace:
#   - Set namespace = "movies"
#   - Click "Reset 'movies' workspace"
#   - Upload a movies .txt file → Ingest
#   - Ask a movies-related question → should return movie content
#
# Test B – mit namespace:
#   - Set namespace = "mit"
#   - Upload mit_license.txt → Ingest
#   - Ask "What does the MIT license allow?" → should return license text
#
# Test C – cross-contamination check:
#   - Set namespace = "movies", ask "What does the MIT license allow?"
#     → should return NO relevant context (or unrelated movie text)
#   - Set namespace = "mit", ask a movies question
#     → should return NO relevant context (or unrelated license text)
# ──────────────────────────────────────────────────────────────
