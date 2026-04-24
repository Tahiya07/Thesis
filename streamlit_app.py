import streamlit as st
import time
import numpy as np

from scripts.run_system import AcademicSystem
from evaluation.metrics import RAGEvaluator
from src.loaders.pdf_loader import load_pdf_text
from src.loaders.web_loader import load_webpage


# =========================================================
# SETUP
# =========================================================
st.set_page_config(page_title="Live RAG Dashboard", layout="wide")

st.title("📊 Live RAG Evaluation System")
st.caption("Upload data → Ask questions → See real-time evaluation")


# =========================================================
# INIT SYSTEM
# =========================================================
@st.cache_resource
def load_system():
    return AcademicSystem("models/qwen.gguf")


system = load_system()
evaluator = RAGEvaluator()


# =========================================================
# SIDEBAR: DATA INGESTION
# =========================================================
st.sidebar.header("📥 Data Ingestion")

pdf_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
url_input = st.sidebar.text_input("Enter Web URL")


# =========================================================
# INGEST DATA
# =========================================================
if st.sidebar.button("📚 Load Data into RAG"):

    if pdf_file:
        with open("temp.pdf", "wb") as f:
            f.write(pdf_file.read())

        system.add_pdf("temp.pdf")
        st.sidebar.success("PDF loaded")

    if url_input:
        system.add_url(url_input)
        st.sidebar.success("URL loaded")

    if not pdf_file and not url_input:
        st.sidebar.warning("No input provided")


# =========================================================
# MAIN QUERY SECTION
# =========================================================
st.subheader("🧠 Ask Question")

question = st.text_input("Enter your question")
use_rag = st.toggle("Enable RAG", value=True)


if st.button("Run Inference"):

    if not question:
        st.warning("Please enter a question")
        st.stop()

    # =====================================================
    # STEP 1: RETRIEVE CONTEXT
    # =====================================================
    raw_context = system.rag.retrieve(question) if use_rag else []

    context_text = "\n".join(raw_context)

    # =====================================================
    # STEP 2: RAG GATING SCORE
    # =====================================================
    if context_text:
        retrieval_score = system.retrieval_score(context_text, question)
    else:
        retrieval_score = 0.0


    # =====================================================
    # STEP 3: ASK MODEL
    # =====================================================
    start = time.time()

    result = system.ask(question, use_rag=use_rag)

    latency = time.time() - start


    # =====================================================
    # STEP 4: COMPUTE LIVE METRICS
    # =====================================================

    answer = result["answer"]

    bleu = evaluator.compute_bleu(question, answer)
    faith = evaluator.faithfulness(context_text, answer) if context_text else 0.0
    hallucination = evaluator.hallucination_score(context_text, answer) if context_text else 1.0


    # =====================================================
    # OUTPUT
    # =====================================================
    st.markdown("## 📌 Answer")
    st.write(answer)

    st.markdown("## 📊 Live Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("⏱️ Latency", f"{latency:.2f}s")
    col2.metric("📚 Retrieval Score", f"{retrieval_score:.3f}")
    col3.metric("🎯 Faithfulness", f"{faith:.3f}")
    col4.metric("⚠️ Hallucination", f"{hallucination:.3f}")

    st.metric("🧪 BLEU", f"{bleu:.3f}")


    # =====================================================
    # DEBUG VIEW
    # =====================================================
    with st.expander("📄 Retrieved Context"):
        st.write(context_text if context_text else "No context retrieved")