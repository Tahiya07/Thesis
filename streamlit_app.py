import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt

from scripts.run_system import AcademicSystem


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Thesis RAG Dashboard",
    layout="wide"
)

st.title("📊 Lightweight RAG Thesis Evaluation Dashboard")
st.caption("Interactive evaluation + live system demo")


# =========================================================
# LOAD SYSTEM (LAZY - CLOUD SAFE)
# =========================================================
@st.cache_resource
def load_system():
    return AcademicSystem("models/qwen.gguf")


system = load_system()


# =========================================================
# LOAD RESULTS
# =========================================================
@st.cache_data
def load_results():
    return pd.read_csv("evaluation/results.csv")


df = None

try:
    df = load_results()
except:
    st.warning("⚠️ No evaluation results found. Run run_eval.py first.")


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
page = st.sidebar.radio(
    "Navigation",
    ["📈 Live Demo", "📊 Evaluation Dashboard"]
)


# =========================================================
# 1. LIVE DEMO PAGE
# =========================================================
if page == "📈 Live Demo":

    st.subheader("🧠 Ask Your Model (Live Inference)")

    question = st.text_input("Enter question")

    use_rag = st.toggle("Enable RAG", value=True)

    if st.button("Generate Answer") and question:

        start = time.time()
        result = system.ask(question, use_rag=use_rag)
        end = time.time()

        st.markdown("### 📌 Answer")
        st.write(result["answer"])

        st.metric("⏱️ Latency", f"{end - start:.2f}s")
        st.write("📚 RAG Used:", result["context_used"])


# =========================================================
# 2. EVALUATION DASHBOARD
# =========================================================
elif page == "📊 Evaluation Dashboard":

    if df is None:
        st.stop()

    st.subheader("📊 Model Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Avg BLEU (RAG)", f"{df['bleu_rag'].mean():.3f}")
    col2.metric("Avg BLEU (No RAG)", f"{df['bleu_no_rag'].mean():.3f}")

    col3.metric("Avg Latency (RAG)", f"{df['latency_rag'].mean():.3f}s")
    col4.metric("Avg Latency (No RAG)", f"{df['latency_no_rag'].mean():.3f}s")


    # =====================================================
    # BLEU GRAPH
    # =====================================================
    st.subheader("📈 BLEU Score Comparison")

    fig, ax = plt.subplots()
    ax.plot(df["bleu_rag"], label="RAG")
    ax.plot(df["bleu_no_rag"], label="No RAG")
    ax.set_title("BLEU Score per Sample")
    ax.legend()
    st.pyplot(fig)


    # =====================================================
    # HALLUCINATION
    # =====================================================
    st.subheader("🧠 Hallucination (LLM Judge)")

    fig, ax = plt.subplots()
    ax.plot(df["llm_hall_rag"], label="RAG")
    ax.plot(df["llm_hall_no_rag"], label="No RAG")
    ax.set_title("Hallucination Score (Lower is Better)")
    ax.legend()
    st.pyplot(fig)


    # =====================================================
    # LATENCY
    # =====================================================
    st.subheader("⏱️ Latency Comparison")

    fig, ax = plt.subplots()
    ax.plot(df["latency_rag"], label="RAG")
    ax.plot(df["latency_no_rag"], label="No RAG")
    ax.set_title("Inference Latency")
    ax.legend()
    st.pyplot(fig)


    # =====================================================
    # FAITHFULNESS (IF EXISTS)
    # =====================================================
    if "faith_rag" in df.columns:

        st.subheader("🎯 Faithfulness Score")

        fig, ax = plt.subplots()
        ax.plot(df["faith_rag"], label="RAG")
        ax.plot(df["faith_no_rag"], label="No RAG")
        ax.set_title("Faithfulness Comparison")
        ax.legend()
        st.pyplot(fig)


    # =====================================================
    # RETRIEVAL QUALITY
    # =====================================================
    if "precision@k" in df.columns:

        st.subheader("📚 Retrieval Quality (RAG Only)")

        fig, ax = plt.subplots()
        ax.plot(df["precision@k"], label="Precision@K")
        ax.plot(df["recall@k"], label="Recall@K")
        ax.set_title("Retrieval Performance")
        ax.legend()
        st.pyplot(fig)


    # =====================================================
    # SUMMARY BAR CHART
    # =====================================================
    st.subheader("📊 Overall System Summary")

    metrics = {
        "BLEU RAG": df["bleu_rag"].mean(),
        "BLEU No RAG": df["bleu_no_rag"].mean(),
        "Hallucination RAG": df["llm_hall_rag"].mean(),
        "Hallucination No RAG": df["llm_hall_no_rag"].mean(),
        "Latency RAG": df["latency_rag"].mean(),
        "Latency No RAG": df["latency_no_rag"].mean(),
    }

    fig, ax = plt.subplots()
    ax.bar(metrics.keys(), metrics.values())
    ax.set_title("Final Performance Comparison")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)


    # =====================================================
    # RAW DATA
    # =====================================================
    with st.expander("📄 Raw Results Table"):
        st.dataframe(df)