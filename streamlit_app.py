import streamlit as st
import requests

API_URL = "http://127.0.0.1:8002"

st.set_page_config(page_title="RAG Academic System", layout="centered")

st.title("📚 High-Accuracy RAG Academic System")


# =====================================================
# ASK SECTION
# =====================================================
st.header("Ask a Question")

question = st.text_input("Enter your question")

if st.button("Ask"):

    if not question.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Thinking... (RAG + LLM running)"):

            try:
                res = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "use_rag": True},
                    timeout=120
                )

                data = res.json()

                # -------------------------
                # SAFE OUTPUT HANDLING
                # -------------------------
                if "answer" in data:
                    st.subheader("Answer")
                    st.write(data["answer"])

                    st.caption(f"Context Used: {data.get('context_used', False)}")
                    st.caption(f"Retrieval Score: {data.get('retrieval_score', 0.0)}")

                elif "error" in data:
                    st.error(data["error"])

                else:
                    st.error("Invalid response from backend")

            except Exception as e:
                st.error(f"Request failed: {str(e)}")


# =====================================================
# INGEST URL SECTION
# =====================================================
st.header("Ingest URL into Knowledge Base")

url = st.text_input("Enter URL")

if st.button("Ingest URL"):

    if not url.strip():
        st.warning("Please enter a URL")
    else:
        with st.spinner("Ingesting URL..."):

            try:
                res = requests.post(
                    f"{API_URL}/ingest/url",
                    json={"url": url},
                    timeout=120
                )

                data = res.json()
                st.write(data)

            except Exception as e:
                st.error(f"Ingestion failed: {str(e)}")