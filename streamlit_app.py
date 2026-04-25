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
                    st.caption(f"Confidence: {data.get('confidence', 0.0):.3f}")
                    st.caption(f"Rejected: {data.get('rejected', False)}")
                    st.caption(f"Bloom Level: {data.get('bloom_level', 'N/A')}")
                    st.caption(f"Bloom Mode: {data.get('bloom_mode', 'N/A')}")
                    st.caption(f"Bloom Uncertainty: {data.get('bloom_uncertainty', 0.0):.3f}")

                    if data.get("rejection_reasons"):
                        st.warning("Rejection reasons: " + ", ".join(data["rejection_reasons"]))

                    chunks = data.get("chunks", [])
                    if chunks:
                        with st.expander("Retrieval Trace"):
                            for i, chunk in enumerate(chunks, start=1):
                                st.markdown(
                                    f"**Chunk {i}** | score={chunk.get('score', 0.0):.3f} | "
                                    f"semantic={chunk.get('semantic_score', 0.0):.3f} | "
                                    f"privacy={chunk.get('privacy_score', 0.0):.3f}"
                                )
                                st.write(chunk.get("text", ""))

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
