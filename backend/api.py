from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os, uuid
import logging

from scripts.run_system import AcademicSystem


# =====================================================
# LOGGING (THESIS-GRADE DEBUGGING)
# =====================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-api")


app = FastAPI(title="Thesis RAG System")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# =====================================================
# SYSTEM INIT (SAFE SINGLETON)
# =====================================================
system = None

def get_system():
    global system

    if system is None:
        logger.info("⚡ Initializing AcademicSystem...")
        system = AcademicSystem("models/qwen.gguf")

    return system


# =====================================================
# MODELS
# =====================================================
class Query(BaseModel):
    question: str
    use_rag: bool = True


class URLRequest(BaseModel):
    url: str


class BloomRequest(BaseModel):
    question: str


# =====================================================
# HEALTH
# =====================================================
@app.get("/")
def health():
    return {
        "status": "running",
        "service": "RAG System"
    }


# =====================================================
# ASK (WITH DEBUG INFO)
# =====================================================
@app.post("/ask")
def ask(req: Query):

    if not req.question.strip():
        return {"error": "Empty question"}

    try:
        sys = get_system()
        result = sys.ask(req.question, req.use_rag)

        # 🔥 DEBUG LOGGING
        logger.info(f"QUESTION: {req.question}")
        logger.info(f"RESPONSE: {result}")

        return result

    except Exception as e:
        logger.exception("/ask failed")
        return {
            "error": str(e)
        }


@app.post("/classify/bloom")
def classify_bloom(req: BloomRequest):

    if not req.question.strip():
        return {"error": "Empty question"}

    try:
        sys = get_system()
        result = sys.classify_bloom_question(req.question)
        logger.info(f"BLOOM QUESTION: {req.question}")
        logger.info(f"BLOOM RESULT: {result}")
        return result
    except Exception as e:
        logger.exception("/classify/bloom failed")
        return {"error": str(e)}


# =====================================================
# URL INGEST (FIXED DEBUG VERSION)
# =====================================================
@app.post("/ingest/url")
def ingest_url(req: URLRequest):

    if not req.url.strip():
        return {"error": "Empty URL"}

    try:
        sys = get_system()

        logger.info(f"🌐 Scraping URL: {req.url}")

        before = len(sys.rag.chunks)

        sys.add_url(req.url)

        after = len(sys.rag.chunks)

        return {
            "status": "success",
            "url": req.url,
            "chunks_added": after - before
        }

    except Exception as e:
        logger.exception("FULL ERROR TRACE")
        return {
            "error": str(e)
        }

# =====================================================
# PDF INGEST
# =====================================================
@app.post("/ingest/pdf")
async def ingest_pdf(file: UploadFile = File(...)):

    try:
        sys = get_system()

        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.pdf")

        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"📄 PDF saved: {file_path}")

        sys.add_pdf(file_path)

        return {
            "status": "success",
            "message": "PDF ingested successfully",
            "file": file.filename
        }

    except Exception as e:
        logger.exception("/ingest/pdf failed")

        return {
            "error": str(e)
        }


@app.post("/classify/bloom/pdf")
async def classify_bloom_pdf(file: UploadFile = File(...)):

    try:
        sys = get_system()
        file_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.pdf")
        content = await file.read()

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"PDF saved for Bloom classification: {file_path}")
        result = sys.classify_bloom_pdf(file_path)

        return {
            "status": "success",
            "file": file.filename,
            "classifications": result
        }
    except Exception as e:
        logger.exception("/classify/bloom/pdf failed")
        return {"error": str(e)}
