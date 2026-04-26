# Project Structure Review

This review reflects the cleaned project after removal of obsolete duplicate modules.

## Top Level

- [app.py](C:/Users/tahiy/PycharmProjects/Thesis/app.py)
  Simple CLI for loading sources and asking questions.
- [streamlit_app.py](C:/Users/tahiy/PycharmProjects/Thesis/streamlit_app.py)
  Streamlit frontend that talks to the FastAPI backend.
- [test_rag.py](C:/Users/tahiy/PycharmProjects/Thesis/test_rag.py)
  Minimal local smoke test for querying the system.
- [requirements.txt](C:/Users/tahiy/PycharmProjects/Thesis/requirements.txt)
  Python dependencies for the project.

## `backend/`

- [backend/api.py](C:/Users/tahiy/PycharmProjects/Thesis/backend/api.py)
  FastAPI backend exposing question answering and URL/PDF ingestion endpoints.

## `src/`

- [src/rag_engine.py](C:/Users/tahiy/PycharmProjects/Thesis/src/rag_engine.py)
  Core retrieval engine. Handles chunk storage, FAISS search, privacy-aware reranking, diversity filtering, Bloom attachment, and ablation controls.
- [src/bloom_ldl_runtime.py](C:/Users/tahiy/PycharmProjects/Thesis/src/bloom_ldl_runtime.py)
  Active Bloom runtime. Loads the trained lightweight Bloom model when good enough, otherwise falls back to heuristic distributions. Also provides uncertainty and reject helpers.
- [src/embed.py](C:/Users/tahiy/PycharmProjects/Thesis/src/embed.py)
  Embedding wrapper. Uses local sentence-transformer embeddings when cached, otherwise falls back to hashing-based offline-safe embeddings.
- [src/llm.py](C:/Users/tahiy/PycharmProjects/Thesis/src/llm.py)
  Local quantized LLM loading and constrained answer generation with `llama.cpp`.
- [src/chunker.py](C:/Users/tahiy/PycharmProjects/Thesis/src/chunker.py)
  Text chunking with overlap for retrieval.
- [src/image_pipeline.py](C:/Users/tahiy/PycharmProjects/Thesis/src/image_pipeline.py)
  OCR + caption fusion into unified text for image ingestion.
- [src/document.py](C:/Users/tahiy/PycharmProjects/Thesis/src/document.py)
  Lightweight document container class.
- [src/summarizer.py](C:/Users/tahiy/PycharmProjects/Thesis/src/summarizer.py)
  Local summarization helper for manual or auxiliary use.

## `src/loaders/`

- [src/loaders/pdf_loader.py](C:/Users/tahiy/PycharmProjects/Thesis/src/loaders/pdf_loader.py)
  PDF text extraction with fallback across available libraries.
- [src/loaders/web_loader.py](C:/Users/tahiy/PycharmProjects/Thesis/src/loaders/web_loader.py)
  HTML fetching and content cleaning for webpage ingestion.
- [src/loaders/multimodal_loader.py](C:/Users/tahiy/PycharmProjects/Thesis/src/loaders/multimodal_loader.py)
  OCR text extraction for images.
- [src/loaders/image_loader.py](C:/Users/tahiy/PycharmProjects/Thesis/src/loaders/image_loader.py)
  Lazy image captioning with local-files-only loading and safe fallback text if weights are unavailable.

## `src/privacy/`

- [src/privacy/privacy.py](C:/Users/tahiy/PycharmProjects/Thesis/src/privacy/privacy.py)
  Sensitive keyword definitions and basic privacy matching helpers.
- [src/privacy/privacy_model.py](C:/Users/tahiy/PycharmProjects/Thesis/src/privacy/privacy_model.py)
  Lightweight TF-IDF + Logistic Regression privacy classifier plus bootstrap builder for default runtime use.

## `scripts/`

- [scripts/run_system.py](C:/Users/tahiy/PycharmProjects/Thesis/scripts/run_system.py)
  Main system wrapper. Integrates RAG, LLM, confidence scoring, Bloom outputs, and safe rejection.
- [scripts/train_lightweight_bloom.py](C:/Users/tahiy/PycharmProjects/Thesis/scripts/train_lightweight_bloom.py)
  Trains the active lightweight Bloom model. Prefers the external exam-question dataset when available.
- [scripts/benchmark_bloom_models.py](C:/Users/tahiy/PycharmProjects/Thesis/scripts/benchmark_bloom_models.py)
  Compares several lightweight Bloom classifiers across feature settings for reproducibility.
- [scripts/run_scienceqa_eval.py](C:/Users/tahiy/PycharmProjects/Thesis/scripts/run_scienceqa_eval.py)
  Runs the external ScienceQA scholarly stress-test evaluation.

## `evaluation/`

- [evaluation/metrics.py](C:/Users/tahiy/PycharmProjects/Thesis/evaluation/metrics.py)
  Evaluation metrics: latency, answer similarity, faithfulness, hallucination, privacy leakage, redundancy, confidence, rejection, and summaries.
- [evaluation/run_experiments.py](C:/Users/tahiy/PycharmProjects/Thesis/evaluation/run_experiments.py)
  Main internal evaluation and ablation runner used for the final paper results.

## `models/`

- [models/qwen.gguf](C:/Users/tahiy/PycharmProjects/Thesis/models/qwen.gguf)
  Quantized local generation model.
- [models/bloom_ldl.pkl](C:/Users/tahiy/PycharmProjects/Thesis/models/bloom_ldl.pkl)
  Trained lightweight Bloom classifier artifact.
- [models/bloom_training_report.json](C:/Users/tahiy/PycharmProjects/Thesis/models/bloom_training_report.json)
  Final Bloom training metrics on the external exam-question dataset.
- [models/bloom_benchmark_report.json](C:/Users/tahiy/PycharmProjects/Thesis/models/bloom_benchmark_report.json)
  Benchmark comparison of lightweight Bloom model variants.
- [models/scienceqa_eval_results.json](C:/Users/tahiy/PycharmProjects/Thesis/models/scienceqa_eval_results.json)
  External ScienceQA safety-enabled evaluation results.
- [models/scienceqa_eval_no_rejection.json](C:/Users/tahiy/PycharmProjects/Thesis/models/scienceqa_eval_no_rejection.json)
  External ScienceQA no-rejection comparison results.
- [models/experiment_runs/run_20260426_031621/results.json](C:/Users/tahiy/PycharmProjects/Thesis/models/experiment_runs/run_20260426_031621/results.json)
  Final internal evaluation and ablation results for the paper.

## `models/external_datasets/`

- [models/external_datasets/exam_combined_dataset.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/exam_combined_dataset.csv)
  External Bloom-labeled exam-question training dataset.
- [models/external_datasets/scienceqa_val.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/scienceqa_val.csv)
  External scholarly QA validation split used for stress testing.
- [models/external_datasets/scienceqa_test.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/scienceqa_test.csv)
  External scholarly QA test split used for stress testing.

## Removed As Obsolete

The following were removed because they were duplicates, superseded prototypes, or no longer part of the live pipeline:

- old Bloom classifier helpers
- old heuristic LDL duplicate files
- old standalone multimodal/vector-store retrieval prototypes
- unused privacy helper duplicates
- old CORAL / TPU training scripts
- superseded intermediate result files and metadata helper files

## Notes

- Some locked `.pyc` cache files could not be deleted by the environment, but they are not part of the source project and do not affect the final thesis package.
- Two top-level legacy files remain locked by the environment:
  `rag_store.pkl` and `testing.txt`
  They are not used by the final pipeline.
