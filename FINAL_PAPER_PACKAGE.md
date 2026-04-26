# Final Paper Package

## Final Title

A Lightweight Multi-Modal Tiny LLM Framework for Privacy-Aware Academic Assistance in University Environments

## Final Datasets

- Bloom training:
  [models/external_datasets/exam_combined_dataset.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/exam_combined_dataset.csv)
- External scholarly stress-test evaluation:
  [models/external_datasets/scienceqa_val.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/scienceqa_val.csv)
  and
  [models/external_datasets/scienceqa_test.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/scienceqa_test.csv)
- Primary document-grounded internal evaluation dataset:
  [evaluation/robot_pdf_eval_dataset.json](C:/Users/tahiy/PycharmProjects/Thesis/evaluation/robot_pdf_eval_dataset.json)

## Final Saved Results

- Main internal evaluation and ablations:
  [models/experiment_runs/run_20260426_044227/results.json](C:/Users/tahiy/PycharmProjects/Thesis/models/experiment_runs/run_20260426_044227/results.json)
- Bloom training report:
  [models/bloom_training_report.json](C:/Users/tahiy/PycharmProjects/Thesis/models/bloom_training_report.json)
- Bloom benchmark report:
  [models/bloom_benchmark_report.json](C:/Users/tahiy/PycharmProjects/Thesis/models/bloom_benchmark_report.json)
- ScienceQA safety-enabled evaluation:
  [models/scienceqa_eval_results.json](C:/Users/tahiy/PycharmProjects/Thesis/models/scienceqa_eval_results.json)
- ScienceQA no-rejection evaluation:
  [models/scienceqa_eval_no_rejection.json](C:/Users/tahiy/PycharmProjects/Thesis/models/scienceqa_eval_no_rejection.json)
- Final plots:
  [models/final_plots/ablation_summary.png](C:/Users/tahiy/PycharmProjects/Thesis/models/final_plots/ablation_summary.png)
  [models/final_plots/privacy_quality_curve.png](C:/Users/tahiy/PycharmProjects/Thesis/models/final_plots/privacy_quality_curve.png)
  [models/final_plots/main_metrics.png](C:/Users/tahiy/PycharmProjects/Thesis/models/final_plots/main_metrics.png)

## Final Quantitative Results

### Main Internal Evaluation

Primary benchmark notes:
- 18 document-grounded questions created from the uploaded firefighting and gas detection robot proposal PDF
- focused on operational and methodology content instead of front-matter or privacy-adjacent metadata
- intended to match the real use case of uploaded university project material

| Metric | Value |
|---|---:|
| Latency | 9.3265 |
| Answer Similarity | 0.5219 |
| Faithfulness | 0.2505 |
| Hallucination | 0.7495 |
| Precision@k | 0.9074 |
| Retrieval Redundancy | 0.4891 |
| Mean Chunk Privacy | 0.2143 |
| Privacy Leakage | 0.0000 |
| Uncertainty Rate | 0.0556 |
| Rejection Rate | 0.0000 |
| Confidence | 0.4426 |
| Bloom Accuracy (internal question set) | 0.3333 |
| Bloom Confidence | 0.2574 |
| Bloom Uncertainty | 0.9589 |
| Attack Success Rate | 0.0000 |

### Ablation Summary

| Setting | Answer Similarity | Faithfulness | Precision@k | Redundancy | Mean Chunk Privacy | Rejection |
|---|---:|---:|---:|---:|---:|---:|
| Full system | 0.4909 | 0.2578 | 0.9074 | 0.4891 | 0.2143 | 0.0000 |
| No privacy | 0.5276 | 0.2621 | 0.9074 | 0.5126 | 0.2141 | 0.0000 |
| No diversity | 0.5102 | 0.2501 | 0.9167 | 0.7817 | 0.2151 | 0.0000 |
| No rejection | 0.4797 | 0.2519 | 0.9074 | 0.4891 | 0.2143 | 0.0000 |

### Privacy-Quality Curve

| Lambda | Answer Similarity | Mean Chunk Privacy |
|---|---:|---:|
| 0.0 | 0.5011 | 0.2141 |
| 0.2 | 0.4813 | 0.2139 |
| 0.5 | 0.4841 | 0.2024 |

### External Bloom Training

| Source Dataset | Validation Accuracy |
|---|---:|
| Exam Question Datasets (combined set) | 0.8119 |

This trained Bloom model passes the runtime quality threshold and is used by the live system.

### ScienceQA External Stress Test

Safety-enabled run:

| Metric | Value |
|---|---:|
| Samples | 40 |
| Answer Similarity | 0.0090 |
| Faithfulness | 0.0271 |
| Mean Chunk Privacy | 0.0756 |
| Privacy Leakage | 0.0000 |
| Rejection Rate | 0.9500 |

No-rejection run:

| Metric | Value |
|---|---:|
| Samples | 40 |
| Answer Similarity | 0.0945 |
| Faithfulness | 0.1087 |
| Mean Chunk Privacy | 0.0756 |
| Privacy Leakage | 0.0150 |
| Rejection Rate | 0.0000 |

## Paper-Ready Interpretation

- The external Bloom dataset is strongly suitable for training.
- The ScienceQA split works as a hard scholarly stress test rather than the main benchmark.
- The main thesis results should be anchored in the document-grounded robot PDF benchmark and its ablation study.
- The internal Bloom accuracy on natural project questions is lower than the external validation score, so Bloom quality should be reported primarily with the external labeled dataset and secondarily with internal qualitative examples.
- Privacy-aware retrieval and diversity-aware selection behave as expected, with diversity showing the clearest improvement by reducing redundancy substantially relative to the no-diversity condition.
- The safety layer now protects against leakage-like attack prompts without blocking normal robot-PDF question answering.

## Paper-Ready Claim Boundaries

- Use `privacy-aware` or `privacy-protective`, not formally `privacy-preserving`.
- Describe uncertainty as retrieval-confidence-based with a practical rejection layer, not as a fully calibrated probabilistic guarantee.
- Report Bloom classifier quality with the external exam-question dataset as the primary quantitative Bloom result.
