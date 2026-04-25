## Main Summary

| Metric | Value |
|---|---:|
| Latency | 6.7704 |
| Answer Similarity | 0.1545 |
| Faithfulness | 0.1467 |
| Hallucination | 0.8533 |
| Precision@k | 0.6000 |
| Retrieval Redundancy | 0.3178 |
| Mean Chunk Privacy | 0.1711 |
| Privacy Leakage | 0.0000 |
| Rejection Rate | 0.6000 |
| Bloom Confidence | 0.2780 |
| Bloom Uncertainty | 0.9664 |
| Attack Success Rate | 0.0000 |

## Ablation Summary

| Setting | Answer Similarity | Faithfulness | Precision@k | Redundancy | Mean Chunk Privacy | Rejection |
|---|---:|---:|---:|---:|---:|---:|
| Full system | 0.1634 | 0.1479 | 0.6000 | 0.3178 | 0.1711 | 0.6000 |
| No privacy | 0.1602 | 0.1394 | 0.9333 | 0.5406 | 0.2137 | 0.6000 |
| No diversity | 0.1661 | 0.1610 | 0.6000 | 0.4579 | 0.1723 | 0.6000 |
| No rejection | 0.2283 | 0.3051 | 0.6000 | 0.3178 | 0.1711 | 0.0000 |

## Privacy-Quality Curve

| Lambda | Answer Similarity | Mean Chunk Privacy |
|---|---:|---:|
| 0.0 | 0.1625 | 0.2139 |
| 0.2 | 0.1567 | 0.2142 |
| 0.5 | 0.1706 | 0.1711 |

## Bloom Benchmark

| Feature Set | Model | Accuracy |
|---|---|---:|
| Question only | Logistic Regression | 0.1689 |
| Question only | SGD Classifier | 0.1714 |
| Question only | Linear SVM (char) | 0.1654 |
| Question + metadata | Logistic Regression | 0.1681 |
| Question + metadata | SGD Classifier | 0.1681 |
| Question + metadata | Linear SVM (char) | 0.1681 |
| Full prompt | Logistic Regression | 0.1672 |
| Full prompt | SGD Classifier | 0.1671 |
| Full prompt | Linear SVM (char) | 0.1693 |

## Interpretation

- Disabling privacy increased both retrieval redundancy and mean chunk privacy risk.
- Disabling diversity increased redundancy, as expected.
- The rejection layer reduced answer generation on weak or high-risk cases, but also lowered answer similarity because many answers were intentionally withheld.
- The Bloom dataset appears weakly learnable from lightweight text features; benchmark accuracies remain near 0.17 across several stronger models, so the live system should treat Bloom as a heuristic interpretability aid rather than a reliable learned classifier.

## External Dataset Upgrades

- External Bloom training dataset:
  [exam_combined_dataset.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/exam_combined_dataset.csv)
- External scholarly QA evaluation:
  [scienceqa_val.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/scienceqa_val.csv)
  and
  [scienceqa_test.csv](C:/Users/tahiy/PycharmProjects/Thesis/models/external_datasets/scienceqa_test.csv)

## External Bloom Training Results

| Source Dataset | Validation Accuracy |
|---|---:|
| Exam Question Datasets (Figshare combined set) | 0.8119 |

The live system now uses the trained Bloom model because it exceeds the runtime quality threshold.

## ScienceQA External Evaluation

Safety-enabled run:
[scienceqa_eval_results.json](C:/Users/tahiy/PycharmProjects/Thesis/models/scienceqa_eval_results.json)

| Metric | Value |
|---|---:|
| Samples | 40 |
| Answer Similarity | 0.0090 |
| Faithfulness | 0.0271 |
| Mean Chunk Privacy | 0.0756 |
| Privacy Leakage | 0.0000 |
| Rejection Rate | 0.9500 |

No-rejection run:
[scienceqa_eval_no_rejection.json](C:/Users/tahiy/PycharmProjects/Thesis/models/scienceqa_eval_no_rejection.json)

| Metric | Value |
|---|---:|
| Samples | 40 |
| Answer Similarity | 0.0945 |
| Faithfulness | 0.1087 |
| Mean Chunk Privacy | 0.0756 |
| Privacy Leakage | 0.0150 |
| Rejection Rate | 0.0000 |

These external results suggest that the downloaded ScienceQA split is useful as a hard out-of-domain scholarly stress test, but not as the strongest primary benchmark for this system because many question strings are noisy or weakly formed.
