> **SYNTHETIC DATA: these numbers validate the code, not real-world accuracy.**

| Approach | Weight MAPE | Pairwise acc. | Small/Big acc. |
|---|---|---|---|
| Raw-image HOG + SVM (baseline) | — | — | 51.2% |
| Raw-image CNN (baseline) | — | — | 76.0% |
| Features + volume proxy (this work) | 5.0% | 98.0% | 93.0% |
| Features + SVM (this work) | 5.5% | 98.0% | 88.4% |

All numbers are leave-one-item-out. Weight MAPE and Small/Big use the singles; pairwise uses
same-crop pairs in the pairs and mixed photos. The volume-proxy row uses `weight = a * proxy`
and a threshold on that estimate; the SVM row uses the multi-feature ridge regressor and an
RBF SVM on the mm features plus its weight estimate.
