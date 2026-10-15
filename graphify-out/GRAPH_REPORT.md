# Graph Report - veg-size-compare  (2026-10-15)

## Corpus Check
- Corpus is ~43,078 words - fits in a single context window. You may not need a graph.

## Summary
- 580 nodes · 1120 edges · 33 communities (31 shown, 2 thin omitted)
- Extraction: 91% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 103 edges (avg confidence: 0.88)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- Feature & Classifier Models
- Synthetic Data Generation
- Design Doc & Mask Annotations
- Raw-Image Baselines (HOG/CNN)
- Evaluation Metrics & Reporting
- Model Loading & Prediction
- API & Demo UI
- Calibration & Measurement
- Segmentation & Detection
- Dataset IO & Mask IoU
- Results Plotting
- Config & Feature Building
- Pipeline Stage Flow
- Item Grouping by Crop
- Overlay Prediction Demo
- Pipeline CLI Entry
- Weight Proxy Regressor
- Pairwise Comparison
- Pairwise Pair Scoring
- Pipeline Output & Overlay
- Baseline Accuracy Comparison
- White Balance & Render Tests
- Stress Case Robustness
- Weight Regression Accuracy
- Heavier-Item Selection Task
- Package Root Init
- Project Root

## God Nodes (most connected - your core abstractions)
1. `DataPaths` - 38 edges
2. `Config` - 27 edges
3. `evaluate()` - 25 edges
4. `Pipeline` - 25 edges
5. `run()` - 20 edges
6. `segment()` - 19 edges
7. `build()` - 18 edges
8. `load_patches()` - 17 edges
9. `Calibration` - 15 edges
10. `white_balance()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `White-balance to the sheet before colour features` --semantically_similar_to--> `No annotation, no calibration marker`  [INFERRED] [semantically similar]
  README.md → docs/PIPELINE.md
- `get_pipeline()` --uses--> `Config`  [INFERRED]
  app/api.py → src/config.py
- `example_images()` --uses--> `DataPaths`  [INFERRED]
  app/gradio_app.py → src/config.py
- `load_patches()` --uses--> `Calibration`  [INFERRED]
  baselines/common.py → src/calibrate.py
- `load_patches()` --uses--> `DataPaths`  [INFERRED]
  baselines/common.py → src/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Online inference pipeline flow (segment -> features -> classify -> weight -> decide)** — docs_pipeline_classical_segmentation, docs_pipeline_features, docs_pipeline_crop_classifier, docs_pipeline_weight_regressor, docs_pipeline_pairwise_decision, docs_pipeline_size_small_big [EXTRACTED 0.90]
- **Reference masks forming the segmentation IoU benchmark** — readme_reference_masks, readme_segmentation_iou, data_reference_masks_mixed_0000_mask, data_reference_masks_mixed_0001_mask, data_reference_masks_mixed_0002_mask, data_reference_masks_mixed_0003_mask, data_reference_masks_pair_0000_mask, data_reference_masks_pair_0001_mask, data_reference_masks_pair_0002_mask, data_reference_masks_pair_0003_mask, data_reference_masks_pair_0004_mask, data_reference_masks_pair_0005_mask [INFERRED 0.85]
- **Headline results comparison across four approaches** — reports_results_table_headline, reports_results_table_volume_proxy, reports_results_table_svm, reports_results_table_hog_svm_baseline, reports_results_table_cnn_baseline [EXTRACTED 0.90]
- **Trained classic ML models deployed to online inference stages** — docs_data_flow_train_models, docs_data_flow_crop_gate, docs_data_flow_estimate_weight, docs_data_flow_decide [EXTRACTED 1.00]

## Communities (33 total, 2 thin omitted)

### Community 0 - "Feature & Classifier Models"
Cohesion: 0.05
Nodes (59): Shared setup for the raw-image baselines (docs/PIPELINE.md, Stage 6). They test…, Series, feature_matrix(), predict(), ndarray, Stage C: which crop is each item (label + probability), so items can be grouped…, Stack the crop-classifier features of several items into one matrix. Args:…, Name the crop of each item, with the classifier's confidence. Args: bundle: The… (+51 more)

### Community 1 - "Synthetic Data Generation"
Cohesion: 0.06
Nodes (54): fixture, ndarray, Measure the ruler's length in pixels. The ruler is taken to be the largest dark…, ruler_length_px(), items(), A dozen synthetic items (four per crop) for rendering test photos. Returns: The…, Automatic ruler detection recovers the simulated rig's scale within 1%., test_auto_ruler_calibration_recovers_scale() (+46 more)

### Community 2 - "Design Doc & Mask Annotations"
Cohesion: 0.06
Nodes (53): Mixed sample 0000 mask annotation, Mixed sample 0001 mask annotation, Mixed sample 0002 mask annotation, Mixed sample 0003 mask annotation, Pair sample 0000 mask annotation, Pair sample 0001 mask annotation, Pair sample 0002 mask annotation, Pair sample 0003 mask annotation (+45 more)

### Community 3 - "Raw-Image Baselines (HOG/CNN)"
Cohesion: 0.07
Nodes (44): main(), make_fit_predict(), Baseline: a fine-tuned ImageNet CNN on raw image patches -> SMALL / BIG.…, Command-line entry point: run the CNN baseline., Build the CNN baseline's training-and-prediction function. PyTorch is imported…, leave_one_item_out(), load_patches(), DataFrame (+36 more)

### Community 4 - "Evaluation Metrics & Reporting"
Cohesion: 0.07
Nodes (32): bin_label(), _col(), evaluate(), headline(), latency(), main(), pairwise_summary(), print_summary() (+24 more)

### Community 5 - "Model Loading & Prediction"
Cohesion: 0.06
Nodes (26): from_points(), main(), Path, Command-line entry point: calibrate from a ruler photo, or check an existing…, Read a calibration from the JSON file written by `python -m src.calibrate`.…, Scale from two points a known distance apart, e.g. two ruler ticks. Args: p1:…, Load the calibration and the trained models. Args: cfg: Settings to use.…, predict_rows() (+18 more)

### Community 6 - "API & Demo UI"
Cohesion: 0.10
Nodes (23): compare(), get_pipeline(), health(), HTTP API for the sorting station: POST a photo, get per-crop groups with…, The shared pipeline, created on the first request and reused after that.…, Report that the service is up and which models it has loaded. Returns: The…, Analyse an uploaded photo and return the per-crop groups. Args: file: Top-down…, build() (+15 more)

### Community 7 - "Calibration & Measurement"
Cohesion: 0.10
Nodes (22): Calibration, One-time calibration for the fixed rig: a ruler photo -> mm per pixel. The…, Scale of the fixed rig, as measured from one ruler photo. Attributes:…, Work out mm/px for a particular photo and check it came from the same rig. A…, match_by_order(), Pair detections with weighed items by their left-to-right order. Both the…, extract_features(), ndarray (+14 more)

### Community 8 - "Segmentation & Detection"
Cohesion: 0.13
Nodes (21): Detection, fill_holes(), foreground(), _kernel(), mask_box(), ndarray, Stage A: classical segmentation of vegetables on a plain white sheet. No…, Elliptical morphology kernel of a given physical size. Args: mm: Kernel… (+13 more)

### Community 9 - "Dataset IO & Mask IoU"
Cohesion: 0.20
Nodes (16): Stage 7: every metric from Section 7 of the design, all leave-one-item-out.…, Compare the segmentation with the hand-traced outlines. Args: dp: Dataset…, reference_iou(), imread(), mask_iou(), match_masks(), polygon_mask(), ndarray (+8 more)

### Community 10 - "Results Plotting"
Cohesion: 0.19
Nodes (17): main(), plot_pairwise(), plot_small_big(), plot_stress(), plot_weights(), DataFrame, Path, Charts for the results write-up, drawn from the files eval/evaluate.py writes.… (+9 more)

### Community 11 - "Config & Feature Building"
Cohesion: 0.18
Nodes (14): Config, Paths and tunable parameters shared by every stage., Tunable settings for the online pipeline. The defaults match the design…, load_images(), Load the photo index, with each photo's item IDs split into a list. Missing…, A small synthetic dataset with features built and models trained, shared by the…, tiny(), build() (+6 more)

### Community 12 - "Pipeline Stage Flow"
Cohesion: 0.21
Nodes (14): Baseline: raw-image classifier, C. Crop type + same-type gate, E. Decide (two modes), D. Estimate weight (g), Evaluation (leave-one-item-out), B. Features in mm, Fixed rig + ruler photo (mm per pixel), Input image (+6 more)

### Community 13 - "Item Grouping by Crop"
Cohesion: 0.15
Nodes (11): group_items(), Grouping, Stage C: group items by crop and set aside the ones that cannot be used. Every…, Which items go into which crop group, and which are left out. Attributes:…, The overall status: "ok" if at least one group exists, otherwise "no_items"., A short human-readable version of `status`., Sort items into crop groups, setting aside the ones that cannot be used. An…, Items are grouped by crop, with low-confidence and cut-off items set aside. (+3 more)

### Community 14 - "Overlay Prediction Demo"
Cohesion: 0.28
Nodes (13): Pixel-to-mm Calibration (Inferred), Chamagadda Detections (3 Instances), Classification Confidence Scores (0.98-0.99), Colored Contour Outlines (Blue Tomato / Orange Chamagadda), Crop Type Classification (Tomato vs Chamagadda), Per-Crop Diff Percentage Summary (tomato 77% / chamagadda 613%), Mixed 0009 Prediction Overlay (Tomato & Chamagadda), Per-Object Instance Segmentation (+5 more)

### Community 15 - "Pipeline CLI Entry"
Cohesion: 0.20
Nodes (11): ArgumentParser, Namespace, add_config_args(), build_config(), _label(), main(), End-to-end inference at the sorting station. image -> A. segment -> B. features…, Draw a block of text on a dark, semi-transparent box. The box is placed just… (+3 more)

### Community 16 - "Weight Proxy Regressor"
Cohesion: 0.17
Nodes (9): BaseEstimator, RegressorMixin, ProxyRegressor, Weight as a straight multiple of a volume proxy: `weight = a * proxy (+ b)`.…, Choose between a line through the origin and one with an offset. Args:…, Fit the slope (and offset) from proxy values and measured weights. Args: X:…, Estimate weights from proxy values. Args: X: Feature matrix whose first column…, The proxy model recovers a known slope through the origin. (+1 more)

### Community 17 - "Pairwise Comparison"
Cohesion: 0.18
Nodes (11): _pct(), rank_group(), Stage E, pairwise mode: rank one crop group and compare neighbours with a…, Turn a relative difference into a rounded percentage for the JSON output. Args:…, Rank the items of one crop group and compare each neighbouring pair. Items are…, Two items reproduce the design document's BIGGER / SMALLER example (59%)., Items within the margin are both SIMILAR., Three items reproduce the design's chamagadda ladder (SIMILAR, then BIGGER). (+3 more)

### Community 18 - "Pairwise Pair Scoring"
Cohesion: 0.24
Nodes (11): item_pairs(), pair_rows(), DataFrame, Score every same-crop pair of items, not just those photographed together. Each…, Score every same-crop pair of items within each photo. A pair only counts as…, pair_verdict(), Compare one item with another using the SIMILAR margin. Args: a: Size of the…, How much bigger the larger value is, relative to the smaller one. For example,… (+3 more)

### Community 19 - "Pipeline Output & Overlay"
Cohesion: 0.24
Nodes (8): draw_overlay(), PipelineOutput, ndarray, Analyse one photo from the rig. Steps: white-balance to the sheet, segment,…, Draw outlines and labels for every item, coloured by crop group. Uncertain and…, Everything one run of the pipeline produced. Attributes: result: The JSON-ready…, Draw the result on top of the photo. Returns: A copy of the photo with…, Find a detection's record in the output, wherever it ended up. Args: det_index:…

### Community 20 - "Baseline Accuracy Comparison"
Cohesion: 0.38
Nodes (10): CNN, raw patches (76%), Engineered Features Outperform Raw-Pixel Baselines, Small/Big Accuracy Comparison (leave-one-item-out), HOG + SVM, raw patches (51%), Leave-One-Item-Out Cross Validation, Logistic Regression, features (88%), Small vs Big Classification Accuracy, SVM, features (88%) (+2 more)

### Community 21 - "White Balance & Render Tests"
Cohesion: 0.22
Nodes (10): Scale each colour channel so the white sheet comes out neutral grey. The sheet…, white_balance(), Render a test photo of the given items. Args: items: The `items` fixture. ids:…, render(), All three crops are found, with masks matching their true outlines (IoU > 0.95)., Detections come back ordered left to right., A dim, tinted sheet comes out neutral at the target grey., test_detections_are_left_to_right() (+2 more)

### Community 22 - "Stress Case Robustness"
Cohesion: 0.39
Nodes (9): Dim Light Stress Case, Stress Cases Figure (Touching, Tilt, Dim), Pairwise Accuracy Metric, Pipeline Robustness to Edge Cases, Synthetic Data Limitation (Code Path Only), Tilt Weight Estimation Degradation (12.3% MAPE), Tilted Camera Stress Case, Touching Items Stress Case (+1 more)

### Community 23 - "Weight Regression Accuracy"
Cohesion: 0.44
Nodes (9): Chamagadda proxy model (MAPE 6.3%), Predicted vs Measured Weight Parity Plot (leave-one-item-out, singles), Kheera proxy model (MAPE 2.5%), Leave-One-Item-Out Evaluation (singles), MAPE (Mean Absolute Percentage Error), Predicted-vs-True Parity Plot (y=x perfect line), Synthetic Data Validation (code path, not real-world accuracy), Tomato proxy model (MAPE 5.1%) (+1 more)

### Community 24 - "Heavier-Item Selection Task"
Cohesion: 0.57
Nodes (7): Heavier-Item Selection Accuracy, Chance Baseline (50%), Picks the Heavier Item: Accuracy by True Weight Difference, All Same-Crop Item Pairs (Difference), Pick the Heavier Item Task, Pairs in the Same Photo (Pairwise), True Weight Difference Bins

## Ambiguous Edges - Review These
- `Predicted-vs-True Parity Plot (y=x perfect line)` → `Chamagadda proxy model (MAPE 6.3%)`  [AMBIGUOUS]
  reports/weight_pred_vs_true.png · relation: conceptually_related_to

## Knowledge Gaps
- **14 isolated node(s):** `veg-size-compare`, `Task: compare same-kind vegetables by size`, `Classical segmentation (HSV/Lab + Otsu)`, `Limitations and assumptions`, `Features + volume proxy result (MAPE 5.0%, pairwise 98.0%, Small/Big 93.0%)` (+9 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Predicted-vs-True Parity Plot (y=x perfect line)` and `Chamagadda proxy model (MAPE 6.3%)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `DataPaths` connect `Evaluation Metrics & Reporting` to `Feature & Classifier Models`, `Synthetic Data Generation`, `Raw-Image Baselines (HOG/CNN)`, `API & Demo UI`, `Dataset IO & Mask IoU`, `Config & Feature Building`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `Config` connect `Config & Feature Building` to `Raw-Image Baselines (HOG/CNN)`, `Evaluation Metrics & Reporting`, `Model Loading & Prediction`, `API & Demo UI`, `Calibration & Measurement`, `Segmentation & Detection`, `Dataset IO & Mask IoU`, `Pipeline CLI Entry`, `White Balance & Render Tests`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Why does `Pipeline` connect `API & Demo UI` to `Raw-Image Baselines (HOG/CNN)`, `Evaluation Metrics & Reporting`, `Model Loading & Prediction`, `Calibration & Measurement`, `Segmentation & Detection`, `Dataset IO & Mask IoU`, `Config & Feature Building`, `Pipeline CLI Entry`, `Pipeline Output & Overlay`?**
  _High betweenness centrality (0.040) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `DataPaths` (e.g. with `example_images()` and `load_patches()`) actually correct?**
  _`DataPaths` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `Config` (e.g. with `get_pipeline()` and `evaluate()`) actually correct?**
  _`Config` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `evaluate()` (e.g. with `Calibration` and `Config`) actually correct?**
  _`evaluate()` has 3 INFERRED edges - model-reasoned connections that need verification._