# Manuscript Results Plan

This document maps the current experiment outputs to manuscript-ready tables and figures.
The intended framing is:

- Main study: interpretable gait-affect mechanism analysis on AVI gait videos.
- Probe study: domain-mismatch comparison on MOV upper-body clips.
- Main claim: emotion-related discrimination is carried primarily by speed, rhythmicity, and selected motion-structure cues, not by pure classifier complexity.

## Recommended Main-Text Tables

### Table 1. Dataset Summary

Purpose:
- Establish the scope of the two datasets and the analytical role of each one.

Use these sources:
- AVI sample count and subject count from outputs/experiments/avi_main/summary.json
- MOV sample count from outputs/experiments/mov_probe/run_20260525_104547/summary.json

Suggested columns:
- Dataset
- Role in study
- Samples
- Subjects
- Modality
- Feature source

Suggested rows:
- AVI main: gait-only point-light videos, main mechanism study, 447 samples, 37 subjects, pose-derived interpretable features
- MOV probe: upper-body emotion clips, domain-shift probe, 60 samples, subject count from labels/features table if needed, contour-proxy interpretable features

Caption point:
- Emphasize that MOV is a probe-only comparison and is not pooled with the AVI main experiment.

### Table 2. AVI Main Performance

Purpose:
- Report the predictive baseline for the mechanism analysis without overselling raw classification.

Use these sources:
- outputs/experiments/avi_main/summary.json
- outputs/experiments/avi_main/report.md

Suggested columns:
- Accuracy
- Macro F1
- Balanced accuracy
- Number of features
- Excluded nuisance families

Current values:
- Accuracy: 0.4049
- Macro F1: 0.3744
- Balanced accuracy: 0.3733
- Features: 19
- Excluded families: quality, capture

Caption point:
- State that nuisance families were removed to isolate interpretable movement cues rather than clip length or extraction quality artifacts.

### Table 3. Top Interpretable AVI Features

Purpose:
- Show the core feature-level mechanism claim in compact tabular form.

Use these sources:
- outputs/experiments/avi_main/permutation_importance.csv
- outputs/experiments/avi_main/feature_effects.csv

Suggested columns:
- Feature
- Family
- Permutation importance mean
- Dominant emotion contrast
- Cohen's d
- Interpretation

Recommended rows:
- hcf_mean_joint_speed
- hcf_temporal_smoothness
- hcf_std_joint_speed
- kin_body_height_mean
- hcf_x_std

Interpretation examples:
- hcf_mean_joint_speed: global movement speed / arousal proxy
- hcf_temporal_smoothness: temporal regularity / rhythmicity proxy
- hcf_std_joint_speed: movement variability proxy
- kin_body_height_mean: body extension / posture proxy
- hcf_x_std: spatial spread in lateral motion structure

## Recommended Main-Text Figures

### Figure 1. AVI Mechanism Overview

Purpose:
- Provide the headline mechanism result in one panel figure.

Panels:
- Panel A: bar chart of top 8 permutation importance features from outputs/experiments/avi_main/permutation_importance.csv
- Panel B: color-coded by feature_family

Use these columns:
- feature
- feature_family
- importance_mean
- importance_std

Message to support:
- Speed and rhythmicity cues are the strongest contributors after nuisance-feature removal.

### Figure 2. Class-Wise Feature Effects

Purpose:
- Show how specific emotions differ along interpretable gait dimensions.

Preferred design:
- Heatmap of mean_gap or Cohen's d using outputs/experiments/avi_main/feature_effects.csv

Recommended feature subset:
- hcf_mean_joint_speed
- hcf_temporal_smoothness
- kin_lateral_sway
- kin_vertical_bounce
- kin_body_height_mean

Use these columns:
- feature
- label
- cohen_d or mean_gap

Message to support:
- Happy and anger trend toward higher movement speed and temporal variation, while fear and sad trend lower on those same dimensions.

### Figure 3. Family Ablation Plot

Purpose:
- Support the argument that interpretable movement families carry nontrivial signal even when isolated.

Use this source:
- outputs/experiments/avi_main/family_ablation.csv

Preferred design:
- Horizontal bar chart of macro_f1 by feature_set

Use these columns:
- feature_set
- macro_f1
- balanced_accuracy

Message to support:
- No single family explains the entire effect, but speed and rhythmicity are the strongest compact families.

### Figure 4. AVI vs MOV Domain-Shift Probe

Purpose:
- Show why upper-body clips should not be merged with the gait main study.

Use this source:
- outputs/experiments/mov_probe/run_20260525_104547/domain_shift.csv

Preferred design:
- Horizontal bar chart of standardized_gap for the top shared features

Recommended features to show:
- hcf_y_std
- hcf_x_std
- kin_body_height_mean
- kin_body_width_mean
- hcf_std_joint_speed
- hcf_temporal_smoothness
- hcf_mean_joint_speed

Use these columns:
- feature
- feature_family
- standardized_gap

Message to support:
- Upper-body clips differ sharply from gait videos in motion structure, posture scale, and speed-related cues, confirming domain mismatch.

## Recommended Supplementary Items

### Supplementary Table S1. Full Classification Report

Use this source:
- outputs/experiments/avi_main/summary.json

Include:
- Precision, recall, F1, support for anger, fear, happy, neutral, sad

### Supplementary Table S2. Full Group Statistics

Use this source:
- outputs/experiments/avi_main/group_stats.csv

Purpose:
- Provide full descriptive statistics for each feature by emotion label.

### Supplementary Figure S1. Confusion Matrix

Use this source:
- outputs/experiments/avi_main/summary.json

Purpose:
- Show which affective states are most confusable without making this the main contribution figure.

### Supplementary Figure S2. Full Domain Shift Ranking

Use this source:
- outputs/experiments/mov_probe/run_20260525_104547/domain_shift.csv

Purpose:
- Provide the full AVI-vs-MOV shared-feature ranking beyond the top main-text subset.

## Suggested Results Section Order

### 1. Main AVI performance

Lead with:
- The model achieved modest classification performance on subject-wise AVI evaluation, indicating that the task is nontrivial and suitable for mechanism-focused rather than benchmark-focused interpretation.

Support with:
- Table 2
- Supplementary Figure S1

### 2. Feature-level mechanism findings

Lead with:
- After excluding nuisance feature families, the dominant predictive cues were mean joint speed, temporal smoothness, speed variability, and posture-related body extent.

Support with:
- Figure 1
- Figure 2
- Table 3

### 3. Family-level interpretation

Lead with:
- Speed and rhythmicity retained the strongest compact explanatory signal among the interpretable feature families.

Support with:
- Figure 3

### 4. Domain-mismatch probe

Lead with:
- The MOV upper-body clips showed large shifts relative to the AVI gait distribution, especially in motion-structure and posture features, supporting their use as a probe rather than as pooled training data.

Support with:
- Figure 4

## What To Keep Out Of The Main Claim

- Do not center the main text on raw accuracy.
- Do not frame MOV as external validation of gait emotion recognition.
- Do not present contour-proxy MOV features as equivalent to true pose-based gait features.
- Do not include quality or capture variables in the main mechanism figures.

## Minimal Figure Build Checklist

- Figure 1: permutation importance top 8, sorted descending, family-colored
- Figure 2: emotion-by-feature heatmap using Cohen's d
- Figure 3: family ablation bar chart using macro F1
- Figure 4: AVI vs MOV domain-shift bar chart using standardized_gap
- Table 2: summary performance metrics copied from summary.json
- Table 3: top feature rows merged from permutation_importance.csv and feature_effects.csv