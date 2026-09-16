# Practical 1 report: preparing biological data for machine learning

**Student:** [name]  
**Date:** [date]  
**Practical:** 1 - Data to ML-ready format

## How to use this template

Replace the text in square brackets with your own answers. Include figures and tables where requested. Every result should be accompanied by one or two sentences explaining what it means biologically and what decision it supports.
## 1. Executive summary

In 150-250 words, summarize:

- the biological question 
- the dataset or datasets used 
- the main preparation decisions 
- the most important data-quality issue 
- the final train/test design 
- one limitation that affects interpretation.

[Write summary here]

## 2. Data-preparation readiness

Complete this section before describing modelling results.

## Scientific purpose and provenance

- [ ] Is the biological question and prediction task stated clearly?
- [ ] Is the unit of observation clear (for example, patient, tissue, cell, mouse, or technical measurement)?
- [ ] Have I recorded the data version, processing history, and transformations already applied?

## Table shape and identifiers

- [ ] Is the table tidy: one observation per row, one variable per column, and one value per cell?
- [ ] If the original assay has features in rows and samples in columns, have I transposed it into the format expected by the analysis tool?
- [ ] Are row names, column names, and identifiers clear, unique, stable, and free from accidental index columns?
- [ ] Can every feature and observation be traced back to its source?
- [ ] Are the feature matrix and metadata aligned by an explicit identifier rather than by row order?

## Types, values, and measurement meaning

- [ ] Are numerical, categorical, ordinal, text, date, and identifier columns correctly classified?
- [ ] Are units, scales, encodings, reference categories, and detection limits understood?
- [ ] Have I checked impossible values, inconsistent labels, unexpected categories, constant features, and extreme values?
- [ ] Are normalization, scaling, log transformation, and batch-correction steps documented?

## Missing values

- [ ] Have I detected missing values using the dataset's actual missing-value representations, not only `NaN`?
- [ ] Have I measured missingness by row, column, group, and relevant batch or time variable?
- [ ] Do I know whether missingness is structural, technical, or plausibly related to the biology or target?
- [ ] Have I justified whether to retain, remove, or impute missing values?
- [ ] If imputing, will the imputer be fitted on training data only and then applied to validation/test data?

## Replicates, dependence, and grouping

- [ ] Have I checked for exact duplicates and near-duplicate observations?
- [ ] Are observations biological replicates, technical replicates, repeated measurements, pseudo-replicates, or independent samples?
- [ ] Is the independent biological unit identified by a grouping variable?
- [ ] Will all related observations and their derivatives remain in the same train/validation/test split?
- [ ] If I created augmented or synthetic observations, is their provenance recorded and reproducible?

## Target, leakage, and study design

- [ ] Is the target defined before modelling, with its classes or values checked for balance and validity?
- [ ] Have I excluded identifiers, post-outcome variables, duplicated target information, and other leakage sources from the features?
- [ ] Have I identified batch, site, plate, subject, cohort, treatment, sex, age, time, or other possible confounders?
- [ ] Have I checked whether the target is associated with a technical or grouping variable?
- [ ] Does the planned split represent the biological generalization question?

## Reproducibility and decision record

- [ ] Are random seeds, software versions, file names, and input/output shapes recorded?
- [ ] Are all preprocessing steps placed in a reproducible pipeline?
- [ ] Have I recorded unresolved data-quality concerns and the decisions made about them?
- [ ] Can another person explain what each row, column, feature, target, and grouping variable means?

### 3.1 Data presentation

Describe:
- what one row represents 
- what one biological mouse represents 
- `MouseID` and derived `mouse_id` 
- `Genotype`, `Treatment`, `Behavior`, and `class` 
- the prediction target selected.

**Answers:**  
[Write answers here.]

Explain the structure of the dataset

### 3.2 Tidy-data assessment and EDA

Report:
- rows and columns: `[shape]` 
- unique mice: `[number]` 
- measurements per mouse: `[number]` 
- categories and counts for Genotype, Treatment, Behavior, and class: `[summary]` 
- whether the table is tidy: `[answer and evidence]`.

Include at least three plots. For each, state the biological question it addresses.

### 3.3 Diagnostics

Report checks for:
- missingness in proteins and metadata: `[result]` 
- duplicate rows or duplicate measurement identifiers: `[result]` 
- expected 15 measurements per mouse: `[result]` 
- unusual categories or inconsistent labels: `[result]` 
- non-finite, negative, or implausible expression values: `[result]` 
- highly correlated or redundant protein variables: `[result]` 
- predictors that are deterministic parts of the target: `[result]`.

### 3.4 Preparation and feature roles

Selected supervised-learning question:

Missing-value decision:


### 3.5 Split strategy
- Independent biological unit: `[mouse]`
- Grouping variable: `[mouse_id]`
- Split method: `[method]`

### 3.6 Confounding and distribution structure

Investigate associations between the target and:
- treatment 
- behavior 
- genotype 
- composite class 
- number of measurements per mouse.

## References and reproducibility
Together with this report you need to provide
- notebook 
- data files
- generated plots