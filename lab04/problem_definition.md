\# Lab 4: Problem Definition



\## Problem



The Adult dataset contains demographic and employment-related information that can be used to predict whether a person's annual income is above $50,000. Before training a machine learning model, the data must be checked for quality issues that could affect the reliability of predictions.



\## Objectives



The objectives of this lab are to validate the dataset schema and numerical ranges, identify missing values and duplicate records, examine income labels, and prepare the data for machine learning. The preprocessing pipeline must avoid data leakage by calculating imputation and outlier-clipping values from the training data only. Duplicate rows must be removed from the training data without removing test records.



The lab also requires creating the `capital\_net` and `hours\_per\_age` features, comparing Logistic Regression models with and without these features, auditing a sample of labels, and recording a hash of the processed dataset for reproducibility.



\## Evaluation



The models will be evaluated using accuracy and the F1 score for the income-above-$50,000 class. The results will help assess whether the engineered features improve prediction performance and highlight limitations caused by class imbalance or inconsistent labels.

