\# Lab 4: Data Pipeline and Data Quality



\## 1. Goal



The goal of this lab was to prepare the Adult dataset, check its quality, and build a machine learning pipeline to predict income.



\## 2. Method



I downloaded the Adult dataset from OpenML. It contains 48,842 rows and 15 columns.



I checked column names, data types, missing values, duplicate rows, numerical ranges, and label quality. I split the data into 70% training data and 30% test data using random state 42.



Missing numerical values were filled using the median, and missing categorical values were filled using the most frequent value. Numerical outliers were clipped using the 1st and 99th percentiles calculated from the training data. Exact duplicate rows were removed from the training data only.



I created two features: `capital\_net` (capital gain minus capital loss) and `hours\_per\_age` (hours per week divided by age minus 16). I trained a Logistic Regression model and compared it with a model without the new features.



\## 3. Results



| Measurement | Result |

|---|---:|

| Original dataset rows | 48,842 |

| Original dataset columns | 15 |

| Training rows before duplicate removal | 34,189 |

| Training rows after duplicate removal | 34,165 |

| Test rows | 14,653 |

| Simple model accuracy | 85.40% |

| Simple model F1 score for `>50K` | 66.01% |

| Model with new features accuracy | 85.78% |

| Model with new features F1 score for `>50K` | 66.96% |



The preprocessing object and trained model were saved in the results folder. The processed dataset SHA-256 hash was also saved.



The label audit found no missing income labels and identified five groups of identical feature values with conflicting labels. A sample of 50 rows was saved for inspection.



\## 4. Conclusions



1\. The dataset contains missing values and duplicate rows that require careful handling.

2\. The model with the two new features performed slightly better than the simple model.

3\. The model with new features achieved 85.78% accuracy and an F1 score of 66.96% for the `>50K` class.



\## 5. Limitations



The dataset has more examples in the `<=50K` group than in the `>50K` group. The model performs better at identifying the `<=50K` class than the `>50K` class. Therefore, accuracy alone does not fully describe its performance.



