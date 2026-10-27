\# Lab 4: Data Pipeline and Data Quality



\## 1. Goal



The goal of this lab was to prepare the Adult dataset and build a machine learning pipeline. I also checked data quality and tested model performance.



\## 2. Method



I downloaded the Adult dataset from OpenML. It contains 48,842 rows and 15 columns.



I checked the column names, data types, missing values, duplicates, and numerical ranges. I split the data into training and test sets using an 80/20 split and random state 42.



I filled missing numerical values with the median and missing categorical values with the most common value. I scaled numerical features and used one-hot encoding for categorical features.



I trained a Logistic Regression model. I also added an age group feature and compared the results.



\## 3. Results



| Measurement | Result |

|---|---:|

| Dataset rows | 48,842 |

| Dataset columns | 15 |

| Duplicate rows found | 52 |

| Missing income labels | 0 |

| Training rows | 39,073 |

| Test rows | 9,769 |

| Simple model accuracy | 85.24% |

| Simple model F1 score | 65.62% |

| Model with age group accuracy | 85.42% |

| Model with age group F1 score | 66.34% |



The dataset hash was saved in `lab04/results/dataset\_hash.txt`. I also saved a 50-row label audit.



\## 4. Conclusions



1\. The dataset had missing values in three columns, but no missing income labels.

2\. The model achieved 85.24% accuracy and an F1 score of 65.62% for the income group above $50K.

3\. Adding the age group feature slightly improved accuracy and F1 score.



\## 5. Limitations



The income groups are not balanced. The model performed better on the <=50K group than on the >50K group. More testing is needed to understand the effect of data quality and other features.



