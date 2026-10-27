\# Lab 4: Adult Dataset Data Card



\## 1. Dataset



I used the Adult dataset from OpenML. It contains 48,842 rows and 15 columns. The goal is to predict a person's income group.



\## 2. Data Quality



I checked column names, data types, missing values, duplicates, numerical ranges, and income labels. Missing values were found in `workclass` (2,799), `occupation` (2,809), and `native-country` (857). The original dataset contains 52 duplicate rows. There are no missing income labels.



The label audit also found five groups of identical feature values with conflicting income labels.



\## 3. Data Preparation



I split the data into 70% training data and 30% test data using random state 42.



Missing numerical values were filled using the median, and missing categorical values were filled using the most frequent value. Numerical outliers were clipped using the 1st and 99th percentiles calculated from training data. Exact duplicate rows were removed from the training data only.



I used one-hot encoding for categorical features and standard scaling for numerical features. The preprocessing object was saved in `results/preprocessor.joblib`.



\## 4. Feature Engineering



I created two new features:



\- `capital\_net`: capital gain minus capital loss.

\- `hours\_per\_age`: hours worked per week divided by age minus 16.



\## 5. Model Results



I used Logistic Regression to predict income.



| Metric | Without new features | With new features |

|---|---:|---:|

| Accuracy | 85.40% | 85.78% |

| F1 score for `>50K` | 66.01% | 66.96% |



The model with the new features performed slightly better on both metrics.



\## 6. Conclusion



Adding the two new features slightly improved the model results. The model performs better at predicting the `<=50K` income group than the `>50K` group.



\## 7. Reproducibility



The random state is 42, and the train/test split is 70/30. The trained model and preprocessing object were saved in the results folder. The processed dataset SHA-256 hash was saved in `results/processed\_data\_hash.txt`.



A 50-row sample for label auditing was saved in `results/label\_audit\_50.csv`.



\## 8. Limitations



The dataset has more examples in the `<=50K` group than in the `>50K` group. The data may also contain historical bias. The conflicting labels identified during the audit may limit prediction quality.



