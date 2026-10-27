\# Lab 4: Adult Dataset Data Card



\## 1. Dataset Overview



The Adult dataset contains 48,842 rows and 15 columns. It is used to predict whether a person's annual income is above $50,000. The target column is `class`, with labels `<=50K` and `>50K`.



\## 2. Source and License



\- \*\*Source:\*\* Adult dataset accessed through OpenML.

\- \*\*Dataset version:\*\* 2, as specified in the project configuration.

\- \*\*Original source:\*\* UCI Adult dataset.

\- \*\*License:\*\* The exact applicable license should be verified from the dataset's official metadata before redistribution.



\## 3. Data Quality



I checked column names, data types, missing values, duplicate rows, numerical ranges, and income labels.



Missing values were found in:

\- `workclass`: 2,799

\- `occupation`: 2,809

\- `native-country`: 857



The original dataset contains 52 duplicate rows. There are no missing income labels. The validation script also checks selected numerical ranges and reports invalid income labels.



The label audit identified five groups of identical feature values with conflicting income labels. A sample of 50 rows was saved for inspection.



\## 4. Data Preparation



The data was split into 70% training data and 30% test data using random state 42.



Missing numerical values were filled using the median, and missing categorical values were filled using the most frequent value. Numerical outliers were clipped using the 1st and 99th percentiles calculated from training data. Exact duplicate rows were removed from the training data only.



Categorical features were encoded using one-hot encoding, and numerical features were standardized. The preprocessing object was saved in `results/preprocessor.joblib`.



\## 5. Feature Engineering



Two features were created:



\- `capital\_net`: capital gain minus capital loss.

\- `hours\_per\_age`: hours worked per week divided by age minus 16.



\## 6. Model Results



Logistic Regression was used to predict income.



| Metric | Without new features | With new features |

|---|---:|---:|

| Accuracy | 85.40% | 85.78% |

| F1 score for `>50K` | 66.01% | 66.96% |



Adding the new features slightly improved both accuracy and F1 score.



\## 7. Known Biases and Limitations



The dataset contains more examples of people earning `<=50K` than `>50K`. The model identifies the `<=50K` class more effectively than the `>50K` class. The dataset may also reflect historical and demographic biases. Conflicting labels for identical feature values may affect model reliability.



The model's predictions should not be treated as definitive judgments about individuals.



\## 8. Privacy Considerations



The dataset contains demographic and employment-related information. It should be handled responsibly and used for appropriate educational or analytical purposes. Model outputs should not be used as the sole basis for consequential decisions about individuals.



\## 9. Reproducibility and Artifacts



\- Random state: `42`

\- Train/test split: `70/30`

\- Trained model: `results/adult\_model.joblib`

\- Preprocessor: `results/preprocessor.joblib`

\- Label audit sample: `results/label\_audit\_50.csv`

\- Processed dataset hash: `results/processed\_data\_hash.txt`



The pipeline can be run from the repository root using:



` .venv\\Scripts\\python.exe lab04\\src\\run\_pipeline.py `



