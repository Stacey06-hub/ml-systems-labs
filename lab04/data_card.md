\# Lab 4: Adult Dataset Data Card



\## 1. Dataset



The Adult dataset was downloaded from OpenML. It has 48,842 rows and 15 columns. The goal is to predict a person's income group.



\## 2. Data Quality



The dataset has missing values in workclass, occupation, and native-country. It also has 52 duplicate rows.



\## 3. Data Preparation



I split the data into training and test sets. I used 80% for training and 20% for testing. I used random state 42 to get the same split each time.



I filled missing numerical values with the median. I filled missing text values with the most common value. I changed text features into numbers and scaled the numerical features.



\## 4. Model Results



I used Logistic Regression to predict income.



\- Accuracy: 85.24%

\- F1 score for the >50K class: 65.62%



\## 5. Conclusion



The model correctly predicted about 85% of the test examples. It performed better for the <=50K class than for the >50K class.



\## 6. Limitations



The dataset has more examples in the <=50K class. Missing values and possible bias in the data may affect the results.



