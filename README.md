# MachineLearning-Codes
This repository contains multiple end-to-end Machine Learning projects, each built using Python and Jupyter Notebooks.
Every notebook demonstrates full ML workflows — data loading, cleaning, EDA, preprocessing, model training, evaluation, and insights.

This collection is ideal for:

✅ ML/AI portfolio

✅ Academic projects

✅ Industry preparation

✅ Demonstrating real data handling skills
<br><br>

📂 Included ML Code:
<br><br>
A)  Supervised Learning Models:
<br><br>
Supervised learning is when we teach a machine using examples that already have the right answers.
It learns from this labeled data so it can make accurate predictions on new information later — kind of like learning from past examples to guess future outcomes.

Example: Teaching a model to predict house prices by showing it many houses along with their actual prices.

<br><br>
The Folder Includes :
<br><br>

1. FinancialData.ipynb — Loan Default Prediction (XYZCorp Lending Data)


📌 Overview:

Predicts whether a loan defaults based on borrower profile and loan attributes.

⭐ Notebook Highlights:

Loaded dataset (73 rows, 246 raw columns)

Removed 200+ unusable “Unnamed” columns

Cleaned missing values using median/mode/Unknown

Dropped non-predictive columns (IDs, payment history, descriptions)

One-hot encoded:

✅ Grade

✅ Term

✅ Sub-grade

✅ Employment length

✅ Home ownership

✅ State

✅ Purpose

Scaled numerical columns using StandardScaler

Trained Logistic Regression

Achieved 82.35% accuracy

Plotted confusion matrix & correlation heatmap

<br><br>
2. banking.ipynb — Banking ML Model (Churn/Loan/Customer Behavior)


📌 Overview:

Machine learning model built on a banking dataset for customer behavior prediction.

⭐ Notebook Highlights:

Cleaned and formatted customer records

One-hot encoded categorical variables

Balanced dataset using SMOTE

Trained models:

✅ Logistic Regression

✅ RandomForest

✅ Gradient Boosting / XGBoost (if included)

Plotted ROC Curve & confusion matrix

Extracted feature importance
<br><br>

3. cs_intrusion_det.ipynb — Cybersecurity Intrusion Detection

📌 Overview:

Builds an Intrusion Detection System (IDS) to detect malicious network traffic.

⭐ Notebook Highlights:

Cleaned network traffic dataset

Visualized normal vs attack traffic

Encoded protocol & attack-type fields

Trained ML models:

✅ Logistic Regression

✅ SVM

✅ RandomForest

✅ DecisionTree

Evaluated model performance

Displayed confusion matrix & classification report
<br><br>

4. heart.ipynb — Heart Disease Prediction

📌 Overview:

Predicts heart disease using medical measurements.

⭐ Notebook Highlights:

Cleaned and normalized clinical dataset

Explored correlations using heatmaps

Removed outliers

Built models:

✅ SVM

✅ KNN

✅ RandomForest

✅ DecisionTree

Compared precision, recall, F1-score

Displayed confusion matrix
<br><br>

5. itu_gci.ipynb — Global Cybersecurity Index Analysis

📌 Overview:

Analyzes the cybersecurity readiness of countries using ITU’s GCI data.

⭐ Notebook Highlights:

Loaded ITU GCI dataset

Ranked countries by cybersecurity index

Visualized top & bottom performers

Regional comparison

Grouped countries by cyber-maturity

Created bar, line, and scatter plots

Extracted key insights about global cyber readiness
<br><br>

6. titanic.ipynb — Titanic Survival Prediction
 
📌 Overview:

Predicts whether a passenger survived using demographic and ticket attributes.

⭐ Notebook Highlights:

Loaded Titanic dataset

Handled missing values (Age, Cabin, Embarked)

Encoded categorical features (Sex, Embarked, Pclass)

Performed EDA (survival rates by gender/class/age)

Trained multiple models:

✅ Logistic Regression

✅ Decision Tree

✅ RandomForest

Evaluated using accuracy & confusion matrix
<br><br>

🧠 Skills Demonstrated Across All Projects:

✅ Data Cleaning & Wrangling

✅ Handling Missing & Imbalanced Data

✅ One-Hot Encoding & Feature Engineering

✅ Scaling & Normalization

✅ Exploratory Data Analysis (EDA)

✅ Classification Models (LR, SVM, RF, DT, KNN)

✅ Evaluation Metrics & Visualization

✅ Working with Financial, Cybersecurity & Healthcare Data
<br><br>

🛠 Technologies Used:

Python 3.x

Pandas, NumPy

Scikit-learn

Seaborn, Matplotlib

Jupyter Notebook

Imbalanced-learn (SMOTE)
<br><br>
B) Unsupervised Learning Models:
<br><br>
Unsupervised learning is when a machine learns from data without any labels or predefined answers.

It tries to find hidden patterns, groups, or structures within the data on its own — without being told what the correct output is.
<br><br>
Example: Grouping customers with similar buying habits into clusters without knowing anything about them beforehand.

<br><br>
The Folder Includes:
<br><br>
1. 



<br><br>
📬 Contact:

If you'd like to collaborate or improve any notebook, feel free to create an Issue or Pull Request.
