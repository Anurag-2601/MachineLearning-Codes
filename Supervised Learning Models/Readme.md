# Supervised Learning — Definition  

Supervised learning is a type of machine learning where a model is trained on **labeled data**.  
Each input example has a corresponding target/output value, and the algorithm learns a mapping from inputs to outputs by minimizing prediction errors.

Typical tasks:
- **Classification** – e.g., spam detection, risk prediction  
- **Regression** – e.g., price or score prediction  

---

# FinancialData — Credit Default Prediction (XYZCorp Lending Data)

A complete machine learning workflow to predict whether a loan will **default** using the `XYZCorp_LendingData` dataset.

---

## 1. Project Overview  

This notebook builds a **credit default prediction model** using real-world lending data.  
It walks through:

- Data inspection and type analysis  
- Feature selection from high-dimensional data (200+ columns)  
- Handling missing values and irrelevant columns  
- Encoding categorical variables  
- Training and evaluating a classification model  

The goal is to predict the `default_ind` flag for each loan.

---

## 2. Dataset & Features  

- **Domain:** Consumer lending / credit risk  
- **Target:** `default_ind` (loan default indicator)  
- **Input features include:**  
  - Loan amount, interest rate, installment  
  - Employment-related fields  
  - Credit history / utilization features  
  - Geographic and application attributes  

A cleaned feature set (`df_f`) is one-hot encoded into `df_encoded` using `pd.get_dummies`.

---

## 3. Modeling Approach  

- Split data into train/test using `train_test_split`  
- Standardize numerical features using `StandardScaler`  
- Train a **Logistic Regression** model as the main classifier  
- Evaluate using:  
  - **Accuracy Score**  
  - **Classification Report** (precision, recall, F1-score)  
  - **Confusion Matrix** (plotted via `ConfusionMatrixDisplay`)  

The current notebook reports an accuracy of **≈ 82.35%** on the test set.

---

## 4. Skills Demonstrated  

- Working with **high-dimensional financial data**  
- Feature selection and one-hot encoding  
- Building a full classification pipeline (scaling → modeling → evaluation)  
- Interpreting confusion matrices and classification reports  
- Applying ML to **credit risk / default prediction**  

---
---

# Online Retail — Customer Country Classification with KNN  

A K-Nearest Neighbors–based model to classify **customer country** from online retail transaction data.

---

## 1. Project Overview  

This notebook uses an **online retail dataset** to predict the **country** associated with each transaction.  
It demonstrates:

- Cleaning transactional data  
- Encoding categorical attributes  
- Feature scaling  
- Training a distance-based classifier  

---

## 2. Dataset & Target  

- **Domain:** E-commerce / Online Retail  
- **Source file:** `Online_retail.csv`  
- **Preprocessing steps:**  
  - Drop `Description` column (unnecessary text)  
  - Drop rows with missing values  
  - Apply `LabelEncoder` to all object (categorical) columns  

- **Features (X):** All columns except `Country`  
- **Target (y):** `Country`  

---

## 3. Modeling Approach  

- Split data into train/test using `train_test_split` (80/20)  
- Scale features using `StandardScaler`  
- Train a **KNeighborsClassifier** with `n_neighbors=5`  
- Evaluate using:  
  - **Accuracy Score**  
  - **Classification Report** (`classification_report`)  
  - **Confusion Matrix** (printed and visualized with `matplotlib`)  

---

## 4. Skills Demonstrated  

- End-to-end workflow on **transactional retail data**  
- Proper handling of categorical features via label encoding  
- Sensible use of **feature scaling** for KNN  
- Model evaluation using accuracy, precision, recall, F1-score  
- Visual interpretation of confusion matrices  

---
---

# banking — Customer Response Prediction (Bank Marketing Data)

A classification pipeline for predicting customer response to a **bank marketing campaign**.

---

## 1. Project Overview  

This notebook uses the `bmarketing.csv` dataset to predict whether a customer will respond positively (`y = yes/no`) to a marketing offer.  
It covers:

- Cleaning and deduplication  
- One-hot encoding of categorical variables  
- Scaling numeric features  
- Training and comparing multiple classifiers  

---

## 2. Dataset & Target  

- **Domain:** Banking / Direct marketing  
- **Target:** `y` (mapped from `"yes"/"no"` to `1/0`)  
- **Feature Engineering:**  
  - `pd.get_dummies` applied to columns like `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day_of_week`, `poutcome`  

- **Features (X):** All columns except `y`  
- **Target (y):** Encoded binary outcome  

---

## 3. Modeling Approach  

- Train/test split using `train_test_split`  
- Scale numeric columns with `StandardScaler`  
- Models trained:  
  - **Logistic Regression** (baseline classifier)  
  - **XGBoost (XGBClassifier)** for stronger performance  
  - **SVM (SVC)** for non-linear decision boundaries  
- Evaluation:  
  - **Accuracy Score**  
  - **Classification Report** for each model  

(Confusion matrix and visualization utilities are also imported and partially used.)

---

## 4. Skills Demonstrated  

- Full preprocessing pipeline on **bank marketing data**  
- Converting categorical “yes/no” outcomes into a proper ML target  
- Using both **linear (Logistic Regression)** and **non-linear (XGBoost, SVM)** models  
- Interpreting classification reports to compare approaches  
- Applying ML to **customer response / campaign analytics**  

---
---

# Cybersecurity Intrusion Detection using Machine Learning  

A classical ML–based **Intrusion Detection System (IDS)** using structured network traffic data.

---

## 1. Project Overview  

This notebook builds a classification model to detect whether a network connection is an **attack or normal** using `intrusion1.csv`.  
It demonstrates the full IDS pipeline from preprocessing through model comparison.

---

## 2. Dataset & Features  

- **Domain:** Cybersecurity / Network Intrusion Detection  
- **Source:** `intrusion1.csv`  
- **Preprocessing:**  
  - Convert `protocol_type` to string  
  - Fill missing values in `encryption_used` with mode  
  - Drop `session_id` (non-predictive identifier)  
  - One-hot encode `protocol_type`, `encryption_used`, `browser_type` using `pd.get_dummies`  
  - Explore correlations using `.corr()`  

- **Features (X):** All columns except `attack_detected`  
- **Target (y):** `attack_detected` (attack label)  

---

## 3. Modeling Approach  

- Train/test split using `train_test_split`  
- Scale numeric columns with `StandardScaler`  
- Models trained and evaluated:  
  - **Logistic Regression (lr_model)**  
  - **DecisionTreeClassifier (dt_model)**  
  - **RandomForestClassifier (rf)**  
- Evaluation:  
  - **Classification Report** for Logistic Regression  
  - **Accuracy Score** for Logistic Regression and Decision Tree  
  - **Feature Importance** from the Decision Tree  
  - Confusion matrix visualization for Random Forest via `ConfusionMatrixDisplay`  

---

## 4. Skills Demonstrated  

- Applying ML in a **security-critical context**  
- Handling categorical protocol and encryption fields  
- Building multiple IDS classifiers and comparing them  
- Using feature importance to understand what drives detections  
- Evaluating models with security-relevant metrics  

---
---

# Stroke Risk Prediction (heart.ipynb)

A supervised learning project to predict **stroke occurrence** from patient-level health data.

---

## 1. Project Overview  

Despite the filename (`heart.ipynb`), this notebook uses a **stroke prediction** dataset (`heart.csv`) where the target is `stroke`.  
It builds a model to estimate stroke risk based on demographic and lifestyle features.

---

## 2. Dataset & Preprocessing  

- **Domain:** Healthcare / Stroke risk prediction  
- **Target:** `stroke` (binary indicator)  
- **Preprocessing steps:**  
  - Handle missing values in `bmi` using median imputation  
  - One-hot encode categorical columns:  
    - `gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`  
  - Compute and visualize correlations via `.corr()`  

- **Features (X):** `df_encoded.drop('stroke', axis=1)`  
- **Target (y):** `df_encoded['stroke']`  

---

## 3. Modeling Approach  

- Train/test split with `train_test_split`  
- Standardize numeric features with `StandardScaler`  
- Address class imbalance using **SMOTE** (`SMOTE().fit_resample(X_train, y_train)`)  
- Models trained:  
  - **Logistic Regression**  
  - **DecisionTreeClassifier (dt_model)**  
  - **RandomForestClassifier (rf, with class_weight='balanced', max_depth=8)**  

- Evaluation:  
  - **Accuracy Score** (explicitly calculated for Random Forest)  
  - Feature importance from the Decision Tree (`feature_importances_`)  

---

## 4. Skills Demonstrated  

- Healthcare-focused ML on **stroke risk**  
- Handling missing clinical attributes (BMI)  
- Managing **class imbalance** using SMOTE  
- Comparing tree-based and linear models for medical prediction  
- Interpreting feature importance for clinical decision support  

---
---

# ITU GCI — Global Cybersecurity Index Regression  

A regression-focused analysis of the **International Telecommunication Union Global Cybersecurity Index (ITU GCI)**.

---

## 1. Project Overview  

This notebook analyzes the `ITU_GCI.csv` dataset and predicts the **GCI score (`OBS_VALUE`)** based on country and indicator metadata.  
It combines:

- Exploratory data analysis  
- Categorical encoding  
- Regression modeling  
- Error and goodness-of-fit evaluation  

---

## 2. Dataset & Features  

- **Domain:** Global cybersecurity readiness / policy metrics  
- **Target:** `OBS_VALUE` (numerical GCI score)  
- **Preprocessing:**  
  - Inspect dtypes, missing values, summary statistics  
  - Compute correlations using only numeric columns (`numeric_df`)  
  - One-hot encode categorical metadata columns such as:  
    - `REF_AREA`, `REF_AREA_LABEL`  
    - `INDICATOR`, `INDICATOR_LABEL`  
    - `UNIT_MEASURE_LABEL`, `TIME_PERIOD`  

- **Features (X):** One-hot–encoded categorical metadata  
- **Target (y):** `OBS_VALUE`  

---

## 3. Modeling Approach  

- Train/test split using `train_test_split`  
- Scale numeric columns with `StandardScaler`  
- Models used (as per code imports and usage):  
  - **LinearRegression** (baseline regressor)  
  - **DecisionTreeRegressor**  
  - **RandomForestRegressor** (referred to as `rf_model` in the code)  

- Evaluation (on Random Forest predictions):  
  - **Mean Squared Error (MSE)**  
  - **R² Score** (coefficient of determination)  
- Feature importance is computed for the Random Forest model to rank the most influential features.

---

## 4. Skills Demonstrated  

- Working with **country-level cybersecurity metrics**  
- Transforming rich categorical metadata into usable features  
- Framing cybersecurity readiness as a **regression problem**  
- Evaluating models with MSE and R²  
- Using feature importance to understand global cybersecurity drivers  

---
---

# Titanic — Survival Prediction (Classification)

A classic ML classification project predicting Titanic passenger survival.

---

## 1. Project Overview  

This notebook uses the **Kaggle Titanic dataset** (`titanic.csv`) to predict whether a passenger survived the disaster.  
It demonstrates a full ML workflow on a well-known benchmark dataset.

---

## 2. Dataset & Preprocessing  

- **Domain:** Transportation / Survival analysis  
- **Target:** `Survived` (0 = no, 1 = yes)  
- **Preprocessing steps:**  
  - Impute missing `Age` with mean  
  - Fill missing `Cabin` with `'U'` and `Embarked` with mode  
  - Drop non-useful identifiers: `PassengerId`, `Ticket`, `Name`  
  - One-hot encode `Sex`, `Embarked`, `Cabin` using `pd.get_dummies(drop_first=True)`  
  - Explore correlations via `.corr()`  

- **Features (X):** `df.drop('Survived', axis=1)`  
- **Target (y):** `df['Survived']`  

---

## 3. Modeling Approach  

- Train/test split using `train_test_split`  
- Models trained:  
  - **Logistic Regression** (baseline)  
  - **DecisionTreeClassifier (dt_model)**  
  - **RandomForestClassifier (rf)**  

- Evaluation:  
  - **Accuracy Score** (for Random Forest and Decision Tree)  
  - **Classification Report**  
  - **Confusion Matrix** plotted via `ConfusionMatrixDisplay`  
  - **Feature Importance** extracted from the Decision Tree model  

---

## 4. Skills Demonstrated  

- Handling missing and categorical data in a well-known public dataset  
- Building multiple baseline and tree-based classifiers  
- Comparing performance and interpretability  
- Using feature importance and confusion matrices to interpret model behavior  

---
---
