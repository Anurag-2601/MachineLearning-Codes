# Pipeline — Definition

A **Machine Learning Pipeline** is an automated, repeatable sequence of steps that moves data from raw input to final prediction.  
Typical pipeline stages include: data ingestion, cleaning/imputation, feature engineering, scaling, modeling, and evaluation.  
Pipelines prevent data leakage, enforce consistent preprocessing, improve reproducibility, and make models production-ready.

---

# Bank Marketing Prediction — ML Pipelines (banking_pipeline.ipynb)

A production-style machine learning pipeline to predict whether a customer will subscribe to a bank term deposit using bank marketing data.

---

## 1. Project Overview  
This notebook implements a robust, end-to-end ML pipeline that standardizes preprocessing and evaluation across multiple models. It demonstrates best-practice engineering using `ColumnTransformer` + `Pipeline` in scikit-learn to ensure identical preprocessing for every estimator and to simplify model comparison and deployment.

**Business goal:** Predict `y` (customer subscribed: yes→1 / no→0) for targeted marketing campaigns.

---

## 2. Dataset Summary  
- **File:** `bmarketing.csv`  
- **Domain:** Banking / Direct marketing  
- **Common features:** age, job, marital, education, contact type, campaign duration, previous outcomes, loan/housing flags, balance, etc.  
- **Target:** `y` mapped as `{'yes':1, 'no':0}`

**Challenges:** many categorical features, mixed datatypes, potential class imbalance, missing or noisy values.

---

## 3. Preprocessing & Pipeline Design  
- **Train/test split:** `train_test_split(X, y, test_size=0.33, random_state=42)`  
- **Numerical pipeline:** `SimpleImputer(strategy='mean')` → `StandardScaler()`  
- **Categorical pipeline:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`  
- **Combined:** `ColumnTransformer` that applies numeric/categorical pipelines to appropriate columns.  
- **Full estimator pipeline:** `Pipeline([('preprocessing', preprocessing), ('estimator', <model>)])`

All models use the same `preprocessing` to ensure consistent evaluation and avoid leakage.

---

## 4. Models Trained & Compared  
A dictionary of pipelines is created and trained, producing side-by-side results:
- Logistic Regression (baseline)  
- Support Vector Machine (SVC)  
- Multi-layer Perceptron (MLP)  
- k-Nearest Neighbors (KNN)  
- XGBoost (XGBClassifier)

Each pipeline is fit identically and evaluated on the held-out test set for fair comparison.

---

## 5. Evaluation Metrics & Outputs  
- **Primary metric:** Accuracy (printed for each model)  
- **Additional outputs:** Classification reports (precision/recall/F1), confusion matrices, and model-specific diagnostics.  
- Model ranking based on test accuracy and business-relevant metrics.

---

## 6. Key Insights  
- Pipelines greatly reduce boilerplate and prevent preprocessing mismatches between training and inference.  
- XGBoost / tree-based models typically perform best on mixed categorical/numerical campaign data.  
- Proper imputation and encoding materially affect downstream accuracy.  
- A single pipeline structure makes it trivial to add more models, cross-validation, or a GridSearch step.

---

## 7. What I Learned / Skills Demonstrated  
- Building production-ready ML pipelines with `ColumnTransformer` + `Pipeline`.  
- Managing mixed datatypes and robust imputation strategies.  
- Automating model comparison and evaluation.  
- Integrating XGBoost with scikit-learn pipelines.  
- Translating a data science notebook into a deployable pipeline.

---
---

# Intrusion Detection — ML Pipelines (cs_intusion_det_pipeline.ipynb)

A production-style ML pipeline for detecting network intrusions; demonstrates end-to-end preprocessing, modelling, and hyperparameter tuning in a security context.

---

## 1. Project Overview  
This notebook constructs an automated pipeline to detect intrusions in network traffic using `intrusion1.csv`. It emphasizes production-ready patterns: consistent preprocessing for numeric & categorical fields, pipeline-wrapped estimators, and GridSearchCV for hyperparameter optimization.

**Business goal:** Predict `attack_detected` (0 = normal, 1 = intrusion).

---

## 2. Dataset Summary  
- **File:** `intrusion1.csv`  
- **Domain:** Cybersecurity / Network monitoring  
- **Common features:** protocol type, encryption_used, browser_type, byte counts, duration, connection flags, etc.  
- **Target:** `attack_detected` (binary)

**Challenges:** mixed datatypes, missing values (e.g., `encryption_used`), class imbalance, high-dimensional categorical features.

---

## 3. Preprocessing & Pipeline Design  
- **Missing value handling:** fill `encryption_used` with mode.  
- **Train/test split:** `train_test_split(x, y, test_size=0.33, random_state=42)`  
- **Numerical pipeline:** `SimpleImputer(strategy='mean')` → `StandardScaler()`  
- **Categorical pipeline:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`  
- **Combined:** `ColumnTransformer` for automated mixed-type preprocessing.  
- **Full pipeline:** `Pipeline([('preprocessing', preprocessing), ('estimator', RandomForestClassifier())])`

---

## 4. Modeling & Tuning  
- Base model: **RandomForestClassifier** wrapped in a pipeline and evaluated on test data.  
- **GridSearchCV** over pipeline estimator hyperparameters (e.g., `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) performed with cross-validation to find best parameters without breaking the preprocessing flow.

---

## 5. Evaluation Metrics & Outputs  
- **Primary metrics:** Accuracy, Precision, Recall, F1-score (recall is especially important in IDS).  
- Grid search returns the best pipeline; final test accuracy and confusion matrix are reported.  
- Feature importances (from the final Random Forest) are extracted post-transform (with care to map back to preprocessed feature names when needed).

---

## 6. Key Insights  
- Pipeline-wrapped GridSearch prevents leakage and ensures consistent CV across preprocessing steps.  
- Tree ensembles (Random Forest) handle mixed feature types and noisy intrusion signals well.  
- Proper categorical encoding and handling of missing protocol/encryption values materially affect detection performance.  
- Tuning improves detection rates while balancing false positives — crucial in SOC workflows.

---

## 7. What I Learned / Skills Demonstrated  
- Building secure, repeatable ML pipelines for cybersecurity use-cases.  
- Advanced pipeline usage with `GridSearchCV` for hyperparameter tuning.  
- Handling categorical protocol fields and mapping feature importances.  
- Interpreting security-relevant metrics (recall, precision) for operational deployment.

---
---

# Heart Stroke Prediction — ML Pipelines (heart_pipeline.ipynb)

A healthcare-focused ML pipeline to predict stroke risk using clinical/lifestyle features with production-style preprocessing and hyperparameter tuning.

---

## 1. Project Overview  
This notebook constructs a pipeline to predict `stroke` (0/1) from patient-level clinical data (`heart.csv`). It demonstrates healthcare-appropriate preprocessing (imputation, encoding), class-imbalance handling, pipeline training, and hyperparameter search.

**Business goal:** Identify patients at elevated risk of stroke for targeted interventions.

---

## 2. Dataset Summary  
- **File:** `heart.csv`  
- **Domain:** Healthcare / Clinical risk prediction  
- **Typical features:** age, bmi, glucose, hypertension indicator, heart disease, work type, smoking status, residence type, etc.  
- **Target:** `stroke` (binary)

**Challenges:** missing `bmi` values, mixed categorical and numeric features, class imbalance (strokes are rarer).

---

## 3. Preprocessing & Pipeline Design  
- **Imputation:** `bmi` median imputation: `df['bmi'] = df['bmi'].fillna(df['bmi'].median())`  
- **Train/test split:** `train_test_split(X, y, test_size=0.33, random_state=42)`  
- **Numerical pipeline:** `SimpleImputer(strategy='mean')` → `StandardScaler()`  
- **Categorical pipeline:** `SimpleImputer(strategy='most_frequent')` → `OneHotEncoder(handle_unknown='ignore')`  
- **Combined:** `ColumnTransformer` to bundle preprocessing.  
- **Full pipeline:** `Pipeline([('preprocessor', preprocessing), ('classifier', RandomForestClassifier())])`

**Class imbalance handling:** SMOTE or class-weighted RandomForest is applied in training (SMOTE used in notebook to resample training set before fit).

---

## 4. Modeling & Tuning  
- Base estimators: Logistic Regression, DecisionTreeClassifier, RandomForestClassifier (within pipeline).  
- **SMOTE** applied to training split to improve minority class recall.  
- **GridSearchCV** applied to pipeline for hyperparameter tuning (n_estimators, max_depth, etc.).

---

## 5. Evaluation Metrics & Outputs  
- **Primary metrics:** Accuracy, Precision, Recall, F1-score — recall is emphasized due to clinical importance.  
- Final test accuracy reported; feature importances examined to highlight key clinical predictors.  
- Recommended additional evaluation: ROC-AUC and confusion matrix for medical thresholds.

---

## 6. Key Insights  
- Targeted imputation (median for `bmi`) and SMOTE improve model sensitivity for rare stroke events.  
- Tree-based ensembles often outperform linear models on mixed clinical features.  
- Feature importance helps clinicians understand which variables drive risk and supports explainability.

---

## 7. What I Learned / Skills Demonstrated  
- Building compliant, pipeline-driven ML workflows for healthcare.  
- Addressing missing clinical data and class imbalance safely.  
- Using scikit-learn Pipelines + ColumnTransformer with SMOTE and GridSearchCV.  
- Balancing model performance with explainability for clinical decision support.

---
---
