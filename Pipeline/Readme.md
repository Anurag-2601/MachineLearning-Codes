# Pipeline

A Machine Learning Pipeline is an automated sequence of steps—such as preprocessing, feature engineering, scaling, and modeling—that ensures data flows consistently from raw input to final prediction. Pipelines improve code organization, prevent data leakage, and make ML workflows repeatable, efficient, and production-ready.

---
---

# Bank Marketing Prediction using Machine Learning Pipelines

A production-style ML pipeline for predicting customer subscription behavior in bank marketing campaigns.

## 1. Project Overview

This notebook builds a multi-model machine learning workflow to predict whether a customer will subscribe to a banking term deposit based on demographic & marketing interaction features.

The project uses a robust pipeline architecture that automates:

- Missing value handling

- Numerical scaling

- Categorical encoding

- Model training

- Model comparison

Five ML models are trained using identical preprocessing via scikit-learn Pipelines + ColumnTransformer.

---

## 2. Dataset Description

The dataset used:

     bmarketing.csv

Contains features such as:

- Age

- Job

- Marital status

- Education

- Loan/Housing details

- Previous campaign contact info

- Communication type & duration

- Many categorical fields

**Target variable:**

     df['y'] = df['y'].map({'yes':1, 'no':0})


- 1 → Customer subscribed

- 0 → Customer did not subscribe

---

## 3. Data Preprocessing

**Feature–Target Split :**

          X = df.drop('y', axis=1)
          y = df['y']

**Train–Test Split:**

     x_train, x_test, y_train, y_test = train_test_split(
         X, y, test_size=0.33, random_state=42)

**Identify Numerical & Categorical Features :**

     numerical_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
     categorical_features = X.select_dtypes(include=['object']).columns.tolist()

---

## 4. Preprocessing Pipelines

 **Numerical Pipeline:**

- Median imputation

- Standard scaling

          numerical_cols = Pipeline([
              ("Simple Imputer", SimpleImputer(strategy='mean')),
              ("Scaling", StandardScaler())
          ])

**Categorical Pipeline:**

- Most frequent imputation

- One-Hot Encoding

          categorical_cols = Pipeline([
              ("Simple Imputer", SimpleImputer(strategy='most_frequent')),
              ("ohe", OneHotEncoder(handle_unknown='ignore'))
          ])

**Combined ColumnTransformer :**

     preprocessing = ColumnTransformer(
         transformers=[
             ("numerical", numerical_cols, numerical_features),
             ("categorical", categorical_cols, categorical_features)
         ]
     )

---

4. Preprocessing Pipelines

**Numerical Pipeline:**

- Median imputation

- Standard scaling

          numerical_cols = Pipeline([
              ("Simple Imputer", SimpleImputer(strategy='mean')),
              ("Scaling", StandardScaler())
          ])

**Categorical Pipeline:**

- Most frequent imputation

- One-Hot Encoding

          categorical_cols = Pipeline([
              ("Simple Imputer", SimpleImputer(strategy='most_frequent')),
              ("ohe", OneHotEncoder(handle_unknown='ignore'))
          ])

**Combined ColumnTransformer:**

     preprocessing = ColumnTransformer(
         transformers=[
             ("numerical", numerical_cols, numerical_features),
             ("categorical", categorical_cols, categorical_features)
         ]
     )

---

## 6. Model Comparison

All models are stored in a dictionary and trained/compared:

     models = {
         "Logistic Regression": lor_pipeline,
         "Support Vector Machine": svr_pipeline,
         "Neural Network (MLP)": mlp_pipeline,
         "k-Nearest Neighbors": knn_pipeline,
         "XGBoost": xgb_pipeline
     }
     
     results = {}
     
     for name, model in models.items():
         model.fit(x_train, y_train)
         y_pred = model.predict(x_test)
         acc = accuracy_score(y_test, y_pred)
         print(f"{name}: {acc*100:.2f}%")

**Outputs include:**

- Accuracy of each model

- Side-by-side performance comparison

This gives a clear view of which pipeline performs best for the banking dataset.

---

## 7. Skills Demonstrated

This notebook proves strong ML engineering skills:

✔ Proper Feature Engineering 

✔ Clean preprocessing using Pipelines

✔ Correct handling of mixed datatypes

✔ Multiple ML models inside pipelines

✔ Model comparison automation

✔ XGBoost integration (industry-favorite)

✔ High-quality, modular code organization

---
---

# Intrusion Detection using Machine Learning Pipelines

A complete ML pipeline for detecting cyber intrusions using Random Forest + automated preprocessing.

## 1. Project Overview

This notebook builds a supervised machine learning pipeline to detect cyber intrusions in network traffic data. Using a combination of:

- Automatic preprocessing (imputation, scaling, encoding)

- Scikit-learn Pipelines

- ColumnTransformer for mixed data types

- RandomForestClassifier

- Hyperparameter tuning using GridSearchCV

…it demonstrates a production-style implementation of intrusion detection suitable for cybersecurity analytics, SOC automation, and ML-security workflows.

---

## 2. Dataset Description

The dataset used is:

          intrusion1.csv


It contains network traffic features such as:

- Encryption usage

- Communication characteristics

- Numerical & categorical protocol attributes

- **Target column:** attack_detected

   - 0 = normal traffic

   - 1 = intrusion attempt

 Missing Value Handling 

The column encryption_used has missing values, filled using its mode:

     df['encryption_used'] = df['encryption_used'].fillna(df['encryption_used'].mode()[0])

---

## 3. Data Preprocessing

Feature/Target Split:

     x = df.drop('attack_detected', axis=1)
     y = df['attack_detected']

 Train–Test Split:
 
     x_train, x_test, y_train, y_test = train_test_split(
         x, y, test_size=0.33, random_state=42)

Identify Numerical & Categorical Columns:

     num_features = x.select_dtypes(include=['int64','float64']).columns.tolist()
     cat_features = x.select_dtypes(include=['object']).columns.tolist()

Numerical Pipeline:

- Mean imputation

- Standard Scaling

          numerical_cols = Pipeline([
              ("simple_imputer", SimpleImputer(strategy='mean')),
              ("scaling", StandardScaler())
          ])

Categorical Pipeline::

- Most frequent imputation

- One-Hot Encoding

          categorical_cols = Pipeline([
              ("simple_imputer", SimpleImputer(strategy='most_frequent')),
              ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
          ])

Combined Preprocessor (ColumnTransformer):

     preprocessing = ColumnTransformer([
         ("categorical", categorical_cols, cat_features),
         ("numerical", numerical_cols, num_features)
     ])


This ensures both numeric and categorical preprocessing happen automatically.

---

## 4. Model Construction (Random Forest Pipeline)

A full end-to-end pipeline is built:

          pipe = Pipeline([
              ("preprocessing", preprocessing),
              ("estimator", RandomForestClassifier())
          ])

Model Training: 

     pipe.fit(x_train, y_train)

Prediction:

     y_pred = pipe.predict(x_test)

---

## 5. Model Evaluation 

          acc = accuracy_score(y_test, y_pred)
          print("Accuracy Score:", round(acc * 100, 2), "%")


This gives the base RandomForest accuracy before tuning.

---

## 6. Hyperparameter Tuning (GridSearchCV)

The notebook runs GridSearchCV to improve the model:

     param_grid = {
         "estimator__n_estimators": [100, 200],
         "estimator__max_depth": [None, 10, 20],
         "estimator__min_samples_split": [2, 5],
         "estimator__min_samples_leaf": [1, 2]
     }


Grid search is applied on the entire pipeline:

     grid = GridSearchCV(
         estimator=pipe,
         param_grid=param_grid,
         cv=3,
         scoring="accuracy",
         n_jobs=-1,
         verbose=2
     )


Final evaluation:

     y_pred = grid.best_estimator_.predict(x_test)
     print("Test accuracy:", accuracy_score(y_test, y_pred)*100, "%")

--- 

6. Hyperparameter Tuning (GridSearchCV)

The notebook runs GridSearchCV to improve the model:

          param_grid = {
              "estimator__n_estimators": [100, 200],
              "estimator__max_depth": [None, 10, 20],
              "estimator__min_samples_split": [2, 5],
              "estimator__min_samples_leaf": [1, 2]
          }


Grid search is applied on the entire pipeline:

          grid = GridSearchCV(
              estimator=pipe,
              param_grid=param_grid,
              cv=3,
              scoring="accuracy",
              n_jobs=-1,
              verbose=2
          )


Final evaluation:

          y_pred = grid.best_estimator_.predict(x_test)
          print("Test accuracy:", accuracy_score(y_test, y_pred)*100, "%")

---

## 7. Skills Demonstrated

This notebook demonstrates strong ML engineering skills:

✔ Handling real intrusion datasets 

✔ Pipeline-based preprocessing

✔ Encoding + scaling via ColumnTransformer

✔ RandomForest classification

✔ Hyperparameter tuning (GridSearchCV)

✔ Clean, modular ML code suitable for production

✔ Understanding of cybersecurity ML workflow

These are highly valuable for internships in:

- Cybersecurity analytics

- Threat detection

- Machine learning engineering

- SOC automation with AI

---
---

# Heart Stroke Prediction using ML Pipelines

A complete end-to-end machine learning pipeline for predicting stroke risk from clinical data.

## 1. Project Overview

This notebook builds a supervised machine learning model to predict whether a patient is likely to suffer a stroke based on clinical and lifestyle features.

The focus of this project is not just on model training, but on building a robust, production-style ML pipeline using:

- Automatic preprocessing for numerical & categorical data

- Scikit-learn Pipeline and ColumnTransformer

- Random Forest–based classification

- Hyperparameter tuning with GridSearchCV

- Model evaluation using accuracy

This project is especially relevant for healthcare analytics, ML in medicine, and risk prediction systems.

---

## 2. Dataset Description

The dataset is loaded from:

     df = pd.read_csv("heart.csv")

It contains patient-level records with:

**Input features:** clinical and lifestyle attributes (e.g., age, BMI, glucose level, categorical risk factors, etc.)

**Target column:**

- stroke → binary label (0 = no stroke, 1 = stroke)

**Handling Missing Values:**

The bmi column contains missing values and is imputed using the median:

    df['bmi'] = df['bmi'].fillna(df['bmi'].median())

---

## 3. Data Preprocessing

**Feature/Target Split:**
 
     X = df.drop('stroke', axis=1)
     y = df['stroke']

**Train–test split:**

     x_train, x_test, y_train, y_test = train_test_split(
                      X, y, test_size=0.33, random_state=42)

**Identifying Numerical & Categorical Features**

      numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
      
      categorical_features = X.select_dtypes(include=['object']).columns.tolist()

-Pipelines for Numerical & Categorical Data

**Numerical pipeline:**

- Imputation using mean

- Standardization using StandardScaler

       numerical_cols = Pipeline(
            steps=[
                ("simple_imputer", SimpleImputer(strategy='mean')),
                ("scaling", StandardScaler())
                ]
            )


**Categorical pipeline:**

- Imputation using most frequent value

- One-hot encoding

       categorical_cols = Pipeline(
                steps=[
                    ("simple_imputer", SimpleImputer(strategy='most_frequent')),
                    ("ohe", OneHotEncoder(handle_unknown='ignore'))
                ]
            )

**Combined Preprocessor (ColumnTransformer):**

       preprocessing = ColumnTransformer(
              transformers=[
                  ("Categorical", categorical_cols, categorical_features),
                  ("Numerical", numerical_cols, numerical_features)
              ]
         )


This ensures all preprocessing is bundled and consistently applied.

---

## 4. Model Training with Pipeline

**A full pipeline is built with preprocessing + model:**

      pipe = Pipeline(
          steps=[
              ("preprocessor", preprocessing),
              ("regressor", RandomForestClassifier())
          ]
      )


Note: Although the step is named "regressor", it is a RandomForestClassifier used for classification.

**Fit the model:**

    pipe.fit(x_train, y_train)


**Predict on test set:**
    
    y_pred = pipe.predict(x_test)

---

## 5. Model Evaluation

The notebook evaluates performance using accuracy score:

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy Score", round(acc * 100, 2), "%")


You can extend this section with:

- Precision, recall, F1-score

- Confusion matrix

- ROC-AUC

for more detailed medical evaluation.

---

## 6. Hyperparameter Tuning (GridSearchCV)

The notebook uses GridSearchCV to search for better hyperparameters for the model within the same pipeline structure.

Example structure (conceptually aligned with your code):

    from sklearn.model_selection import GridSearchCV
    
    param_grid = {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [None, 10, 20],
        "regressor__min_samples_split": [2, 5],
        "regressor__min_samples_leaf": [1, 2]
    }
    
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2
    )
    
    grid.fit(x_train, y_train)
    
    print("Best Params:", grid.best_params_)
    print("Best CV Score:", grid.best_score_)
    y_pred = grid.best_estimator_.predict(x_test)
    print("Test accuracy:", accuracy_score(y_test, y_pred) * 100, "%")


This step shows that you understand how to tune models properly while respecting the pipeline structure.

---

## 7. Skills Demonstrated

This notebook clearly shows your ability to:

- Work with healthcare/clinical tabular data

- Handle missing values appropriately (bmi imputation)

- Separate and process numerical vs categorical features

- Build scikit-learn Pipelines and ColumnTransformers

- Train and evaluate a RandomForestClassifier

- Use GridSearchCV for model optimization

- Maintain clean, modular ML code that is close to production style

These are highly valued skills for ML Engineer, Data Scientist, and Healthcare AI internship roles.

---
---


