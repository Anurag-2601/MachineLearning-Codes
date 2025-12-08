## Unsupervised Learning — Definition

Unsupervised learning is a machine learning approach where the model is trained on unlabeled data.
The algorithm tries to find hidden patterns, structures, or relationships within the dataset without predefined output labels.

Examples: Clustering (customer segmentation), Dimensionality Reduction (PCA).

---
---

## 1. Project Overview

This notebook performs K-Means clustering on an online retail dataset to uncover meaningful patterns in product quantities and pricing.
The project demonstrates the full workflow of data preprocessing, feature engineering, scaling, clustering, and exporting results, making it valuable for e-commerce analytics and machine learning portfolios.

---

## 2. Dataset Description

The dataset contains online retail transaction records with columns such as:

- InvoiceNo

- StockCode

- Description (removed during preprocessing)

- Quantity

- UnitPrice

- CustomerID

- Country

This dataset is commonly used for analyzing product performance and customer behavior patterns.

---

## 3. Data Preprocessing

The notebook carries out several cleaning and formatting steps:

 Dropped unnecessary columns

--> df.drop('Description', axis=1)

 Encoded categorical features

Label Encoding was used for:

- Country

- StockCode

--> le = LabelEncoder()

--> df['Country'] = le.fit_transform(df['Country'])

--> df['StockCode'] = le.fit_transform(df['StockCode'])


These steps make the data ML-ready.

---

## 4. Feature Selection

The clustering model uses:

- Quantity

- UnitPrice

--> X = df[['Quantity', 'UnitPrice']]


These two features help identify purchasing and price-related behavior patterns.

---

##  5. Feature Scaling

Since K-Means is distance-based, data was standardized:

--> scaler = StandardScaler()

--> X_scaled = scaler.fit_transform(X)


Scaling ensures all features contribute equally to the distance calculations.

---

## 6. Finding Optimal K (Elbow Method)

The notebook computes inertia values for multiple K values and plots the Elbow Curve:

-->  for k in K:
        model = KMeans(n_clusters=k, random_state=42)


This helps determine the most suitable number of clusters.

---

## 7. Training the K-Means Model

After identifying the optimal cluster count, the model is trained:

--> kmeans = KMeans(n_clusters=optimal_k, random_state=42)

--> clusters = kmeans.fit_predict(X_scaled)


Cluster labels are added back to the dataset for further analysis.

---

## 8. Exporting Results

The final clustered dataset is exported as an Excel file:

--> df.to_excel("OnlineRetail_cluster.xlsx", index=False)


This allows for easy business interpretation and external visualization.

---

## 9. Key Insights

- Products can be grouped by purchase volume and price.

- Helps identify fast-moving, high-value, and low-value items.

- Useful for inventory planning, customer segmentation, and promotional targeting.

- K-Means clustering provides a quick, effective way to analyze retail behavioral patterns.

---

## 10. Skills Demonstrated

This notebook highlights your competency in:

✔ Data Cleaning & Encoding

✔ Feature Scaling

✔ Unsupervised Learning (K-Means)

✔ Elbow Method Analysis

✔ Data Export & Reporting

✔ Practical retail analytics

---
---
