# Unsupervised Learning — Definition

Unsupervised learning trains models on **unlabeled data** to discover hidden structure: clusters, groups, anomalies, or lower-dimensional representations.  
Common tasks:
- **Clustering** — group similar samples (e.g., customer/product segmentation).  
- **Dimensionality reduction** — compress features while preserving structure (e.g., PCA).  

---

#  Online Retail — K-Means Clustering (OnlineRetail.ipynb)

**Purpose:** discover product/purchase patterns by clustering transactions on `Quantity` and `UnitPrice`.  
This notebook provides a practical, business-focused unsupervised workflow for e-commerce analytics.

## Project Summary
- Input: online retail transaction log (InvoiceNo, StockCode, Quantity, UnitPrice, CustomerID, Country).  
- Goal: group transactions/products to surface business-actionable clusters (fast-moving SKUs, bulk purchases, high-value low-volume items).

## What the Notebook Contains (step-by-step)
1. **Data loading & inspection**
   - Read CSV into `df`
   - Quick shape, dtypes, missing count checks

2. **Preprocessing**
   - Remove noisy text field: `df.drop('Description', axis=1)`  
   - Handle missing rows (drop or impute depending on notebook cell)  
   - Label encode categorical identifiers used for analysis:
     ```python
     le = LabelEncoder()
     df['Country'] = le.fit_transform(df['Country'])
     df['StockCode'] = le.fit_transform(df['StockCode'])
     ```

3. **Feature selection**
   - Use `Quantity` and `UnitPrice` as clustering features:
     ```python
     X = df[['Quantity', 'UnitPrice']]
     ```

4. **Feature scaling**
   - Standardize features because K-Means is distance-based:
     ```python
     scaler = StandardScaler()
     X_scaled = scaler.fit_transform(X)
     ```

5. **Determine optimal K (Elbow Method)**
   - Compute inertia over a K range and plot the elbow to select `optimal_k`.

6. **Train K-Means**
   - Fit K-Means with `n_clusters=optimal_k`, predict cluster labels:
     ```python
     kmeans = KMeans(n_clusters=optimal_k, random_state=42)
     clusters = kmeans.fit_predict(X_scaled)
     df['cluster'] = clusters
     ```

7. **Postprocessing & Export**
   - Add cluster labels back to `df`, inspect cluster centroids & sizes  
   - Export final dataset: `df.to_excel("OnlineRetail_cluster.xlsx", index=False)`

## Evaluation / Validation
- Elbow plot (inertia vs k) to choose K  
- Optional: silhouette scores for cluster quality

## Key Business Insights
- Clusters separate **high-value low-volume** vs **low-price high-volume** items.  
- Useful for inventory prioritization, targeted promotions, and SKU lifecycle analysis.

## Skills Demonstrated
- Data cleaning & encoding  
- Feature selection for clustering  
- Proper scaling for distance algorithms  
- Elbow method & silhouette validation  
- Exporting clustered data for BI use

---
---


