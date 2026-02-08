# AI-Based Cybersecurity: Banking Fraud Detection Pipeline

## 📌 Project Overview
This project implements a multi-layered Machine Learning solution to detect fraudulent banking transactions. To meet High Distinction (HD) academic criteria, we utilize a **Hybrid ML Pipeline** that integrates unsupervised behavioral profiling with supervised classification to mitigate the inherent 1.21% class imbalance in the BankSim dataset.

---

## 🤝 Why This Duo? (Robust Hybrid Strategy)

The integration of **K-Medoids Clustering** and **Random Forest Classification** creates a high-resilience system specifically engineered for imbalanced financial datasets.

*1. **Outlier-Resistant Profiling (The 'Who'):** Financial data is frequently skewed by extreme outliers, such as the €530.92 average fraud amount compared to the €31.84 normal average. By using **K-Medoids**, we establish behavioral "Peer Groups" based on actual data members (medoids) rather than averages. This ensures the model's definition of "normal" is grounded in real customer behavior and isn't distorted by massive, one-off fraud spikes.

*2. **Granular Context (The 'What'):** Standard models often apply rigid thresholds to transaction amounts, leading to high false positives. By feeding the `Medoid_Cluster_ID` into the **Random Forest**, the classifier gains **Contextual Intelligence**. Instead of asking if an amount is high for the *entire* population, it asks if the amount is high for that specific behavioral cluster (e.g., a "High-Value Traveler" vs. a "Local Commuter").

*3. **Cybersecurity Resilience:** This hybrid duo addresses **Zero-Day anomalies**. While the Random Forest identifies known "Fraud Signatures," the K-Medoids model acts as a secondary shield by flagging transactions that deviate significantly from a user's assigned Medoid cluster, even if the specific attack pattern is novel.

---

## 🛠️ Technical Implementation

### Method 1: Unsupervised Behavioral Profiling (Type: Clustering)
* **Algorithm:** `K-Medoids` (PAM).
* **Feature Set:** `Amount`, `Age_Label`, `Gender_Label`, `Category`.
* **Outlier Resilience:** Unlike K-Means, K-Medoids minimizes the sum of dissimilarities between points and an actual data member. This protects the cluster centers from being "pulled" toward high-value fraudulent outliers (which reached up to €8,000 in this dataset).
* **Justification:** This ensures stable behavioral centers that accurately represent the 98.78% benign transactions.

### Method 2: Supervised Classification (Type: Classification)
* **Algorithm:** `RandomForestClassifier`.
* **Hybrid Integration:** The `Medoid_Cluster_ID` is included as a high-weight categorical feature to provide behavioral context for every classification.
* **Feature Set:** Original transaction data + `Medoid_Cluster_ID` + `Step_Hour` (Cyclical Feature).
* **Handling Imbalance:** Optimized using `class_weight='balanced'` and **Gini Impurity** to ensure the model focuses on the rare 1.21% fraud minority.