# Banking Fraud Detection: An Integrated AI Pipeline

This project implements a multi-stage machine learning pipeline to identify, classify, and assess the impact of fraudulent banking transactions. Rather than using isolated models, this system integrates **Unsupervised Learning**, **Supervised Learning**, and **Regression** into a single cohesive workflow.

---

## 🚀 Project Architecture: The Three-Stage Pipeline

### Stage 1: Behavioral Segmentation (Unsupervised Clustering)
**Method:** K-Means Clustering  
**Objective:** Group transactions into "Behavioral Profiles" without using fraud labels. This identifies the natural "shapes" of user behavior.

* **Cluster 0 (Standard):** Low-value, frequent grocery/daily shopping.
* **Cluster 1 (Premium):** High-value, rare luxury or electronics purchases.
* **Cluster 2 (Anomalous):** High-velocity, late-night ATM bursts (Suspicious).

**Integration:** The cluster assignments are injected back into the dataset as a new feature: `behavior_profile`.

---

### Stage 2: Fraud Detection (Supervised Classification)
**Method:** Random Forest Classifier  
**Objective:** Predict the binary `Is_Fraud` label.

By including the `behavior_profile` from Stage 1, the model gains **behavioral context**. 
> **Justification:** This allows the Random Forest to understand that a $5,000 transaction is "normal" for a *Premium Cluster* user but "highly suspicious" for a *Standard Cluster* user, significantly reducing false positives.

---

### Stage 3: Severity Assessment (Regression)
**Method:** Linear Regression  
**Objective:** Predict the **Potential Financial Loss** for flagged transactions.

**Logic:** In a production environment, banks need to prioritize investigations.
> **Justification:** This layer transforms the project from a simple detector into a **Risk Management Tool**. By estimating the dollar amount at risk, the system helps human investigators focus on high-impact fraud first.

---

## 🛠️ Implementation Strategy

```python
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_