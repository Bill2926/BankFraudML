# Used for data processing (feature Engineer) the user input into appropriate numerical values
import joblib
import numpy as np
import sqlite3
import shap
from pydantic import BaseModel
from enum import Enum

class Gender(Enum):
    # Male, Female and Enterprise (as in BankSim paper)
    M = "M"
    F = "F"
    E = "E"

# class DataShape: defined as schema => FastAPI to know what to expect from the input data
class DataShape(BaseModel):
    step: int
    customer: str
    age: int
    gender: Gender
    merchant: str
    category: str
    amount: float

class ModelProcess:
    def __init__(self):
        self.kmed = joblib.load('app/models/kmedoids_model.pkl')
        self.rf = joblib.load('app/models/randomForest_model.pkl')
        self.scaler = joblib.load('app/models/scaler.pkl')
        self.explainer = shap.TreeExplainer(self.rf)
        self.feature_names = [
            'vel_3h', 'vel_6h', 'vel_12h', 
            'freq_3h', 'freq_6h', 'freq_12h',
            'category_score', 'merchant_score', 'age_score', 
            'isEnterprise', 'clusterID'
        ]

    def kmed_processing(self, ds: DataShape):
        """
        Velocity: 3-6-12h, for this step, I substracted the lastest step (which is 179 on the dataset)
        => which mean the velocity (amount spent) on the 3-6-12 steps away from the 179th step
        the same applied to frequency
        
        This function returns entry with cluster id labeled
        """

        conn = sqlite3.connect('bank_data.db')
        cursor = conn.cursor()

        # Velocities and Frequencies
        vel_result = []     # in order 3-6-12
        freq_result = []

        for i in [3, 6, 12]:
            cursor.execute(
                    f"""
                    SELECT SUM(amount) AS vel_3h
                    FROM BankSim
                    WHERE customer = '{ds.customer}'
                    AND step BETWEEN (179 - {i}) AND 179;
                    """
                )
            vel_result.append(cursor.fetchone()[0])    # fetchone always returns tuple even when there was only 1 returned value

            cursor.execute(
                    f"""
                    SELECT COUNT(amount) AS vel_3h
                    FROM BankSim
                    WHERE customer = '{ds.customer}'
                    AND step BETWEEN (179 - {i}) AND 179;
                    """
                )
            freq_result.append(cursor.fetchone()[0])
        
        # Merchant and Category risk scores
        cursor.execute(
            f"""
                SELECT risk_mean_score AS score
                FROM merchant_risk_lookup
                WHERE merchant = '{ds.merchant}';
            """
        )
        merchant_score = cursor.fetchone()[0]
        cursor.execute(
            f"""
                SELECT risk_mean_score AS score
                FROM category_risk_lookup
                WHERE category = '{ds.category}';
                """
            )
        category_score = cursor.fetchone()[0]

        # Age risk score and isEnterprise
        # age_map = {
        #     "'0'": "<=18", "'1'": "19-25", "'2'": "26-35", "'3'": "36-45", 
        #     "'4'": "46-55", "'5'": "56-65", "'6'": ">65", "'U'": "Unknown"
        # }


        cursor.execute(
            f"""
                SELECT risk_mean_score AS score
                FROM age_risk_lookup
                WHERE age = '{ds.age}';
                """
            )
        age_score = cursor.fetchone()[0]
        isEnterprise = 1 if ds.gender == Gender.E else 0

        cursor.close()

        # Now wrap every values into 2D Numpy Arr
        entry = []
        entry = [
            *vel_result,    # Unpacks the arr 
            *freq_result,
            category_score,
            merchant_score,
            age_score,
            isEnterprise
        ]

        entry_2d = np.array(entry).reshape(1, -1)

        # 2. Scale the data (CRITICAL for K-Medoids distance)
        entry_scaled = self.scaler.transform(entry_2d)

        # 3. Predict Cluster
        clusterID = self.kmed.predict(entry_scaled)
        cluster_feature = clusterID.reshape(1, -1)

        # 4. Combine Scaled Features + Cluster ID for the Random Forest
        entry_with_cluster = np.hstack([entry_scaled, cluster_feature])

        print(entry_with_cluster)

        return entry_with_cluster
    
    def get_reasons(self, entry_with_cluster):
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(entry_with_cluster)
        
        # SHAP output varies by version/model:
        # 1. If it's a list (usually older versions or specific RF types)
        if isinstance(shap_values, list):
            # Index [1] is the positive class (Fraud)
            contributions = shap_values[1][0]
        # 2. If it's a single array (newer versions or certain tree models)
        else:
            # If the array is 3D (samples, features, classes), get class 1
            if len(shap_values.shape) == 3:
                contributions = shap_values[0, :, 1]
            # If it's 2D, it might already be the positive class or 
            # requires indexing differently based on the explainer
            else:
                contributions = shap_values[0]

        # Map features and return top 3 as before
        reason_map = dict(zip(self.feature_names, contributions))
        top_reasons = sorted(reason_map.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        
        return top_reasons

    def rf_processing(self, entry_with_cluster):
        prediction = self.rf.predict(entry_with_cluster)
        # Maybe will change to predicted proba for more thresold constrolling
        # print(f"DEBUG: Type of prediction is {type(prediction)}")

        reasons = self.get_reasons(entry_with_cluster)

        # 3. PRINT TO TERMINAL ONLY
        print("\n" + "="*40)
        print(f"TERMINAL LOG - PREDICTION: {'FRAUD' if prediction == 1 else 'NORMAL'}")
        print("-" * 40)
        for feature, impact in reasons:
            direction = "Towards FRAUD" if impact > 0 else "Towards NORMAL"
            print(f"-> {feature:15} | Impact: {impact:8.4f} | {direction}")
        print("="*40 + "\n")

        return prediction.item()
    
    def predict(self, ds: DataShape):
        return self.rf_processing(self.kmed_processing(ds))