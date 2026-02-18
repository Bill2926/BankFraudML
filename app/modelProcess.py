# Used for data processing (feature Engineer) the user input into appropriate numerical values
import joblib
import numpy as np
import sqlite3
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

    def predict(self, ds: DataShape):
        return self.rf_processing(self.kmed_processing(ds))

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
        entry = np.array(entry).reshape(1, -1)
        clusterID = self.kmed.predict(entry)
        cluster_feature = clusterID.reshape(1, -1)
        entry_with_cluster = np.hstack([entry, cluster_feature])

        return entry_with_cluster
    
    def rf_processing(self, entry_with_cluster):
        prediction = self.rf.predict(entry_with_cluster)
        # Maybe will change to predicted proba for more thresold constrolling
        # print(f"DEBUG: Type of prediction is {type(prediction)}")
        return prediction.item()