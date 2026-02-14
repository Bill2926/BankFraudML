# Used for data processing (feature Engineer) the user input into appropriate numerical values
import joblib
import numpy as np
import sqlite3
from pydantic import BaseModel
from enum import Enum
from app.modelProcess import DataProcess

class Gender(Enum):
    # Male, Female and Enterprise (as in BankSim paper)
    M = "M"
    F = "F"
    E = "E"

# class DataShape: defined as schema => FastAPI to know what to expect from the input data
class DataShape(str, BaseModel):
    step: int
    customer: str
    age: int
    gender: Gender
    merchant: str
    category: str
    amount: float

class ModelProcess:
    def __init__(self):
        self.kmed = joblib.load('models/kmedoids_model.pkl')
        self.rf = joblib.load('models/randomForest_model.pkl')
    
    def kmed_data_process(self):
        pass