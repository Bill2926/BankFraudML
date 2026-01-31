import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('dataset.csv')

# --- STEP 1: CONVERT & SYNTHESIZE ---
# We define a helper list to silently map your current data to "Domestic"
indian_keywords = [
    "Andhra", "Arunachal", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat", 
    "Haryana", "Himachal", "Jharkhand", "Karnataka", "Kerala", "Madhya", 
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", 
    "Punjab", "Rajasthan", "Sikkim", "Tamil", "Telangana", "Tripura", 
    "Uttar", "Uttarakhand", "West Bengal", "Delhi", "Mumbai", "Kolkata", 
    "Bangalore", "Chennai", "Hyderabad", "Pune", "Ahmedabad", "Surat", 
    "Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", 
    "Visakhapatnam", "Pimpri", "Patna", "Vadodara", "Ghaziabad", "Ludhiana"
]

def get_scope(location):
    location = str(location).lower()
    for keyword in indian_keywords:
        if keyword.lower() in location:
            return 'Domestic'
    return 'Domestic' # Default to Domestic if unsure, we will inject International next

# Apply the mapping
df['Transaction_Scope'] = df['Transaction_Location'].apply(get_scope)

# --- STEP 2: FORCE 10% INTERNATIONAL ---
# Randomly overwrite 10% of rows to 'International' so the model has 2 classes to learn
random_indices = df.sample(frac=0.10, random_state=42).index
df.loc[random_indices, 'Transaction_Scope'] = 'International'

# --- STEP 3: DROP THE OLD COLUMN ---
# The 'Transaction_Location' column is removed permanently
df = df.drop(columns=['Transaction_Location'])

# --- STEP 4: SAVE ---
df.to_csv('dataset.csv', index=False)