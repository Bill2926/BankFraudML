import pandas as pd
import joblib
import os

# 1. Load your Model
model = joblib.load('randomForestModel.joblib')

# 2. Define Feature Order (Matches your X_train columns exactly)
feature_cols = [
    'Gender', 'Age', 'State', 'Account_Type', 'Transaction_Time', 
    'Transaction_Amount', 'Transaction_Type', 'Merchant_Category', 
    'Account_Balance', 'Transaction_Device', 'Transaction_Location', 'Device_Type'
]

# 3. Create the "Bad" vs "Good" Data
# Transaction 1: Typical daily spending (Legit)
# Transaction 2: High amount, unusual time (3 AM), low balance (Suspect)
test_data = [
    {
        'Gender': 'Male', 'Age': 60, 'State': 'Kerala', 'Account_Type': 'Savings',
        'Transaction_Time': '16:04:07', 'Transaction_Amount': 50.00, 
        'Transaction_Type': 'Transfer', 'Merchant_Category': 'Restaurant',
        'Account_Balance': 74000.00, 'Transaction_Device': 'Mobile', 
        'Transaction_Location': 'Thiruvanar', 'Device_Type': 'Smartphone'
    },
    {
        'Gender': 'Male', 'Age': 22, 'State': 'Kerala', 'Account_Type': 'Savings',
        'Transaction_Time': '03:15:00', 'Transaction_Amount': 9999999999.00, 
        'Transaction_Type': 'Transfer', 'Merchant_Category': 'Luxury',
        'Account_Balance': 200.00, 'Transaction_Device': 'Laptop', 
        'Transaction_Location': 'Overseas', 'Device_Type': 'Web'
    }
]

df_test = pd.DataFrame(test_data)

# A. Convert Time to Hour Integer
df_test['Transaction_Time'] = pd.to_datetime(df_test['Transaction_Time']).dt.hour

# B. Encode Text Columns
text_cols = ['Gender', 'State', 'Account_Type', 'Transaction_Type', 
             'Merchant_Category', 'Transaction_Device', 'Transaction_Location', 'Device_Type']

for col in text_cols:
    le = joblib.load(f'encoders/le_{col}.joblib')
    # Handle unseen labels (like 'Overseas') by defaulting to the first known class
    known_labels = set(le.classes_)
    df_test[col] = df_test[col].apply(lambda x: x if x in known_labels else le.classes_[0])
    df_test[col] = le.transform(df_test[col])

# C. Align columns and Predict
df_test = df_test[feature_cols]
predictions = model.predict(df_test)
probs = model.predict_proba(df_test)

# D. Display Results
for i, pred in enumerate(predictions):
    label = "🔴 FRAUD" if pred == 1 else "🟢 LEGIT"
    confidence = max(probs[i]) * 100
    print(f"Transaction {i+1}: {label} ({confidence:.2f}% confidence)")