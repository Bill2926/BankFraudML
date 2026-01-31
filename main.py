import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import datetime

df = pd.read_csv('dataset.csv', encoding='utf-8')
# df.head()   # Return the first n rows head(n)
# print(df.to_string())   # to_string to return entire df
# print(df.head())

# Drop noise columns
df_clean = df.drop(
    columns=[
        'Customer_ID', 
        'Customer_Name', 
        'Transaction_ID', 
        'Merchant_ID', 
        'Bank_Branch', 
        'Customer_Contact', 
        'Customer_Email',
        'Transaction_Description',
        'Transaction_Currency',
        'Transaction_Date',
        'City',
])

# 1. Convert to datetime objects first
df_clean['Transaction_Time'] = pd.to_datetime(df_clean['Transaction_Time'], format='%H:%M:%S').dt.hour

text_cols = [
    'Transaction_Type', 
    'Merchant_Category', 
    'Transaction_Device', 
    'Device_Type',
    'Gender',
    'State',
    'Account_Type',
    'Transaction_Location'
]

df_clean = df_clean.sample(n=15000, random_state=17)

# Encode the text
folder_name = 'encoders'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

encoder_dict = {}
for c in text_cols:
    le = LabelEncoder() # Text-to-Number translator for each column
    df_clean[c] = le.fit_transform(df_clean[c])
    file_path = os.path.join(folder_name, f'le_{c}.joblib') # Save inside designated folder
    joblib.dump(le, file_path)  # Save the encoder as a file for later reversed engineer

# The criterias (features) - iloc[rows chosen, cols chosen]
X = df_clean.iloc[:, 0:-1]  # Every cols except the last col
# The target - the possibleFraud column
y= df_clean.iloc[:, -1] # Only the last col


# Apply SMOTE to generate new synthetic fraud examples based on REAL fraud patterns
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled = X_resampled.round().astype(int)   # To avoid synthetic value being float like 1.5

# Split the dataset into train (80%) and test (20%) set
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, 
    y_resampled, 
    random_state=42, 
    test_size=0.2
)

rf = RandomForestClassifier(
    class_weight='balanced',
    max_depth=15,
    min_samples_leaf=10,
    min_samples_split=20,
    verbose=1   # Show progress
)
rf.fit(X_train, y_train)    # Training part
joblib.dump(rf, "randomForestModel.joblib")
y_pred = rf.predict(X_test)
score = rf.score(X_test, y_test)
print(score)
print(classification_report(y_test, y_pred, digits=3))
