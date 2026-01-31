import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('dataset.csv')

# --- STEP 1: DROP THE STATE COLUMN ---
if 'State' in df.columns:
    df = df.drop(columns=['State'])
    print("Column 'State' has been dropped.")
else:
    print("Column 'State' not found (maybe already deleted).")

# --- STEP 2: REPLACE CITIES WITH VIETNAMESE CITIES ---
# List of major cities and provinces in Vietnam
vn_cities = [
    "Ho Chi Minh City", "Hanoi", "Da Nang", "Hai Phong", "Can Tho",
    "Nha Trang", "Vung Tau", "Hue", "Da Lat", "Bien Hoa", 
    "Buon Ma Thuot", "Quy Nhon", "Vinh", "Ha Long", "Phan Thiet"
]

# Check if 'City' column exists, if not create it (or overwrite it)
# We assign a random Vietnamese city to every row
df['City'] = np.random.choice(vn_cities, size=len(df))

# --- OPTIONAL: HANDLE INTERNATIONAL TRANSACTIONS ---
# If you have the 'Transaction_Scope' column we created earlier, 
# you might want to ensure 'International' rows have foreign cities.
if 'Transaction_Scope' in df.columns:
    intl_cities = ["New York", "London", "Singapore", "Tokyo", "Bangkok", "Berlin"]
    
    # Find rows where Scope is International
    intl_mask = df['Transaction_Scope'] == 'International'
    
    # Overwrite City for those specific rows only
    df.loc[intl_mask, 'City'] = np.random.choice(intl_cities, size=intl_mask.sum())

# --- STEP 3: SAVE ---
df.to_csv('dataset.csv', index=False)

print("City column updated to Vietnamese locations.")
print(df[['City']].head(10))