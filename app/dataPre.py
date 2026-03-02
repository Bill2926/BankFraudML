import pandas as pd
import sqlite3
import os

# --- 1. SETUP PATHS & DIRECTORIES ---
input_path = "data/banksim.csv"
output_dir = "data/processed"
output_csv = os.path.join(output_dir, "banksim_cleaned.csv")
db_path = "bank_data.db"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- 2. DATA CLEANING ---
print("Loading and cleaning data...")
df = pd.read_csv(input_path, encoding="utf-8")

# Remove redundant single quotes and clean category names
cols_to_strip = ['category', 'customer', 'age', 'gender', 'merchant']
for col in cols_to_strip:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip("'")

# Specific cleaning for 'category' (e.g., 'es_transportation' -> 'transportation')
if 'category' in df.columns:
    df['category'] = df['category'].str.split('_').str[1]

# Feature Engineering: Age labels and Hour conversion
age_map = {
    "0": "<=18", "1": "19-25", "2": "26-35", "3": "36-45", 
    "4": "46-55", "5": "56-65", "6": ">65", "U": "Unknown"
}
df['age_labeled'] = df['age'].map(age_map)

if 'step' in df.columns:
    df["hour_of_day"] = df["step"].apply(lambda x: x % 24)

# Filter columns for the database
keep_cols = ['step', 'hour_of_day', 'customer', 'age', 'age_labeled', 
             'gender', 'merchant', 'category', 'amount', 'fraud']
df_final = df[keep_cols].copy()

# Save the processed CSV
df_final.to_csv(output_csv, index=False)
print(f"Cleaned CSV saved to: {output_csv}")

# --- 3. SQLITE DATABASE SETUP ---
print("Setting up SQLite database and risk lookups...")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Upload cleaned data to the main table
df_final.to_sql("BankSim", conn, if_exists='replace', index=False)

# Create Index for faster queries on customer patterns
index_query = "CREATE INDEX IF NOT EXISTS idx_customer_time ON BankSim (customer, step);"
cursor.execute(index_query)

# Define Lookup Queries
lookups = {
    "merchant_risk_lookup": "merchant",
    "category_risk_lookup": "category",
    "gender_risk_lookup": "gender",
    "age_risk_lookup": "age"
}

for table, column in lookups.items():
    cursor.execute(f"DROP TABLE IF EXISTS {table}")
    query = f"""
    CREATE TABLE {table} AS
    SELECT 
        {column}, 
        AVG(fraud) AS risk_mean_score,
        COUNT(*) AS total_transactions
    FROM BankSim
    GROUP BY {column};
    """
    cursor.execute(query)
    print(f"Created lookup table: {table}")

conn.commit()
conn.close()