import pandas as pd
import sqlite3

# Set up SQLite db
conn = sqlite3.connect('bank_data.db')
df = pd.read_csv('data/lastest.csv', encoding="utf-8")

# Remove the uneccesary single quotes
df['category'] = df['category'].str.strip("'").str.split('_').str[1]
df['customer'] = df['customer'].str.strip("'")
df['age'] = df['age'].str.strip("'")
df['gender'] = df['gender'].str.strip("'")
df['merchant'] = df['merchant'].str.strip("'")
keep_cols = ['step', 'customer', 'age', 'gender', 'merchant', 'category', 'amount', 'fraud']
df = df[keep_cols]

df.to_sql("BankSim", conn, if_exists='replace', index=False)

# Create composite index (customer, step) ~ customerID and their transaction time
cursor = conn.cursor()
index_query = """
CREATE INDEX IF NOT EXISTS idx_customer_time 
ON BankSim (customer, step);
"""

merchant_lookup = """
CREATE TABLE merchant_risk_lookup AS
SELECT 
    merchant, 
    AVG(fraud) AS risk_mean_score,
    COUNT(*) AS total_transactions -- Useful for knowing if the score is reliable
FROM BankSim
GROUP BY merchant;
"""

category_lookup = """
CREATE TABLE category_risk_lookup AS
SELECT 
    category, 
    AVG(fraud) AS risk_mean_score,
    COUNT(*) AS total_transactions -- Useful for knowing if the score is reliable
FROM BankSim
GROUP BY category;
"""

gender_lookup = """
CREATE TABLE gender_risk_lookup AS
SELECT 
    gender, 
    AVG(fraud) AS risk_mean_score,
    COUNT(*) AS total_transactions -- Useful for knowing if the score is reliable
FROM BankSim
GROUP BY gender;
"""

age_lookup = """
CREATE TABLE age_risk_lookup AS
SELECT 
    age, 
    AVG(fraud) AS risk_mean_score,
    COUNT(*) AS total_transactions -- Useful for knowing if the score is reliable
FROM BankSim
GROUP BY age;
"""

cursor.execute(index_query)
cursor.execute(merchant_lookup)
cursor.execute(category_lookup)
cursor.execute(gender_lookup)
cursor.execute(age_lookup)
conn.commit()
conn.close()