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
keep_cols = ['step', 'customer', 'age', 'gender', 'merchant', 'category', 'amount']
df = df[keep_cols]

df.to_sql("BankSim", conn, if_exists='replace', index=False)