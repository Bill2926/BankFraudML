import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('dataset.csv')

# --- PART 1: FIX TRANSACTION AMOUNTS ---
def generate_amount(currency):
    if currency == 'VND':
        amount = np.random.randint(20000, 20000000)
        return round(amount, -3) 
    elif currency == 'USD':
        return round(np.random.uniform(5.00, 2000.00), 2)
    elif currency == 'AUD':
        return round(np.random.uniform(10.00, 3000.00), 2)
    elif currency == 'CNY':
        return round(np.random.uniform(50.00, 10000.00), 2)
    else:
        return 100.00

df['Transaction_Amount'] = df.apply(lambda row: generate_amount(row['Transaction_Currency']), axis=1)

# --- PART 1.5: FIX ACCOUNT BALANCES ---
def generate_balance(amount, currency):
    # A realistic balance is usually 2x to 50x the current transaction
    multiplier = np.random.uniform(1.1, 50.0)
    balance = amount * multiplier
    
    if currency == 'VND':
        return round(balance, -3) # Round to nearest 1,000
    else:
        return round(balance, 2)

# Ensure balance is always logically higher than the transaction amount
df['Account_Balance'] = df.apply(lambda row: generate_balance(row['Transaction_Amount'], row['Transaction_Currency']), axis=1)

# --- PART 2: REPLACE BANK BRANCH WITH BANK NAME ---
vietnam_banks = ["Vietcombank", "BIDV", "Techcombank", "Agribank", "MB Bank", "VPBank", "TPBank", "ACB", "Sacombank", "VIB"]
intl_banks_in_vn = ["Shinhan Bank", "HSBC", "Standard Chartered", "UOB", "CitiBank", "ANZ"]
all_banks = vietnam_banks + intl_banks_in_vn

df['Bank_Name'] = np.random.choice(all_banks, size=len(df))

# --- PART 3: SWAP THE COLUMNS ---
if 'Bank_Branch' in df.columns:
    col_index = df.columns.get_loc('Bank_Branch')
    df = df.drop(columns=['Bank_Branch'])
    df.insert(col_index, 'Bank_Name', df.pop('Bank_Name'))

# --- PART 4: SAVE ---
df.to_csv('dataset.csv', index=False)

print("SUCCESS: Amounts, Balances, and Bank Names updated.")
print(df[['Transaction_Amount', 'Account_Balance', 'Transaction_Currency']].head(10))