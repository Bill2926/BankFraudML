import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('dataset.csv')

# --- PART 1: FIX TRANSACTION AMOUNTS ---
# Function to generate realistic amounts based on currency
def generate_amount(currency):
    if currency == 'VND':
        # VND: Usually between 20,000 and 20,000,000. No decimals.
        # We round to the nearest 1,000 for realism (e.g., 50,000, not 50,231)
        amount = np.random.randint(20000, 20000000)
        return round(amount, -3) 
    elif currency == 'USD':
        # USD: Between $5.00 and $2,000.00
        return round(np.random.uniform(5.00, 2000.00), 2)
    elif currency == 'AUD':
         # AUD: Between $10.00 and $3,000.00
        return round(np.random.uniform(10.00, 3000.00), 2)
    elif currency == 'CNY':
        # CNY: Between ¥50.00 and ¥10,000.00
        return round(np.random.uniform(50.00, 10000.00), 2)
    else:
        # Fallback
        return 100.00

# Apply the function row by row
df['Transaction_Amount'] = df.apply(lambda row: generate_amount(row['Transaction_Currency']), axis=1)

# --- PART 2: REPLACE BANK BRANCH WITH BANK NAME ---
# Define Bank Lists
vietnam_banks = [
    "Vietcombank", "BIDV", "Techcombank", "Agribank", "MB Bank", 
    "VPBank", "TPBank", "ACB", "Sacombank", "VIB"
]
intl_banks_in_vn = [
    "Shinhan Bank", "HSBC", "Standard Chartered", "UOB", "CitiBank", "ANZ"
]

# Combine them (You can adjust weights if you want intl banks to be rarer)
all_banks = vietnam_banks + intl_banks_in_vn

# Generate new Bank Names
df['Bank_Name'] = np.random.choice(all_banks, size=len(df))

# --- PART 3: SWAP THE COLUMNS ---
# We find where 'Bank_Branch' is, drop it, and put 'Bank_Name' there.
if 'Bank_Branch' in df.columns:
    col_index = df.columns.get_loc('Bank_Branch')
    df = df.drop(columns=['Bank_Branch'])
    df.insert(col_index, 'Bank_Name', df.pop('Bank_Name'))
else:
    # If Bank_Branch doesn't exist, just leave Bank_Name at the end
    pass

# --- PART 4: SAVE ---
df.to_csv('dataset.csv', index=False)

print("Transaction Amounts updated to realistic VND/Foreign values.")
print("Bank_Branch replaced with Bank_Name (Vietnamese & Intl banks).")
print(df[['Transaction_Amount', 'Transaction_Currency', 'Bank_Name']].head(10))