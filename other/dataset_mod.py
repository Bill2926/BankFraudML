import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('dataset.csv')
print(f"Starting with {len(df)} rows.")

# ---------------------------------------------------------
# STEP 1: CURRENCY LOGIC (US DEFAULT)
# ---------------------------------------------------------
# Set default to USD and inject minor foreign currencies for International transactions
df['Transaction_Currency'] = 'USD'
foreign_currencies = ['CAD', 'GBP', 'EUR']
weights = [0.40, 0.30, 0.30] 

# 10% of transactions are international
random_indices_curr = df.sample(frac=0.10, random_state=42).index
df.loc[random_indices_curr, 'Transaction_Currency'] = np.random.choice(
    foreign_currencies, 
    size=len(random_indices_curr), 
    p=weights
)

# ---------------------------------------------------------
# STEP 2: BANK AND AMOUNT LOGIC (US SCALING)
# ---------------------------------------------------------
def generate_amount(currency):
    # US Transactions usually range from small coffee to large rent/mortgage
    if currency == 'USD':
        return round(np.random.uniform(1.00, 5000.00), 2) 
    elif currency == 'EUR': return round(np.random.uniform(5.00, 4000.00), 2)
    elif currency == 'GBP': return round(np.random.uniform(5.00, 3500.00), 2)
    elif currency == 'CAD': return round(np.random.uniform(5.00, 6000.00), 2)
    return 100.00

df['Transaction_Amount'] = df.apply(lambda row: generate_amount(row['Transaction_Currency']), axis=1)

def generate_balance(amount, currency):
    # Standard US bank balance multiplier
    multiplier = np.random.uniform(2.0, 100.0)
    balance = amount * multiplier
    return round(balance, 2)

df['Account_Balance'] = df.apply(lambda row: generate_balance(row['Transaction_Amount'], row['Transaction_Currency']), axis=1)

# Major US Banks
us_banks = ["JPMorgan Chase", "Bank of America", "Wells Fargo", "Citigroup", "U.S. Bancorp", "PNC Financial Services", "Truist Financial", "Goldman Sachs", "Capital One", "TD Bank"]
df['Bank_Name'] = np.random.choice(us_banks, size=len(df))

if 'Bank_Branch' in df.columns:
    df = df.drop(columns=['Bank_Branch'])

# ---------------------------------------------------------
# STEP 3 & 4: LOCATION & CITY LOGIC (US CITIES)
# ---------------------------------------------------------
df['Transaction_Scope'] = 'Domestic'
intl_mask = df['Transaction_Currency'] != 'USD'
df.loc[intl_mask, 'Transaction_Scope'] = 'International'

# US Cities for Domestic
us_cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
df['City'] = np.random.choice(us_cities, size=len(df))

# Intl Cities for International scope
intl_cities = ["London", "Paris", "Toronto", "Berlin", "Tokyo", "Mexico City"]
df.loc[intl_mask, 'City'] = np.random.choice(intl_cities, size=intl_mask.sum())

if 'State' in df.columns:
    # Adding US States mapping based on cities could be complex, 
    # so we'll just assign a random major state for simplicity.
    us_states = ["NY", "CA", "IL", "TX", "AZ", "PA", "TX", "CA", "TX", "CA"]
    df['State'] = np.random.choice(us_states, size=len(df))

# ---------------------------------------------------------
# STEP 5: ID DUPLICATION (HISTORY GENERATION)
# ---------------------------------------------------------
# Keep your existing logic for duplicates to simulate customer history
repeat_ids = df['Customer_ID'].sample(min(5000, len(df))).values
random_indices_dup = df.sample(min(150000, len(df))).index
df.loc[random_indices_dup, 'Customer_ID'] = np.random.choice(repeat_ids, len(random_indices_dup))

# ---------------------------------------------------------
# FINAL SAVE
# ---------------------------------------------------------
df.to_csv('dataset_usa_modified.csv', index=False)
print("SUCCESS: USA-based script executed. Output saved to 'dataset_usa_modified.csv'")