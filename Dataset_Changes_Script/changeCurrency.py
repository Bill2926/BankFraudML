import pandas as pd
import numpy as np

# 1. Load the dataset
df = pd.read_csv('dataset.csv')

# --- STEP 1: RESET EVERYONE TO VND ---
df['Transaction_Currency'] = 'VND'

# --- STEP 2: DEFINE CURRENCIES AND PROBABILITIES ---
# These are the options for the foreign transactions
foreign_currencies = ['USD', 'AUD', 'CNY']

# These are the probabilities (must add up to 1.0)
# This means: 60% USD, 20% AUD, 20% CNY
weights = [0.60, 0.20, 0.20] 

# --- STEP 3: INJECT WEIGHTED FOREIGN CURRENCIES ---
# Select 15% of the rows to be foreign
random_indices = df.sample(frac=0.15, random_state=42).index

# Use the 'p' (probability) parameter to favor USD
df.loc[random_indices, 'Transaction_Currency'] = np.random.choice(
    foreign_currencies, 
    size=len(random_indices), 
    p=weights  # <--- This enforces the bias for USD
)

# --- STEP 4: SAVE ---
df.to_csv('dataset.csv', index=False)

# Verification
print("Currency Distribution:")
print(df['Transaction_Currency'].value_counts())