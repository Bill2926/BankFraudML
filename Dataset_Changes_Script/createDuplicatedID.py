import pandas as pd
import numpy as np

# 1. Load your dataset
file_path = 'dataset.csv'
df = pd.read_csv(file_path) 

# Optional: Create a backup before overwriting
# df.to_csv('dataset_backup.csv', index=False)

# 2. Identify Repeat Customers
# We pick 500 unique IDs that already exist in the data
repeat_ids = df['Customer_ID'].sample(500).values

# 3. Create Overlap
# We pick 5,000 random rows and force them to use those 500 IDs
# This creates "history" for those specific customers
random_indices = df.sample(5000).index
df.loc[random_indices, 'Customer_ID'] = np.random.choice(repeat_ids, 5000)

# 4. Overwrite the original file
df.to_csv(file_path, index=False)

print(f"Success! {file_path} has been updated with repeating Customer_IDs.")
print(f"Unique IDs remaining: {df['Customer_ID'].nunique()}")