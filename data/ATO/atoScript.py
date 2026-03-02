import pandas as pd

# Configuration
input_csv = 'data/ato.csv'
output_csv = 'data/processed/stratified_eda_sample.csv'
chunk_size = 50000  # Reduced slightly for better stability with Python engine
normal_sample_rate = 0.01

ato_rows = []
attack_ip_rows = []
normal_sample_rows = []

print("Starting the hunt through the 8GB file...")

try:
    # 1. Added engine='python' to handle the ParserError
    # 2. Added on_bad_lines='skip' to ignore rows with formatting issues
    reader = pd.read_csv(
        input_csv, 
        chunksize=chunk_size, 
        engine='python', 
        on_bad_lines='skip'
    )

    for i, chunk in enumerate(reader):
        # Filter for ATOs
        ato = chunk[chunk['Is Account Takeover'] == True]
        ato_rows.append(ato)
        
        # Filter for Attack IPs (excluding those that are also ATOs)
        attack_ip = chunk[(chunk['Is Attack IP'] == True) & (chunk['Is Account Takeover'] == False)]
        attack_ip_rows.append(attack_ip)
        
        # 1% Sample of Normal Data
        normal = chunk[(chunk['Is Attack IP'] == False) & (chunk['Is Account Takeover'] == False)]
        if not normal.empty:
            normal_sample_rows.append(normal.sample(frac=normal_sample_rate))
        
        if i % 10 == 0:
            print(f"Processed approx {i * chunk_size} rows...")

    print("Finalizing the sample...")
    df_final = pd.concat(ato_rows + attack_ip_rows + normal_sample_rows)
    
    # Shuffle and Save
    df_final = df_final.sample(frac=1).reset_index(drop=True)
    df_final.to_csv(output_csv, index=False)
    
    print(f"Done! Saved {len(df_final)} rows to {output_csv}")
    print(df_final['Is Account Takeover'].value_counts())

except Exception as e:
    print(f"Script failed: {e}")