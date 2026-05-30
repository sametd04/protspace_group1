import pandas as pd
import numpy as np

def normalize_id(id_str):
    if pd.isna(id_str): return ""
    return str(id_str).split('|')[1] if '|' in str(id_str) else str(id_str).split('.')[0]

def detect_signal_peptide(row):
    """Calculates signal peptide status based on sequence lengths."""
    full = str(row.get('full_seq', ''))
    mature = str(row.get('mature_seq', ''))
    
    # Simple logic: If full sequence exists, mature exists, and full is longer
    if len(full) > len(mature) and len(mature) > 0:
        return 'yes'
    return 'no'

def main():
    print("Building annotations and engineering SP features...")
    
    # 1. Load Excel
    df = pd.read_excel('data/3FTx/3FTx_data.xlsx')
    id_col = 'identifier' if 'identifier' in df.columns else df.columns[0]
    df['identifier'] = df[id_col].apply(normalize_id)
    
    # 2. Engineer the Feature
    # This guarantees 'yes' and 'no' values exist if your sequences vary in length
    df['signal_peptide'] = df.apply(detect_signal_peptide, axis=1)
    
    print(f"Engineered Signal Peptide variance: {df['signal_peptide'].value_counts().to_dict()}")

    # 3. Save
    df.to_csv('data/combined_annotations.csv', index=False)
    
    # 4. Save a specialized truth file for the pipeline
    df[['identifier', 'signal_peptide']].to_csv('data/combined_signal_peptide.csv', index=False)
    print("Files saved successfully.")

if __name__ == "__main__":
    main()