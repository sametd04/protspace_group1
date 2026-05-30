import pandas as pd
import math
import os

def main():
    print("Extracting FASTA from 3FTx_data.xlsx...")
    
    # 1. Load the Excel file
    excel_path = 'data/3FTx/3FTx_data.xlsx'
    df = pd.read_excel(excel_path)
    
    # 2. Determine the identifier column
    id_col = 'identifier' if 'identifier' in df.columns else df.columns[0]
    
    fasta_lines = []
    fallback_count = 0
    
    # 3. Iterate and apply the sequence rule
    for _, row in df.iterrows():
        identifier = str(row[id_col])
        full_seq = str(row.get('full_seq', 'nan'))
        mature_seq = str(row.get('mature_seq', 'nan'))
        
        # Check if full_seq is valid
        if full_seq.lower() != 'nan' and len(full_seq.strip()) > 0:
            seq = full_seq.strip()
        # Fallback to mature_seq
        elif mature_seq.lower() != 'nan' and len(mature_seq.strip()) > 0:
            seq = mature_seq.strip()
            fallback_count += 1
        else:
            print(f"Warning: No valid sequence found for {identifier}")
            continue
            
        fasta_lines.append(f">{identifier}\n{seq}")
        
    # 4. Save to FASTA
    os.makedirs('data/3FTx', exist_ok=True)
    fasta_path = 'data/3FTx/sequences.fasta'
    with open(fasta_path, 'w') as f:
        f.write("\n".join(fasta_lines))
        
    print(f"Successfully extracted {len(fasta_lines)} sequences to {fasta_path}.")
    print(f"Used 'mature_seq' as fallback for {fallback_count} proteins.")

if __name__ == "__main__":
    main()