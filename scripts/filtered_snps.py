import pandas as pd
from pathlib import Path
import json
import os


def main():
    CURR_DIR = Path.cwd()
    RESULTS = CURR_DIR / 'results'
    MATRICES = Path(
        '/data/scratch/projects/punim1703/false_negative/old_tbprof/results/matrices')

    combined_path = RESULTS / 'essentials/combined_snps.json'
    if not combined_path.exists():
        print(f"Error: {combined_path} not found.")
        return

    with open(combined_path, 'r') as f:
        curr_snps = json.load(f)

    for drug in curr_snps:
        drug_lower = drug.lower()
        csv_path = MATRICES / drug_lower / f'filtered_{drug_lower}_snps.csv'

        print(f"Checking: {csv_path}")

        if csv_path.exists() and os.path.getsize(csv_path) > 0:
            try:
                f_snps = pd.read_csv(csv_path)

                if 'log_or' in f_snps.columns and 'snp' in f_snps.columns:
                    f_snps = f_snps[f_snps['log_or'].astype(
                        str).str.lower() != 'inf']

                    snps_list = f_snps['snp'].tolist()
                    for snp in snps_list:
                        # Ensure the drug key exists in our dict
                        if drug not in curr_snps:
                            curr_snps[drug] = {}
                        curr_snps[drug][snp] = 1
                else:
                    print(
                        f"  ! Warning: Required columns missing in {csv_path}")

            except Exception as e:
                print(f"  ! Error reading {csv_path}: {e}")
        else:
            print(f"  ! Skipping: {csv_path} is empty or does not exist.")

    # Save the updated dictionary
    output_path = RESULTS / 'filtered_combined_snps.json'
    with open(output_path, 'w') as f:
        json.dump(curr_snps, f, indent=4)
    print(f"Successfully saved results to {output_path}")


if __name__ == "__main__":
    main()
