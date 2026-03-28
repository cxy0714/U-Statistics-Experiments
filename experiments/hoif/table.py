import json
import pandas as pd
import numpy as np
import glob
import os

# Find the latest benchmark file in the results directory
list_of_files = glob.glob('experiments/hoif/results/benchmark_20260327_155952.json')
INPUT_FILE = max(list_of_files, key=os.path.getctime)
base_name = os.path.splitext(os.path.basename(INPUT_FILE))[0]
OUTPUT_FILE = f"experiments/hoif/results/{base_name}_summary.txt"

def generate_table(input_path, output_path):
    # 1. Load the JSON data
    with open(input_path, "r") as f:
        data = json.load(f)
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(data)
    
    # 3. Calculate the mean of the 10 runs for each row
    df['avg_time'] = df['times'].apply(np.mean)
    
    # 4. Pivot the table: rows = order, columns = size
    # This aligns the data exactly as you requested
    pivot_df = df.pivot(index='order', columns='size', values='avg_time')
    
    # 5. Format the header and the table string
    header = (
        f"Source: {os.path.basename(input_path)}\n"
        f"Metric: Average Execution Time (seconds)\n"
        f"{'='*60}\n"
    )
    
    # Use 5 decimal places as requested
    table_str = pivot_df.to_string(float_format='%.5f')
    
    # 6. Save to TXT file
    with open(output_path, "w") as f:
        f.write(header)
        f.write(table_str)
    
    print(f"Success! Table generated from {os.path.basename(input_path)}")
    print(f"Results saved to: {output_path}")
    print("\nPreview:")
    print(table_str)

if __name__ == "__main__":
    generate_table(INPUT_FILE, OUTPUT_FILE)