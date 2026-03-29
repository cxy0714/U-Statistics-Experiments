import json
import pandas as pd
import numpy as np

def generate_latex_table(json_file, output_file):
    """
    Generate LaTeX table from benchmark results
    
    Args:
        json_file: path to JSON results file
        output_file: path to output txt file
    """
    # Load results
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    df = pd.DataFrame(results)
    
    # Calculate mean times for each p value
    summary = df.groupby('p').agg({
        'u_time': lambda x: x.dropna().mean() if len(x.dropna()) > 0 else None,
        'cugraph_time': lambda x: x.dropna().mean() if len(x.dropna()) > 0 else None,
    }).reset_index()
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("\\begin{table}[htbp]")
    latex_lines.append("\\centering")
    latex_lines.append("\\caption{Triangle Counting Computation Time Comparison (seconds)}")
    latex_lines.append("\\label{tab:triangle_counting}")
    latex_lines.append("\\begin{tabular}{c|cc}")
    latex_lines.append("\\hline")
    latex_lines.append("Edge Probability ($p$) & U-statistics & CuGraph \\\\")
    latex_lines.append("\\hline")
    
    for _, row in summary.iterrows():
        p_val = row['p']
        u_time = row['u_time']
        cugraph_time = row['cugraph_time']
        
        # Format p value
        p_str = f"{p_val:.3f}"
        
        # Format U-statistics time
        if pd.notna(u_time):
            u_str = f"{u_time:.4f}"
        else:
            u_str = "---"  # Common LaTeX notation for missing/unavailable data
        
        # Format CuGraph time
        if pd.notna(cugraph_time):
            cg_str = f"{cugraph_time:.4f}"
        else:
            cg_str = "OOM"  # Out Of Memory - standard notation
        
        latex_lines.append(f"{p_str} & {u_str} & {cg_str} \\\\")
    
    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
    
    # Add note about OOM
    latex_lines.append("")
    latex_lines.append("% Note: OOM indicates Out Of Memory error")
    latex_lines.append("% ---  indicates data not available")
    
    # Write to file
    latex_content = "\n".join(latex_lines)
    with open(output_file, 'w') as f:
        f.write(latex_content)
    
    print(f"LaTeX table saved to: {output_file}")
    print("\n" + "="*60)
    print(latex_content)
    print("="*60)
    
    return latex_content

if __name__ == "__main__":
    json_file = "experiments/motif_count/results/GPU_triangle_benchmark_20251115_162807.json"
    output_file = "experiments/motif_count/results/triangle_comparison_table.txt"
    
    generate_latex_table(json_file, output_file)