import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_process_data(n):
    """
    Load and process JSON data file
    
    Args:
        n (int or str): Desired 'n' value to filter
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Load JSON file
    path = os.path.join("experiments", "motif_count", "results", "motif_size_4_results_20250630_063408.json")
    with open(path, "r") as f:
        data = json.load(f)
    
    # Convert to list if single data point
    if isinstance(data, dict):
        data = [data]
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Filter data for given n (convert n to string if needed for matching)
    df_filtered = df[df['n'].astype(str) == str(n)].copy()
    
    # Calculate speedup ratio (igraph time / our algorithm time)
    df_filtered['speedup_ratio'] = df_filtered['ig_time'] / df_filtered['u_time']
    
    # Calculate single-core performance comparison (speedup / 64)
    df_filtered['single_core_speedup'] = df_filtered['speedup_ratio'] / 64
    
    return df_filtered

def create_speedup_plot(df, n):
    """
    Create speedup ratio plot with p (edge density) as x-axis
    
    Args:
        df (pd.DataFrame): Processed data
        n (int or str): The value of n used
    """
    plt.figure(figsize=(10, 6))
    
    # Sort by p value for smooth line plotting
    df_sorted = df.sort_values('p')
    
    # Plot actual speedup ratio
    plt.plot(df_sorted['p'], df_sorted['speedup_ratio'], 
             'o-', linewidth=2, markersize=6, 
             label='Actual Speedup (64-core vs 1-core)', 
             color='blue')
    
    # Plot 64x speedup baseline (perfect parallelization)
    plt.axhline(y=64, color='red', linestyle='--', linewidth=2, 
                label='Perfect Parallelization (64x)')
    
    # Plot single-core performance comparison
    plt.plot(df_sorted['p'], df_sorted['single_core_speedup'], 
             's-', linewidth=2, markersize=6, 
             label='Single-core Performance Ratio', 
             color='green')
    
    plt.xlabel('Edge Density (p)')
    plt.ylabel('Speedup Ratio')
    plt.title(f'Algorithm Performance Comparison (n={n})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Use log scale for better visualization
    
    # Add text annotation for interpretation
    plt.text(0.02, 0.98, 'Above red line: Better single-core performance', 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return plt

def generate_summary_stats(df, n):
    """
    Generate summary statistics and export as LaTeX table
    
    Args:
        df (pd.DataFrame): Processed data
        n (int or str): The value of n (used for filename)
    """
    print("=" * 50)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Number of test cases: {len(df)}")
    print(f"Edge density range: {df['p'].min():.3f} - {df['p'].max():.3f}")
    print(f"Average speedup ratio: {df['speedup_ratio'].mean():.2f}x")
    print(f"Maximum speedup ratio: {df['speedup_ratio'].max():.2f}x")
    print(f"Minimum speedup ratio: {df['speedup_ratio'].min():.2f}x")
    print(f"Average single-core performance gain: {df['single_core_speedup'].mean():.2f}x")
    
    # Check if single-core performance is better
    better_single_core = (df['single_core_speedup'] > 1).sum()
    print(f"Cases where single-core performance is better: {better_single_core}/{len(df)}")
    
    # Prepare LaTeX table
    summary_df = df[['p', 'u_time', 'ig_time', 'speedup_ratio', 'single_core_speedup']].copy()
    summary_df = summary_df.round(3)  # Round for nicer LaTeX output
    
    latex_table = summary_df.to_latex(index=False, caption=f'Performance Comparison for n={n}', label=f'tab:speedup_n{n}')
    
    # Save to .tex file
    with open(f'motif_summary_n{n}.tex', 'w') as f:
        f.write(latex_table)
    
    print("\nLaTeX table saved to:", f'motif_summary_n{n}.tex')
    
    return summary_df


def main(n=10000):
    """
    Main execution function
    
    Args:
        n (int or str): The value of n to filter and analyze
    """
    # Load and process data
    df = load_and_process_data(n)
    
    if df.empty:
        print(f"No data found for n = {n}.")
        return None
    
    # Generate summary statistics
    summary_df = generate_summary_stats(df, n)

    print("\nDetailed Results:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # Create and show plot
    plt_obj = create_speedup_plot(df, n)
    plt_obj.savefig(f'motif_speedup_comparison_n{n}.pdf', dpi=300, bbox_inches='tight')
    plt_obj.savefig(f'motif_speedup_comparison_n{n}.png', dpi=300, bbox_inches='tight')
    plt_obj.show()
    
    return df

if __name__ == "__main__":
    # You can change this value or parse from command line if needed
    results_df = main(n=2000)
