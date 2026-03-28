import torch
import igraph as ig
from itertools import permutations, combinations
import networkx as nx
import numpy as np
import time
import pandas as pd
import json
from datetime import datetime
import traceback
import gc
import os

from U_stats import ustat
from U_stats._utils._backend import set_backend
set_backend("torch")  

# Constants (your existing constants)
ISOMAP_3 = {
    0: [0,0,0],  
    1: [1,0,0],   
    2: [1,0,1],   
    3: [1,1,1],   
}
ISOMAP_3_DIVISORS = {
    0: None,  # nan
    1: None,  # nan
    2: 2,
    3: 6,
}

ISOMAP_4 = {
    0: [0,0,0,0,0,0],
    1: [1,0,0,0,0,0],
    2: [1,0,0,0,1,0],
    3: [1,1,0,0,0,0],
    4: [1,1,1,0,0,0],
    5: [0,0,1,1,0,0],
    6: [1,0,1,1,0,0],
    7: [1,1,1,1,0,0],
    8: [1,0,1,1,0,1],
    9: [1,1,0,1,1,1],
    10:[1,1,1,1,1,1],
}
ISOMAP_DIVISORS_4 = {
    0: None,   # NaN
    1: None,   # NaN
    2: None,   # NaN
    3: None,   # NaN
    4: 6,
    5: None,   # NaN
    6: 2,
    7: 2,
    8: 8,
    9: 4,
    10: 24
}

def count_all_motifs_by_U(G, size, use_einsum= True):
    """Count all motifs for a given size using U-statistics"""
    if size == 3:
        isomap = ISOMAP_3
        divisors = ISOMAP_3_DIVISORS
        mode = [['i','j'], ['i','k'], ['j','k']]
    elif size == 4:
        isomap = ISOMAP_4
        divisors = ISOMAP_DIVISORS_4
        mode = [['i','j'], ['i','k'], ['i','l'], ['j','k'], ['j','l'], ['k','l']]
    else:
        raise ValueError("Size must be 3 or 4")
    
    A = nx.to_numpy_array(G)
    B = 1 - A
    results = {}
    
    for motif_id, mask in isomap.items():
        divisor = divisors[motif_id]
        if divisor is None:
            results[motif_id] = np.nan
            continue
        
        tensors = [A if m else B for m in mask]
        count = ustat(
            tensors, 
            mode, 
            average=False, 
            path_method="double-greedy-fill-minus-degree", 
            dediag=True, 
            use_einsum=use_einsum
        )
        results[motif_id] = int(count // divisor)
    
    return results

def count_igraph_motifs(G, size):
    """Count motifs using igraph for comparison"""
    g = ig.Graph.from_networkx(G)
    motifs = g.motifs_randesu(size=size)
    return motifs

def generate_random_graph(n, p=0.1, seed=None):
    """Generate a random graph with n nodes and edge probability p"""
    G = nx.gnp_random_graph(n, p, seed=seed)
    return G

def warmup_torch():
    """Warmup torch operations"""
    print("Warming up torch...")
    G_small = generate_random_graph(50, p=0.1, seed=42)
    try:
        count_all_motifs_by_U(G_small, size=3)
        count_all_motifs_by_U(G_small, size=4)
    except:
        pass
    print("Warmup completed.")

def time_function_with_retries(func, *args, max_retries=3, **kwargs):
    """Time function execution with retries for robustness"""
    times = []
    results = []
    
    for attempt in range(max_retries):
        try:
            gc.collect()  # Clean up memory before each attempt
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            execution_time = end_time - start_time
            
            times.append(execution_time)
            results.append(result)
            
        except Exception as e:
            print(f"    Attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return None, None, str(e)
            continue
    
    # Return the median time and corresponding result
    median_idx = len(times) // 2
    sorted_indices = sorted(range(len(times)), key=lambda i: times[i])
    median_time = times[sorted_indices[median_idx]]
    median_result = results[sorted_indices[median_idx]]
    
    return median_result, median_time, None

def convert_igraph_to_dict(ig_result):
    """Convert igraph list format to dict format"""
    if isinstance(ig_result, list):
        return {i: (ig_result[i] if i < len(ig_result) else None) for i in range(len(ig_result))}
    return ig_result

def test_grid_performance(size, n_list, p_list, output_dir="experiments/motif_count/results"):
    """
    Test performance across grid of n and p values with real-time saving
    
    Args:
        size: motif size (3 or 4)
        n_list: list of graph sizes to test
        p_list: list of edge probabilities to test  
        output_dir: directory to save results
    
    Returns:
        results: list of result dictionaries
    """
    print(f"Testing size-{size} motifs with grid search...")
    print(f"n values: {n_list}")
    print(f"p values: {p_list}")
    print(f"Total combinations: {len(n_list) * len(p_list)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"motif_size_{size}_results_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    results = []
    total_tests = len(n_list) * len(p_list)
    test_count = 0
    
    print(f"Results will be saved to: {filepath}")
    
    for n in n_list:
        for p in p_list:
            test_count += 1
            print(f"\nTest {test_count}/{total_tests}: n={n}, p={p}")
            
            # Generate graph
            G = generate_random_graph(n, p=p, seed=42)
            edge_count = G.number_of_edges()
            actual_density = edge_count / (n * (n-1) / 2) if n > 1 else 0
            
            print(f"  Graph: {n} nodes, {edge_count} edges, density={actual_density:.4f}")
            
            # Test U-statistics method
            print("  Testing U-statistics...")
            u_result, u_time, u_error = time_function_with_retries(
                count_all_motifs_by_U, G, size, use_einsum= True
            )
            
            # Test igraph method
            print("  Testing igraph...")
            # ig_result, ig_time, ig_error = time_function_with_retries(
            #     count_igraph_motifs, G, size
            # )
            ig_result, ig_time, ig_error = None, None, None
            
            # Store results
            result_entry = {
                'size': size,
                'n': n,
                'p': p,
                'edge_count': edge_count,
                'actual_density': actual_density,
                'u_time': u_time,
                'u_error': u_error,
                'u_result': u_result,  # Keep your dict format
                'ig_time': ig_time,
                'ig_error': ig_error,
                'ig_result': convert_igraph_to_dict(ig_result),  # Convert to dict format
                'timestamp': datetime.now().isoformat()
            }
            results.append(result_entry)
            
            # Save results immediately after each test
            try:
                with open(filepath, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"  ✓ Saved to {filename} ({len(results)} results)")
            except Exception as e:
                print(f"  ✗ Save failed: {e}")
            
            # Print timing results
            if u_time is not None and ig_time is not None:
                speedup = ig_time / u_time
                print(f"  Results: U={u_time:.4f}s, igraph={ig_time:.4f}s, speedup={speedup:.2f}x")
            else:
                print(f"  Results: U={u_time}, igraph={ig_time}")
                if u_error:
                    print(f"    U-stats error: {u_error}")
                if ig_error:
                    print(f"    igraph error: {ig_error}")
    
    return results

def save_summary(results, size, output_dir="experiments/motif_count/results"):
    """Save summary log at the end"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"motif_size_{size}_summary_{timestamp}.txt"
    log_filepath = os.path.join(output_dir, log_filename)
    
    with open(log_filepath, 'w') as f:
        f.write(f"Motif Size {size} Performance Test Summary\n")
        f.write(f"Timestamp: {datetime.now()}\n")
        f.write(f"Total tests: {len(results)}\n\n")
        
        successful_tests = [r for r in results if r['u_time'] is not None and r['ig_time'] is not None]
        failed_tests = [r for r in results if r['u_time'] is None or r['ig_time'] is None]
        
        f.write(f"Successful tests: {len(successful_tests)}\n")
        f.write(f"Failed tests: {len(failed_tests)}\n\n")
        
        if successful_tests:
            u_times = [r['u_time'] for r in successful_tests]
            ig_times = [r['ig_time'] for r in successful_tests]
            speedups = [ig_t / u_t for u_t, ig_t in zip(u_times, ig_times)]
            
            f.write("Performance Summary:\n")
            f.write(f"  U-stats time - mean: {np.mean(u_times):.4f}s, std: {np.std(u_times):.4f}s\n")
            f.write(f"  igraph time - mean: {np.mean(ig_times):.4f}s, std: {np.std(ig_times):.4f}s\n")
            f.write(f"  Speedup - mean: {np.mean(speedups):.2f}x, std: {np.std(speedups):.2f}x\n")
            f.write(f"  Speedup - min: {np.min(speedups):.2f}x, max: {np.max(speedups):.2f}x\n\n")
        
        if failed_tests:
            f.write("Failed Test Details:\n")
            for i, test in enumerate(failed_tests):
                f.write(f"  Test {i+1}: n={test['n']}, p={test['p']}\n")
                if test['u_error']:
                    f.write(f"    U-stats error: {test['u_error']}\n")
                if test['ig_error']:
                    f.write(f"    igraph error: {test['ig_error']}\n")
    
    print(f"\nSummary saved to {log_filepath}")
    return log_filepath

def main():
    """Example usage"""
    print("Motif Counting Grid Performance Test")
    print("="*50)
    
    # Warmup
    warmup_torch()
    
    # Example test configurations
    # You can modify these when calling the function
    
    # For size 3 - larger n range  
    n_list_3 = np.logspace(np.log10(1000), np.log10(10000), 5).astype(int)
    n_list_3 = sorted(list(set(n_list_3)))
    
    # For size 4 - smaller n range due to memory constraints
    n_list_4 = np.logspace(np.log10(1000), np.log10(4000), 5).astype(int)  
    n_list_4 = sorted(list(set(n_list_4)))
    
    # Non-linear p values
    p_list = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8]
    
    print("Example configurations:")
    print(f"Size 3 - n values: {n_list_3}")
    print(f"Size 4 - n values: {n_list_4}")
    print(f"p values: {p_list}")
    
    # Uncomment to run tests:
    test_size_3(n_list_3, p_list)
    test_size_4(n_list_4, p_list)

def test_size_3(n_list, p_list):
    """Test size-3 motifs with real-time saving"""
    results = test_grid_performance(3, n_list, p_list)
    save_summary(results, 3)
    return results

def test_size_4(n_list, p_list):  
    """Test size-4 motifs with real-time saving"""
    results = test_grid_performance(4, n_list, p_list)
    save_summary(results, 4)
    return results

if __name__ == "__main__":
    main()