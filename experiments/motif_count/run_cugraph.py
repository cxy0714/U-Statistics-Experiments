import torch
import cugraph
import cudf
import networkx as nx
import numpy as np
import time
import pandas as pd
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt
import seaborn as sns
from u_stats import ustat, set_backend

set_backend("torch")

# CuGraph motif counting
def count_cugraph_triangles(G):
    """
    Count triangles using cugraph
    
    Args:
        G: NetworkX graph
    
    Returns:
        int: triangle count by cugraph
    """
    try:
        edge_list = list(G.edges())
        cg = cugraph.Graph(directed=False)
        df = cudf.DataFrame(edge_list, columns=["src", "dst"])
        cg.from_cudf_edgelist(df, source="src", destination="dst")
        
        triangle_result = cugraph.triangle_count(cg)
        total_triangles = triangle_result['counts'].sum() // 3
        
        return int(total_triangles)
    except Exception as e:
        raise e

def count_U_triangles(G):
    """
    Count triangles using U-statistics
    
    Args:
        G: NetworkX graph
    
    Returns:
        int: triangle count by U-statistics
    """
    isomap = [1, 1, 1]
    divisor = 6
    mode = [['i','j'], ['i','k'], ['j','k']]
    
    A = nx.to_numpy_array(G)
    B = 1 - A
    tensors = [A if m else B for m in isomap]
    count = ustat(tensors, mode, average=False)
    results = int(count // divisor)
    return results

def generate_random_graph(n, p=0.1, seed=None):
    """Generate a random graph with n nodes and edge probability p"""
    G = nx.gnp_random_graph(n, p, seed=seed)
    return G

def run_single_experiment(n, p, seed):
    """
    Run a single experiment comparing both methods
    
    Returns:
        dict: results containing counts and times for both methods
    """
    result = {
        'n': n,
        'p': p,
        'seed': seed,
        'u_time': None,
        'u_count': None,
        'u_error': None,
        'cugraph_time': None,
        'cugraph_count': None,
        'cugraph_error': None,
        'num_edges': None
    }
    
    # Generate graph
    G = generate_random_graph(n, p, seed=seed)
    result['num_edges'] = G.number_of_edges()
    
    # Test U-statistics method
    try:
        start_time = time.time()
        u_count = count_U_triangles(G)
        u_time = time.time() - start_time
        result['u_time'] = u_time
        result['u_count'] = u_count
    except Exception as e:
        result['u_error'] = str(e)
        print(f"  U-statistics failed: {e}")
    
    # Test CuGraph method
    try:
        start_time = time.time()
        cugraph_count = count_cugraph_triangles(G)
        cugraph_time = time.time() - start_time
        result['cugraph_time'] = cugraph_time
        result['cugraph_count'] = cugraph_count
    except Exception as e:
        result['cugraph_error'] = str(e)
        print(f"  CuGraph failed: {e}")
    
    return result

def run_benchmark(n=10000, ps=None, num_trials=10,   output_dir="experiments/motif_count/results"):
    """
    Run benchmark experiments
    
    Args:
        n: number of nodes
        ps: list of edge probabilities
        num_trials: number of trials per configuration
    
    Returns:
        list: all experimental results
    """
    if ps is None:
        ps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.100, 0.150, 0.200, 0.8]
    
    all_results = []
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"GPU_triangle_benchmark_{timestamp}.json"
    output_file = os.path.join(output_dir, filename)

    
    print(f"Starting benchmark: n={n}, trials={num_trials}")
    print(f"Results will be saved to: {output_file}")
    print("=" * 80)
    
    for p in ps:
        print(f"\nTesting p={p}")
        print("-" * 40)
        
        for trial in range(num_trials):
            seed = int(time.time() * 1000) % (2**31) + trial
            print(f"  Trial {trial+1}/{num_trials}, seed={seed}...")
            
            result = run_single_experiment(n, p, seed)
            all_results.append(result)
            
            # Print result summary
            if result['u_time'] and result['cugraph_time']:
                print(f"    U-stats: {result['u_count']} triangles in {result['u_time']:.4f}s")
                print(f"    CuGraph: {result['cugraph_count']} triangles in {result['cugraph_time']:.4f}s")
                print(f"    Match: {result['u_count'] == result['cugraph_count']}")
            
            # Save results incrementally
            with open(output_file, 'w') as f:
                json.dump({
                    'metadata': {
                        'n': n,
                        'ps': ps,
                        'num_trials': num_trials,
                        'timestamp': timestamp
                    },
                    'results': all_results
                }, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Benchmark completed! Results saved to {output_file}")
    return all_results, output_file

    

if __name__ == "__main__":
    # Configuration
    n = 10000
    ps = [0.001, 0.005, 0.01, 0.02, 0.05, 0.08, 0.100, 0.150, 0.200, 0.8]
    num_trials = 10
    
    # Run benchmark
    results, results_file = run_benchmark(n=n, ps=ps, num_trials=num_trials, output_dir="experiments/motif_count/results")