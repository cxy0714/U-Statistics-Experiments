import torch
import igraph as ig
from itertools import permutations, combinations
import networkx as nx
import numpy as np
import time
import pandas as pd

from U_stats import ustat
from U_stats._utils._backend import set_backend
set_backend("torch")  

# Constants
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

def count_motif_by_U(G, size, motif_id, use_einsum="torch"):
    """
    Count specific motif by size and motif_id using U-statistics
    
    Args:
        G: NetworkX graph
        size: motif size (3 or 4)
        motif_id: specific motif id to count
        use_einsum: einsum backend
    
    Returns:
        int: count of the specific motif
    """
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
    
    if motif_id not in isomap:
        raise ValueError(f"Invalid motif_id {motif_id} for size {size}")
    
    divisor = divisors[motif_id]
    if divisor is None:
        return np.nan
    
    mask = isomap[motif_id]
    A = nx.to_numpy_array(G)
    B = 1 - A
    tensors = [A if m else B for m in mask]
    
    count = ustat(
        tensors, 
        mode, 
        average=False, 
        path_method="double-greedy-fill-minus-degree", 
        dediag=True, 
        use_einsum=use_einsum
    )
    return int(count // divisor)

def count_all_motifs_by_U(G, size, use_einsum="torch"):
    """
    Count all motifs for a given size using U-statistics
    
    Args:
        G: NetworkX graph
        size: motif size (3 or 4)
        use_einsum: einsum backend
    
    Returns:
        dict: dictionary with motif_id as key and count as value
    """
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

# Utility functions for testing and comparison
def generate_random_graph(n, p=0.1, seed=None):
    """Generate a random graph with n nodes and edge probability p"""
    G = nx.gnp_random_graph(n, p, seed=seed)
    return G

def time_function(func, *args, **kwargs):
    """Time the execution of a function"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Comparison functions with igraph
def count_igraph_motifs(G, size):
    """Count motifs using igraph for comparison"""
    g = ig.Graph.from_networkx(G)
    motifs = g.motifs_randesu(size=size)
    return motifs
  
if __name__ == "__main__":
      # Count specific motif
    G = generate_random_graph(n = 50, p=0.1, seed=42)  # Example graph

    # Count all motifs
    all_4_motifs = count_all_motifs_by_U(G, size=4)
    all_3_motifs = count_all_motifs_by_U(G, size=3)
    print("All 4-motifs counts:", all_4_motifs)
    print("All 3-motifs counts:", all_3_motifs) 
    
    all_4_motifs_igraph = count_igraph_motifs(G, size=4)
    all_3_motifs_igraph = count_igraph_motifs(G, size=3)
    print("All 4-motifs counts (igraph):", all_4_motifs_igraph)
    print("All 3-motifs counts (igraph):", all_3_motifs_igraph)
        
