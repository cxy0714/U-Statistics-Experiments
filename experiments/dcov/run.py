from kernel import  get_true_data_matlab
from u_stats import UStats, set_backend
import numpy as np
import json
import time
import os
import argparse
from datetime import datetime

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration
COEFF = [1, 1, -1, -1]
EXPRESSIONS = ["ij,qr->", "ij,ij->", "ij,iq->", "ij,jr->"]
US = [UStats(expression) for expression in EXPRESSIONS]

def kernel(S: np.ndarray) -> np.ndarray:
    """
    Computes the kernel value for a given matrix S.

    Parameters:
    S (np.ndarray): sector data matrix, shape (T, N).
                    N is the number of stocks and T is the number of time points.


    Returns:
    A (np.ndarray): kernel value, shape (T, T).
    """
    G = S @ S.T
    d = np.diag(G)
    A = d[:, None] + d[None, :] - 2 * G
    A = np.maximum(A, 0)
    return np.sqrt(A)


def dcov(A, B):
    """
    Computes the distance covariance between two kernel matrices.
    
    Parameters:
    A (np.ndarray): First kernel matrix
    B (np.ndarray): Second kernel matrix
    
    Returns:
    float: The computed distance covariance
    """
    tensors = [A, B]
    result = 0.0
    
    for coeff, u in zip(COEFF, US):
        res = u.compute(tensors)
        result += coeff * res
    
    return result


def compute_dcov_matrix(kernels):
    """
    Compute the distance covariance matrix for all pairs of kernels.
    
    Parameters:
    kernels (list): List of kernel matrices
    
    Returns:
    np.ndarray: Upper triangular distance covariance matrix
    """
    n_sockets = len(kernels)
    result = np.zeros((n_sockets, n_sockets), dtype=np.float32)
    
    for i in range(n_sockets):
        for j in range(i, n_sockets):
            result[i, j] = dcov(kernels[i], kernels[j])
    
    return result


def run_experiment(n_runs=10, backend="torch", output_file=None):
    """
    Run the distance covariance experiment multiple times and save results.
    
    Parameters:
    n_runs (int): Number of times to run the experiment
    backend (str): Backend to use for computation
    output_file (str): Output JSON file name
    """
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join("experiments", "dcov", "results", f"dcov_results_{backend}_{timestamp}.json")
    
    set_backend(backend)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load data once
    print("Loading data...")
    socket_names, datas = get_true_data_matlab()
    
    results = {
        "experiment_info": {
            "backend": backend,
            "n_runs": n_runs,
            "n_sockets": len(socket_names),
            "socket_names": socket_names,
            "timestamp": datetime.now().isoformat()
        },
        "runs": []
    }
    
    print(f"Running {n_runs} experiments with backend: {backend}")
    
    for run_idx in range(n_runs):
        print(f"Run {run_idx + 1}/{n_runs}...")
        
        # Time kernel computation
        kernel_start_time = time.time()
        kernels = [kernel(data) for data in datas]
        kernel_end_time = time.time()
        kernel_time = kernel_end_time - kernel_start_time
        
        # Time dcov matrix computation
        dcov_start_time = time.time()
        result_matrix = compute_dcov_matrix(kernels)
        dcov_end_time = time.time()
        dcov_time = dcov_end_time - dcov_start_time
        
        # Total time
        total_time = kernel_time + dcov_time
        
        # Store results for this run
        run_result = {
            "run_id": run_idx + 1,
            "kernel_time": kernel_time,
            "computation_time": dcov_time,
            "total_time": total_time,
            "result_matrix": result_matrix.tolist()  # Convert to list for JSON serialization
        }
        
        results["runs"].append(run_result)
        print(f"  Kernel time: {kernel_time:.4f} seconds")
        print(f"  Dcov time: {dcov_time:.4f} seconds")
        print(f"  Total time: {total_time:.4f} seconds")
    
    # Save results to JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary statistics
    kernel_times = [run["kernel_time"] for run in results["runs"]]
    dcov_times = [run["computation_time"] for run in results["runs"]]
    total_times = [run["total_time"] for run in results["runs"]]
    
    print(f"\nSummary:")
    print(f"Kernel computation:")
    print(f"  Mean: {np.mean(kernel_times):.4f} seconds")
    print(f"  Std: {np.std(kernel_times):.4f} seconds")
    print(f"Dcov computation:")
    print(f"  Mean: {np.mean(dcov_times):.4f} seconds")
    print(f"  Std: {np.std(dcov_times):.4f} seconds")
    print(f"Total time:")
    print(f"  Mean: {np.mean(total_times):.4f} seconds")
    print(f"  Std: {np.std(total_times):.4f} seconds")
    print(f"Results saved to: {output_file}")


def load_and_analyze_results(json_file=None):
    """
    Load and analyze results from JSON file.
    
    Parameters:
    json_file (str): Path to the JSON results file
    """
    if json_file is None:
        json_file = os.path.join("experiments", "dcov", "results", "dcov_results.json")
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    print(f"Experiment Info:")
    print(f"  Backend: {results['experiment_info']['backend']}")
    print(f"  Number of runs: {results['experiment_info']['n_runs']}")
    print(f"  Number of sockets: {results['experiment_info']['n_sockets']}")
    print(f"  Timestamp: {results['experiment_info']['timestamp']}")
    
    kernel_times = [run["kernel_time"] for run in results["runs"]]
    dcov_times = [run["computation_time"] for run in results["runs"]]
    total_times = [run["total_time"] for run in results["runs"]]
    
    print(f"\nTiming Statistics:")
    print(f"Kernel computation:")
    print(f"  Mean: {np.mean(kernel_times):.4f} seconds")
    print(f"  Std: {np.std(kernel_times):.4f} seconds")
    print(f"  Min: {np.min(kernel_times):.4f} seconds")
    print(f"  Max: {np.max(kernel_times):.4f} seconds")
    
    print(f"Dcov computation:")
    print(f"  Mean: {np.mean(dcov_times):.4f} seconds")
    print(f"  Std: {np.std(dcov_times):.4f} seconds")
    print(f"  Min: {np.min(dcov_times):.4f} seconds")
    print(f"  Max: {np.max(dcov_times):.4f} seconds")
    
    print(f"Total time:")
    print(f"  Mean: {np.mean(total_times):.4f} seconds")
    print(f"  Std: {np.std(total_times):.4f} seconds")
    print(f"  Min: {np.min(total_times):.4f} seconds")
    print(f"  Max: {np.max(total_times):.4f} seconds")
    
    # Check consistency of results
    first_result = np.array(results["runs"][0]["result_matrix"])
    all_same = all(np.allclose(first_result, np.array(run["result_matrix"])) 
                   for run in results["runs"])
    print(f"  Results consistent across runs: {all_same}")


def main():
    parser = argparse.ArgumentParser(description="Distance Covariance Experiment")
    parser.add_argument("--runs", type=int, default=10, help="Number of runs")
    parser.add_argument("--backend", type=str, default="numpy", help="Backend to use")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: experiments/dcov/results/dcov_results_{backend}_{timestamp}.json)")
    parser.add_argument("--analyze", type=str, help="Analyze existing results file")
    
    args = parser.parse_args()
    
    if args.analyze:
        load_and_analyze_results(args.analyze)
    else:
        run_experiment(args.runs, args.backend, args.output)


if __name__ == "__main__":
    main()