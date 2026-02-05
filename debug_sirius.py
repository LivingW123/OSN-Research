
import numpy as np
import sys
from Benchmarking_Metrics import calculate_sirius_capacity, SiriusConfig, _get_all_pairs_dist
from Sirius import generate_full_system

def debug():
    N = 16
    W = 4
    P = 4
    
    # 1. Generate Topology
    print(f"Generating Sirius (N={N}, W={W}, P={P})...")
    As, Ws, P_mat = generate_full_system(W, P, N)
    sirius_adj = [[(v - 1) % N for v in row] for row in As[0]]
    
    print("Adjacency List (First 3 nodes):")
    degrees = []
    for i in range(N):
        # Unique neighbors only
        neighbors = set(sirius_adj[i])
        degrees.append(len(neighbors))
        if i < 3:
            print(f"Node {i}: {list(neighbors)}")
            
    print(f"\nAverage Degree: {np.mean(degrees)}")
    
    # 2. Check Distances
    dist_matrix = _get_all_pairs_dist(sirius_adj, N)
    print("\nDistance Matrix (First 3x3):")
    print(dist_matrix[:3, :3])
    
    reachable_pairs = np.sum(dist_matrix != float('inf')) - N # Exclude diagonal
    total_pairs = N * (N - 1)
    print(f"Reachable Pairs: {reachable_pairs} / {total_pairs} ({reachable_pairs/total_pairs*100:.1f}%)")
    
    avg_hops = np.mean(dist_matrix[dist_matrix != float('inf')])
    print(f"Average Path Length: {avg_hops}")
    
    # 3. Run Calculation with Debug Prints
    print("\nRunning Calculation...")
    traffic = np.ones((N, N))
    np.fill_diagonal(traffic, 0)
    
    config = SiriusConfig()
    
    # Manually reproduce logic to check values
    eff = (config.slot_duration_ns - config.reconfig_time_ns) / config.slot_duration_ns
    print(f"Efficiency (eta): {eff}")
    
    weight_sum_target = N * 10.0
    print(f"Weight Sum Target: {weight_sum_target}")
    
    i = 0
    current_sum = sum(dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
                        for j in range(N) if i != j)
    print(f"Node 0 Current Sum: {current_sum}")
    scale_factor = weight_sum_target / current_sum if current_sum > 0 else 1.0
    print(f"Node 0 Scale Factor: {scale_factor}")
    
    # Run actual function
    fct, metrics = calculate_sirius_capacity(sirius_adj, traffic, config, total_power=50.0)
    
    print("\nResults:")
    print(f"Normalized Throughput: {metrics.throughput_normalized}")
    print(f"Mean Latency: {metrics.latency_mean}")
    print(f"Avg Hops: {metrics.average_hop_count}")

if __name__ == "__main__":
    debug()
