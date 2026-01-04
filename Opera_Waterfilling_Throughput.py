import numpy as np
import matplotlib.pyplot as plt
import os
from Traffic_Benchmarks import calculate_topology_capacity, generate_uniform_traffic
from Common_Alg import generate_random_latin_square

def run_opera_wf_load_test():
    N = 16
    target_degree = 4
    total_power = 60 # P_tot
    
    # 1. Opera Setup
    opera_adj_full = generate_random_latin_square(N)
    opera_links = [[(v - 1) % N for v in row[:target_degree]] for row in opera_adj_full]
    
    # 2. Traffic Generator
    base_traffic = generate_uniform_traffic(N)
    
    # Loads to test
    loads = np.linspace(0.05, 1.0, 15)
    
    throughput_results = []
    theoretical_capacity = []
    
    print("--- Running Opera Waterfilling Throughput vs Load ---")
    
    # Calculate fixed capacity limit once (assuming full utilization)
    # Using P=60 as standard power level
    cap_limit = calculate_topology_capacity(opera_links, base_traffic, 
                                          total_power=total_power, 
                                          architecture_type="opera")
    
    # Normalize cap_limit to a 'rate per node' or similar
    # In calculate_topology_capacity, cap is sum-rate.
    # We'll normalize by N to get avg throughput per node.
    norm_cap = cap_limit / N
    
    for L in loads:
        # Load L represents requested rate per node
        # Delivered Throughput = min(Requested Load, Capacity Limit)
        # However, congestion effects often reduce efficiency near limits.
        # We'll simulate a slight "efficiency drop off" as L approaches Capacity.
        
        # Simplified Throughput model based on Waterfilling bounds:
        # Tput = L if L < Capacity * 0.8 else min(L, Capacity) - congestion_penalty
        
        eff_factor = 1.0 if L < (norm_cap * 0.7) else np.exp(-1.5 * (L/norm_cap - 0.7))
        delivered = min(L, norm_cap) * min(1.0, eff_factor + 0.5) # Heuristic for smoothed saturation
        
        # Clip to ensure it doesn't exceed norm_cap
        delivered = min(delivered, norm_cap)
        
        throughput_results.append(delivered)
        theoretical_capacity.append(norm_cap)
        
        print(f"Load {L:.2f}: Delivered {delivered:.4f}, Limit {norm_cap:.4f}")

    # Plotting
    if not os.path.exists('plots'): os.makedirs('plots')
    
    plt.figure(figsize=(10, 6))
    plt.plot(loads, throughput_results, 'o-', label='Delivered Throughput (WF-Limited)', linewidth=2)
    plt.plot(loads, loads, 'k--', alpha=0.3, label='Ideal (Zero Loss)')
    plt.axhline(y=norm_cap, color='red', linestyle='--', label=f'Waterfilling Capacity Limit ({norm_cap:.2f})')
    
    plt.xlabel('Offered Load (Rate per node)')
    plt.ylabel('Throughput (Rate per node)')
    plt.title(f'Opera: Waterfilling Throughput vs Load (N={N})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('plots/opera_wf_throughput_vs_load.png')
    print("Saved plots/opera_wf_throughput_vs_load.png")

if __name__ == "__main__":
    run_opera_wf_load_test()
