import numpy as np
import matplotlib.pyplot as plt
import os
import random
from Shale_Alg import RR1, RR2 
from Common_Alg import generate_random_latin_square
from Sirius import generate_full_system
from AI_Topology import evolve_topology, generate_random_topology
from Traffic_Benchmarks import (
    calculate_topology_capacity, 
    generate_uniform_traffic, 
    generate_skewed_traffic, 
    generate_hotspot_traffic
)

def run_comprehensive_benchmark():
    print("--- Running Comprehensive Waterfilling Traffic Benchmark ---")
    N = 9
    D = 4
    powers = np.linspace(20, 100, 5)
    
    # 1. Topologies
    # Opera: Staggered Latin Square
    opera_adj = [[(v - 1) % N for v in row[:D]] for row in generate_random_latin_square(N)]
    
    # Shale: RR2 (Multi-dimensional Torus) - For N=9 (Base 3, Dim 2)
    # Using a 9-node RR2 as a representative for Shale
    shale_node_count = 9
    shale_adj = RR2(3, 2)
    
    # Sirius: AWGR Schedule with P=4 ports
    # Using generate_full_system(w=2, ports=4, N=10)
    sirius_adj = [[(v - 1) % N for v in row] for row in generate_full_system(2, 4, N)[0][0]]
    
    # GA: Evolved for traffic (using robust mode for generalization)
    ga_adj = evolve_topology(N, D, generations=50, traffic_type="robust")
    
    archs = {
        "Opera": opera_adj,
        "Shale (RR2)": shale_adj,
        "Sirius": sirius_adj,
        "GA-Robust": ga_adj
    }
    
    # 2. Traffic Benchmarks
    traffics = {
        "Uniform": generate_uniform_traffic(N),
        "Skewed": generate_skewed_traffic(N),
        "Hotspot": generate_hotspot_traffic(N, hotspot_nodes=[0, 1])
    }
    
    if not os.path.exists('plots'): os.makedirs('plots')
    
    plt.figure(figsize=(15, 6))
    
    for i, (traffic_name, traffic_matrix) in enumerate(traffics.items(), 1):
        plt.subplot(1, 3, i)
        for name, adj in archs.items():
            caps = []
            arch_type = name.lower() if name in ["Opera", "Shale (RR2)", "Sirius"] else None
            if "Shale" in name: arch_type = "shale"
            if "GA" in name: arch_type = None
            
            for p in powers:
                cap = calculate_topology_capacity(adj, traffic_matrix, total_power=p, architecture_type=arch_type)
                caps.append(cap)
            
            plt.plot(powers, caps, marker='o', label=name)
        
        plt.title(f"{traffic_name} Traffic")
        plt.xlabel("Power Budget (P)")
        plt.ylabel("Shannon Capacity")
        if i == 1: plt.legend()
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    plt.savefig("plots/traffic_benchmark_capacity.png")
    print("Saved plots/traffic_benchmark_capacity.png")

if __name__ == "__main__":
    # Mocking evolve_topology locally if import fails or is slow
    try:
        run_comprehensive_benchmark()
    except Exception as e:
        print(f"Benchmark failed: {e}")
        # Fallback to simple random if needed
