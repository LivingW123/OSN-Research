import numpy as np
import matplotlib.pyplot as plt
import os

def sirius_inverse_waterfilling(demand_matrix, connectivity_tensor, credit_limit=0.5):
    """
    Implements the Sirius 'Inverse Waterfilling' logic.
    Instead of arbitrary allocation, it fills based on a fixed schedule.
    
    Args:
        demand_matrix: N x N traffic demand
        connectivity_tensor: T x N x N (1 if u connected to v at time t)
        credit_limit: Max bandwidth allowed for spraying via intermediate
        
    Returns:
        throughput_matrix: N x N achieved throughput
    """
    T, N, _ = connectivity_tensor.shape
    achieved = np.zeros((N, N))
    link_utilization = np.zeros((T, N)) # Each node has 1 uplink port per slot (simplified)
    
    # 1. Primary Fill: Direct Paths
    for t in range(T):
        for u in range(N):
            for v in range(N):
                if connectivity_tensor[t, u, v] == 1:
                    # u can send directly to v
                    sendable = min(demand_matrix[u, v], 1.0) # Assume 1.0 unit per slot
                    achieved[u, v] += sendable
                    demand_matrix[u, v] -= sendable
                    link_utilization[t, u] = sendable
                    
    # 2. Secondary Fill: Spraying (Indirect Paths)
    # Goal: Use idle slots (valleys) to send to intermediate nodes
    for t in range(T):
        for u in range(N):
            if link_utilization[t, u] < 1.0:
                remaining_link = 1.0 - link_utilization[t, u]
                
                # Find an intermediate node 'i' that 'u' is currently connected to
                # In Sirius, u is connected to some 'i' at time t
                intermediate_i = -1
                for i in range(N):
                    if connectivity_tensor[t, u, i] == 1:
                        intermediate_i = i
                        break
                
                if intermediate_i != -1:
                    # Spray traffic destined for some 'v' through 'i'
                    # Constraint: Credit Limit C
                    for v in range(N):
                        if demand_matrix[u, v] > 0 and v != intermediate_i:
                            spray_amt = min(demand_matrix[u, v], remaining_link, credit_limit)
                            achieved[u, v] += spray_amt
                            demand_matrix[u, v] -= spray_amt
                            link_utilization[t, u] += spray_amt
                            if link_utilization[t, u] >= 1.0: break
                            
    return achieved

def run_sirius_wf_demo():
    N = 8
    T = 8
    
    # Generate a simple round-robin schedule
    connectivity = np.zeros((T, N, N))
    for t in range(T):
        for u in range(N):
            v = (u + t + 1) % N
            connectivity[t, u, v] = 1
            
    # Case A: Uniform Demand
    demand_uniform = np.ones((N, N)) * 0.5 # Each pair wants 0.5 units
    achieved_uniform = sirius_inverse_waterfilling(demand_uniform.copy(), connectivity)
    
    # Case B: Skewed Demand
    demand_skewed = np.zeros((N, N))
    demand_skewed[0, 1] = 5.0 # High demand between node 0 and 1
    achieved_skewed = sirius_inverse_waterfilling(demand_skewed.copy(), connectivity)
    
    # Visualization of 'Valleys' being filled
    plt.figure(figsize=(10, 5))
    plt.bar(['Uniform (Total)', 'Skewed (Flow 0->1)'], 
             [np.sum(achieved_uniform), achieved_skewed[0, 1]], 
             color=['blue', 'orange'])
    plt.ylabel('Achieved Throughput')
    plt.title('Sirius Inverse Waterfilling Performance')
    plt.savefig('plots/sirius_inverse_wf.png')
    print("Saved plots/sirius_inverse_wf.png")

if __name__ == "__main__":
    run_sirius_wf_demo()
