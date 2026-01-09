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
    for t in range(T):
        for u in range(N):
            if link_utilization[t, u] < 1.0:
                remaining_link = 1.0 - link_utilization[t, u]
                
                # Find an intermediate node 'i' that 'u' is currently connected to
                intermediate_i = -1
                for i in range(N):
                    if connectivity_tensor[t, u, i] == 1:
                        intermediate_i = i
                        break
                
                if intermediate_i != -1:
                    # Spray traffic destined for some 'v' through 'i'
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
    connectivity_rr = np.zeros((T, N, N))
    for t in range(T):
        for u in range(N):
            v = (u + t + 1) % N
            connectivity_rr[t, u, v] = 1

    # Generate a Random Schedule (Non-Round-Robin) for comparison
    connectivity_random = np.zeros((T, N, N))
    for t in range(T):
        # Create a random permutation for each timeslot (ensure 1-to-1 mapping if possible, or just random links)
        # To be fair to Sirius constraints (1 trans, 1 recv), use random matching/permutation
        perm = np.random.permutation(N)
        for u in range(N):
            connectivity_random[t, u, perm[u]] = 1
            
    # Define Power Budget (Credit Limit for secondary paths)
    POWER_BUDGET = 0.5 
            
    # Case A: Uniform Demand with Round Robin
    demand_uniform = np.ones((N, N)) * 0.5 
    achieved_rr = sirius_inverse_waterfilling(demand_uniform.copy(), connectivity_rr, credit_limit=POWER_BUDGET)
    
    # Case A2: Uniform Demand with Random Schedule
    achieved_random = sirius_inverse_waterfilling(demand_uniform.copy(), connectivity_random, credit_limit=POWER_BUDGET)
            

    # Visualization of Scheduling Impact
    plt.figure(figsize=(10, 5))
    plt.bar(['Round Robin', 'Random Schedule'], 
             [np.sum(achieved_rr), np.sum(achieved_random)], 
             color=['blue', 'green'])
    plt.ylabel('Total Achieved Throughput')
    plt.title(f'Sirius Schedule Comparison (Power Budget={POWER_BUDGET})')
    plt.savefig('plots/sirius_schedule_comparison.png')
    print("Saved plots/sirius_schedule_comparison.png")


if __name__ == "__main__":
    run_sirius_wf_demo()
