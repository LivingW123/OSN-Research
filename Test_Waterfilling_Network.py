import numpy as np
import random
from Waterfilling_Alg import waterfilling
from Shale_Alg import RR2
from Common_Alg import generate_random_latin_square

def test_opera_waterfilling():
    print("\n=== Testing Waterfilling with Opera Context ===")
    
    rack_weights = [5, 10, 5, 20, 100, 5, 10, 5]
    total_power = 50
    
    print(f"Rack Weights (Noise): {rack_weights}")
    print(f"Total Power Budget: {total_power}")
    
    allocation = waterfilling(rack_weights, total_power)
    
    print("\nPower Allocation per Rack:")
    for i, p in enumerate(allocation):
        print(f"  Rack {i} (Weight {rack_weights[i]}): {p:.2f}")
        
    print(f"\nTotal Allocated Power: {np.sum(allocation):.2f}")
    
    # Verify: Racks with weight 100 should likely get 0 power
    if allocation[4] == 0:
        print("SUCCESS: High cost rack received 0 power.")
    else:
        print("NOTE: High cost rack received power (budget was large enough).")

def test_shale_waterfilling():
    print("\n=== Testing Waterfilling with Shale Context ===")
    print("Generating Shale Topology (RR2, Base 3, Dim 2)...")
    adj_matrix = RR2(3, 2)
    
    node_index = 0
    node_0_links = adj_matrix[node_index]
    active_links = [x for x in node_0_links if x is not None]
    
    num_links = len(active_links)
    print(f"Node {node_index} has {num_links} active links: {active_links}")
    
    random.seed(42)
    link_noise = [random.randint(1, 20) for _ in range(num_links)]
    total_power = 30
    
    print(f"Simulated Link Noise Levels: {link_noise}")
    print(f"Total Power Budget: {total_power}")
    
    allocation = waterfilling(link_noise, total_power)
    
    print("\nPower Allocation per Link:")
    for i, neighbor in enumerate(active_links):
        p = allocation[i]
        n = link_noise[i]
        print(f"  Link to Node {neighbor} (Noise {n}): {p:.2f}")
        
    print(f"\nTotal Allocated Power: {np.sum(allocation):.2f}")

if __name__ == "__main__":
    test_opera_waterfilling()
    test_shale_waterfilling()
