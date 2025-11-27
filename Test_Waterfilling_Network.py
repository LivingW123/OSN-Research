import numpy as np
import random
import matplotlib.pyplot as plt
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

def visualize_waterfilling(channels, total_power):
    """
    Visualizes the waterfilling power allocation using a stacked bar chart.
    """
    allocation = waterfilling(channels, total_power)
    channels = np.array(channels) # Ensure numpy array for indexing
    n = len(channels)
    indices = np.arange(n)
    
    # Calculate water level for plotting line
    # The water level is constant for active channels: noise + power
    # For inactive channels, it's just the noise level (which is above the water level)
    # We can find the effective water level by looking at an active channel
    # If no power is allocated, the water level is effectively 0 or below the min noise
    
    # A robust way to find the water level from the output for plotting:
    # It is the value (noise + power) for any channel with power > 0.
    # If all powers are 0, water level is effectively 0 (or undefined/below min noise).
    
    active_mask = allocation > 1e-9 # use small epsilon for float comparison
    if np.any(active_mask):
        water_level = channels[active_mask][0] + allocation[active_mask][0]
    else:
        water_level = 0 # Or max(channels) if we want to show it's too low? 
                        # But strictly, if P_tot > 0, there must be some allocation unless P_tot is tiny?
                        # Actually if P_tot > 0, at least one channel gets power.
                        # If P_tot = 0, water_level is 0.
        pass

    plt.figure(figsize=(10, 6))
    
    # Plot Noise Levels
    plt.bar(indices, channels, label='Noise Level', color='lightgray', edgecolor='black')
    
    # Plot Allocated Power on top of Noise
    plt.bar(indices, allocation, bottom=channels, label='Allocated Power', color='skyblue', edgecolor='black')
    
    # Plot Water Level Line
    if np.any(active_mask):
        plt.axhline(y=water_level, color='red', linestyle='--', label=f'Water Level ({water_level:.2f})')
    
    plt.xlabel('Channel Index')
    plt.ylabel('Power / Noise Level')
    plt.title(f'Waterfilling Algorithm Results (Total Power: {total_power})')
    plt.xticks(indices, [f'Ch {i}' for i in indices])
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.show()

if __name__ == "__main__":
    test_opera_waterfilling()
    test_shale_waterfilling()
    
    # Visualization Example
    print("\n=== Running Visualization ===")
    example_channels = [10, 20, 30, 40, 15, 25, 35, 45]
    example_power = 60
    visualize_waterfilling(example_channels, example_power)
