import numpy as np
import random
import matplotlib.pyplot as plt
from Waterfilling_Alg import waterfilling
from Shale_Alg import RR2, RR2_path
from Common_Alg import generate_random_latin_square
from Sirius import generate_full_system

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
def test_waterfilling_timeslots():
    print("\n=== Testing Waterfilling with Multiple Timeslots ===")
    
    # 3 Timeslots, 4 Channels
    # Timeslot 0: Low noise
    # Timeslot 1: High noise
    # Timeslot 2: Mixed noise
    channels = [
        [5, 5, 5, 5],
        [20, 20, 20, 20],
        [5, 20, 5, 20]
    ]
    
    total_power = 20 # Power per timeslot
    
    print(f"Channels (3 Timeslots x 4 Channels):\n{np.array(channels)}")
    print(f"Total Power per Timeslot: {total_power}")
    
    allocations = waterfilling(channels, total_power)
    
    print("\nPower Allocation (Timeslots x Channels):")
    print(allocations)
    
    # Verification
    for t in range(3):
        total_p = np.sum(allocations[t])
        print(f"Timeslot {t} Total Power: {total_p:.2f}")
        if abs(total_p - total_power) > 1e-5:
             print(f"WARNING: Timeslot {t} power mismatch!")
        else:
             print(f"SUCCESS: Timeslot {t} power budget met.")

def test_sirius_waterfilling():
    print("\n=== Testing Waterfilling with Sirius Context ===")
    
    # Parameters for Sirius
    wavelengths = 3
    ports = 2
    nodes = 6
    
    print(f"Generating Sirius Topology (W={wavelengths}, P={ports}, N={nodes})...")
    As, Ws, P = generate_full_system(wavelengths, ports, nodes)
    
    num_timeslots = len(Ws)
    print(f"Number of Timeslots: {num_timeslots}")
    
    # Construct channels for each timeslot
    # In Sirius, W matrices are permutation matrices (size N x N)
    # W[i][j] = 1 means there is a link from node i to node j
    # Since it's a permutation, each node has exactly 1 outgoing link per timeslot (in W)
    # So we have N links per timeslot.
    
    channels = []
    
    random.seed(100)
    
    for t, W in enumerate(Ws):
        # Find active links
        active_links = []
        rows, cols = np.where(np.array(W) == 1)
        
        # Just to verify we have N links
        if len(rows) != nodes:
            print(f"WARNING: Timeslot {t} has {len(rows)} links, expected {nodes}")
            
        # Assign random noise to each active link
        # We'll represent the channel state as just the noise level for now
        # In a real scenario, we might map (src, dst) to a noise value
        timeslot_noise = [random.randint(1, 20) for _ in range(nodes)]
        channels.append(timeslot_noise)
        
        print(f"Timeslot {t} Active Links (Src->Dst): {list(zip(rows, cols))}")
        print(f"  Noise Levels: {timeslot_noise}")
        
    total_power = 30
    print(f"Total Power per Timeslot: {total_power}")
    
    allocations = waterfilling(channels, total_power)
    
    print("\nPower Allocation (Timeslots x Links):")
    print(allocations)
    
    # Verification
    for t in range(num_timeslots):
        total_p = np.sum(allocations[t])
        print(f"Timeslot {t} Total Power: {total_p:.2f}")
        if abs(total_p - total_power) > 1e-5:
             print(f"WARNING: Timeslot {t} power mismatch!")
        else:
             print(f"SUCCESS: Timeslot {t} power budget met.")

def visualize_waterfilling(channels, total_power):
    """
    Visualizes the waterfilling power allocation using a stacked bar chart.
    """
    allocation = waterfilling(channels, total_power)
    channels = np.array(channels) # Ensure numpy array for indexing
    n = len(channels)
    indices = np.arange(n)
    
    active_mask = allocation > 1e-9 # use small epsilon for float comparison
    if np.any(active_mask):
        water_level = channels[active_mask][0] + allocation[active_mask][0]
    else:
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

def test_rr2_path():
    print("\n=== Testing RR2 Path Finding with Cost Limit ===")
    
    # 1. Generate RR2(3, 2) graph
    # This creates a graph with 3^2 = 9 nodes (0 to 8).
    # Each node represents a tuple, e.g., 0 -> (0,0), 1 -> (0,1), 2 -> (0,2)...
    base = 3
    dim = 2
    adj_matrix = RR2(base, dim)
    
    print(f"Generated RR2({base}, {dim}) Adjacency Matrix.")
    
    # 2. Define Weights
    # Let's assign weights such that some paths are clearly cheaper.
    # Node weights: 0..8
    # We'll make node 4 expensive to see if paths avoid it or it increases cost.
    weights = {i: 1 for i in range(base**dim)}
    weights[4] = 100 # High cost node
    
    print("Weights defined. Node 4 is expensive (100). Others are 1.")
    
    start_node = 0
    end_node = 8
    
    print(f"Finding top 10 paths from Node {start_node} to Node {end_node}...")
    
    # 3. Call RR2_path
    paths_dict = RR2_path(adj_matrix, weights, start_node, end_node)
    
    # 4. Verify Results
    print(f"\nFound {len(paths_dict)} paths (capped at 10).")
    
    # Check if we got at most 10 paths
    if len(paths_dict) > 10:
        print("ERROR: More than 10 paths returned!")
    else:
        print("SUCCESS: Returned path count is within limit.")
        
    print("\nTop Paths and Costs:")
    sorted_paths = sorted(paths_dict.items(), key=lambda item: item[1])
    
    prev_cost = -1
    for path, cost in sorted_paths:
        print(f"  Cost {cost}: {path}")
        
        # Verify valid path
        is_valid = True
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            if v not in adj_matrix[u]:
                print(f"    ERROR: Invalid link {u} -> {v}")
                is_valid = False
        
        # Verify cost calculation
        # Cost = sum of weights of all nodes in path
        calc_cost = sum(weights[n] for n in path)
        if calc_cost != cost:
             print(f"    ERROR: Calculated cost {calc_cost} != Returned cost {cost}")
        
        # Verify sorted order
        if prev_cost != -1 and cost < prev_cost:
            print("    ERROR: Paths are not sorted by cost!")
        prev_cost = cost
        
        if 4 in path and cost < 100:
             print("    WARNING: Path contains high cost node 4 but cost is low? (Should be > 100)")

    if not paths_dict:
        print("WARNING: No paths found!")

if __name__ == "__main__":
    test_opera_waterfilling()
    test_shale_waterfilling()
    test_waterfilling_timeslots()
    test_sirius_waterfilling()
    test_rr2_path()
    
    # Visualization Example
    # print("\n=== Running Visualization ===")
    # example_channels = [10, 20, 30, 40, 15, 25, 35, 45]
    # example_power = 60
    # visualize_waterfilling(example_channels, example_power)
