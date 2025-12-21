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
    
    # channels for each timeslot
    channels = []
    
    random.seed(100)
    
    for t, W in enumerate(Ws):
        # Find active links
        active_links = []
        rows, cols = np.where(np.array(W) == 1)
        
        # N links
        if len(rows) != nodes:
            print(f"WARNING: Timeslot {t} has {len(rows)} links, expected {nodes}")
            
        # Assign random noise
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
    
    #RR2 graph
    base = 3
    dim = 2
    adj_matrix = RR2(base, dim)
    
    print(f"Generated RR2({base}, {dim}) Adjacency Matrix.")
    
    # weights
    weights = {i: 1 for i in range(base**dim)}
    weights[4] = 100 # High cost node
    
    print("Weights defined. Node 4 is expensive (100). Others are 1.")
    
    start_node = 0
    end_node = 8
    
    print(f"Finding top 10 paths from Node {start_node} to Node {end_node}...")
    
    # RR2_path
    paths_dict = RR2_path(adj_matrix, weights, start_node, end_node)
    
    # Results
    print(f"\nFound {len(paths_dict)} paths (capped at 10).")
    
    # at most 10 paths
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
        # Verify cost
        calc_cost = sum(weights[n] for n in path)
        if calc_cost != cost:
            print(f"    ERROR: Calculated cost {calc_cost} != Returned cost {cost}")
        # Verify order
        if prev_cost != -1 and cost < prev_cost:
            print("    ERROR: Paths are not sorted by cost!")
        prev_cost = cost
        
        if 4 in path and cost < 100:
            print("    WARNING: Path contains high cost node 4 but cost is low? (Should be > 100)")

    if not paths_dict:
        print("WARNING: No paths found!")

def test_spray_short():
    print("\n=== Testing Spray-Short Algorithm ===")
    base = 3
    dim = 2
    adj_matrix = RR2(base, dim)
    
    weights = {i: 1 for i in range(base**dim)}
    
    start_node = 0 # (0,0)
    end_node = 8   # (2,2)
    
    print(f"Graph: RR2({base}, {dim})")
    print(f"Spray Short from {start_node} to {end_node}, K=3")
    
    from Opera_Alg import (
    find_optimal_path_broken_racks,
    find_least_cost_path_weighted_racks,
    check_guard_band
)
    from Shale_Alg import spray_short
    paths = spray_short(adj_matrix, weights, start_node, end_node, k=3, penalty_factor=2.0)
    
    print("\nFound Paths (Path, Original Cost):")
    for i, (p, c) in enumerate(paths):
        print(f"  Path {i}: {p}, Cost: {c}")
        
    #valid path
    if not paths:
        print("ERROR: No paths found!")
        return
    if paths[0][1] > paths[1][1] and paths[0][1] > paths[2][1]: # Just a loose check, usually first should be shortest
         pass 

    #Diversity:
    path0 = paths[0][0]
    path1 = paths[1][0] if len(paths) > 1 else None
    
    if path1:
        # Check intersection of intermediate nodes
        inter0 = set(path0[1:-1])
        inter1 = set(path1[1:-1])
        common = inter0.intersection(inter1)
        
        print(f"\nIntermediate Nodes Path 0: {inter0}")
        print(f"Intermediate Nodes Path 1: {inter1}")
        print(f"Common Intermediate Nodes: {common}")
        
        if not common:
            print("SUCCESS: Path 1 deviated from Path 0 (no common intermediate nodes).")
        else:
            print("NOTE: Paths share some nodes.")

def test_ai_topology():
    print("\n=== Testing AI Topology Generation ===")
    from AI_Topology import evolve_topology, calculate_aspl, generate_random_topology
    
    N = 20
    Degree = 3
    
    #baseline random topology
    baseline_topo = generate_random_topology(N, Degree)
    baseline_aspl = calculate_aspl(baseline_topo)
    print(f"Baseline Random Topology ASPL: {baseline_aspl:.4f}")
    
    #AI Evo
    evolved_topo = evolve_topology(N, Degree, population_size=10, generations=100)
    evolved_aspl = calculate_aspl(evolved_topo)
    
    #verify
    print(f"Evolved Topology ASPL: {evolved_aspl:.4f}")
    
    if evolved_topo is None:
        print("ERROR: Evolution failed to produce a topology.")
        return
        
    if evolved_aspl <= baseline_aspl:
        print("SUCCESS: AI generated a topology with equal or better efficiency (lower/equal ASPL).")
    else:
        print("NOTE: AI ASPL is higher. This can happen with small generation counts or random chance.")
        
    if evolved_aspl != float('inf'):
        print("SUCCESS: Evolved topology is connected.")
        
        # Visualize
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            G = nx.Graph()
            for i, neighbors in enumerate(evolved_topo):
                for n in neighbors:
                    G.add_edge(i, n)
            
            plt.figure(figsize=(8, 8))
            pos = nx.circular_layout(G)
            nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_weight='bold')
            plt.title(f"Evolved Topology (N={N}, D={Degree})\nASPL: {evolved_aspl:.4f}")
            plt.show()
            
        except ImportError:
            print("Visualization skipped (networkx or matplotlib not found).")
            
    else:
        print("ERROR: Evolved topology is disconnected!")

def test_guard_band():
    from Opera_Alg import check_guard_band
    print("\n=== Testing Guard Band Check ===")
    
    slot_duration = 10.0
    guard_band = 1.0
    
    # Synced (Arrivals within [0, 9.0])
    arrivals_1 = [0.0, 5.0, 9.0, 10.0, 18.0, 20.0]
    
    synced, violations = check_guard_band(arrivals_1, slot_duration, guard_band)
    if synced:
        print("Case 1 (Synced): SUCCESS")
    else:
        print(f"Case 1 (Synced): FAILED. Violations: {violations}")
        
    #Desynced (Arrivals > 9.0 and < 10.0 mod 10)
    arrivals_2 = [9.5, 19.1, 29.9]
    synced, violations = check_guard_band(arrivals_2, slot_duration, guard_band)
    if not synced and len(violations) == 3:
        print(f"Case 2 (Desynced): SUCCESS. Caught violations: {violations}")
    else:
        print(f"Case 2 (Desynced): FAILED. Expected 3 violations, got synced={synced}, violations={violations}")

def compare_waterfilling_performance():
    print("\n=== Comparing Waterfilling Performance (Capacity vs Power) ===")
    opera_noise = [5, 10, 5, 20, 100, 5, 10, 5]
    shale_noise = [4, 1, 9, 8]
    sirius_noise = [5, 15, 15, 6, 13, 12]
    generic_noise = [5, 5, 5, 5]
    
    power_levels = np.linspace(10, 100, 10)
    
    results = {
        "Opera": [],
        "Shale": [],
        "Sirius": [],
        "Generic": []
    }
    
    scenarios = [
        ("Opera", opera_noise),
        ("Shale", shale_noise),
        ("Sirius", sirius_noise),
        ("Generic", generic_noise)
    ]
    
    for label, noises in scenarios:
        for P in power_levels:
            alloc = waterfilling(noises, P)
            
            # Calculate Capacity: sum(log2(1 + Pi/Ni))
            alloc_arr = np.array(alloc)
            noise_arr = np.array(noises)
            
            # Handle potential zeros in noise if any (though ours are >0)
            with np.errstate(divide='ignore'):
                snr = alloc_arr / noise_arr
                
            capacity = np.sum(np.log2(1 + snr))
            results[label].append(capacity)
            
    # Visualize
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        for label, capacities in results.items():
            plt.plot(power_levels, capacities, marker='o', label=label)
            
        plt.xlabel("Total Power Budget (P)")
        plt.ylabel("Capacity (Shannon Sum Rate)")
        plt.title("Waterfilling Performance Comparison: All 4 Contexts")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        print("Displaying comparison plot...")
        plt.show()
        
    except ImportError:
        print("Comparison visualization skipped (matplotlib not found).")

if __name__ == "__main__":
    test_opera_waterfilling()
    test_shale_waterfilling()
    test_waterfilling_timeslots()
    test_sirius_waterfilling()
    test_rr2_path()
    test_spray_short()
    test_guard_band()
    test_ai_topology()
    compare_waterfilling_performance()
