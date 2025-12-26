import numpy as np
import random
import matplotlib.pyplot as plt
import os
import collections

# Mocking the folder structure and imports for testing
from Waterfilling_Alg import waterfilling
from Shale_Alg import RR2, RR2_path, spray_short
from Common_Alg import generate_random_latin_square
from Sirius import generate_full_system
from AI_Topology import evolve_topology, generate_random_topology, calculate_aspl
from Opera_Alg import find_optimal_path_broken_racks, find_path_2d
from Traffic_Benchmarks import calculate_topology_capacity, generate_uniform_traffic, generate_skewed_traffic, generate_hotspot_traffic

if not os.path.exists('plots'):
    os.makedirs('plots')

def run_rigorous_waterfilling(base_noise, total_power, iterations=10, noise_std=1.0):
    base_noise = np.array(base_noise)
    all_allocations = []
    for _ in range(iterations):
        stochastic_noise = np.maximum(base_noise + np.random.normal(0, noise_std, base_noise.shape), 0.1)
        alloc = waterfilling(stochastic_noise, total_power)
        all_allocations.append(alloc)
    return np.mean(all_allocations, axis=0), np.std(all_allocations, axis=0)

def visualize_waterfilling(channels, total_power, title="Waterfilling Results", filename=None, yerr=None):
    channels = np.array(channels)
    if channels.ndim == 2:
        num_timeslots = channels.shape[0]
        powers = [total_power] * num_timeslots if np.isscalar(total_power) else total_power
        for t in range(num_timeslots):
            ts_suffix = f"_ts{t}"
            ts_filename = f"plots/{filename}{ts_suffix}.png" if filename else None
            ts_yerr = yerr[t] if yerr is not None else None
            visualize_waterfilling(channels[t], powers[t], title=f"{title} - TS{t}", filename=ts_filename, yerr=ts_yerr)
        return
    allocation = waterfilling(channels, total_power)
    n = len(channels)
    indices = np.arange(n)
    active_mask = allocation > 1e-9
    bottleneck_mask = ~active_mask
    water_level = np.mean(channels[active_mask] + allocation[active_mask]) if np.any(active_mask) else 0
    # plt.figure(figsize=(10, 6)) # Skip plotting to avoid GUI issues in command line
    print(f"DEBUG: visualize_waterfilling called for {title}")

# 1. Opera
rack_weights = [5, 10, 5, 20, 100, 5, 10, 5]
total_power = 50
avg_p, std_p = run_rigorous_waterfilling(rack_weights, total_power)
visualize_waterfilling(rack_weights, total_power, title="Opera Rigorous Allocation", filename="opera_rigorous", yerr=std_p)

# 2. Shale
num_nodes, degree = 10, 4
shale_links = generate_random_topology(num_nodes, degree)
link_noise = [random.randint(5, 25) for _ in range(degree)]
avg_p, std_p = run_rigorous_waterfilling(link_noise, 50)
visualize_waterfilling(link_noise, 50, title="Shale Node Neighborhood", filename="shale_rigorous", yerr=std_p)

# 3. Sirius
As, Ws, P_mat = generate_full_system(wavelengths=2, ports=2, nodes=6)
channels = [[random.randint(5, 20) for _ in range(6)] for _ in Ws]
sirius_stds = [run_rigorous_waterfilling(ch, 50)[1] for ch in channels]
visualize_waterfilling(channels, 50, title="Sirius Multi-Slot Rigorous", filename="sirius_rigorous", yerr=np.array(sirius_stds))

# 4. Hybrid GA
num_nodes = 10
target_degree = 4
ring_backbone = [[] for _ in range(num_nodes)]
for i in range(num_nodes):
    ring_backbone[i].extend([(i + 1) % num_nodes, (i - 1) % num_nodes])
best_adj = evolve_topology(num_nodes, target_degree, generations=2, traffic_type="skewed", frozen_backbone=ring_backbone)
traffic = generate_skewed_traffic(num_nodes)
final_cap = calculate_topology_capacity(best_adj, traffic, total_power=50)

# 5. Grand Architecture Comparison
power_levels = np.linspace(10, 100, 3) # Fewer levels for speed
opera_raw = generate_random_latin_square(num_nodes)
opera_adj = [[(v - 1) % num_nodes for v in row[:target_degree]] for row in opera_raw]
shale_adj = generate_random_topology(num_nodes, target_degree)
As, _, _ = generate_full_system(2, 2, num_nodes)
sirius_adj = [[(v - 1) % num_nodes for v in row] for row in As[0]]
architectures = {
    "Opera (Static)": opera_adj,
    "Shale (Regular)": shale_adj,
    "Sirius (Slotted)": sirius_adj,
    "Hybrid GA (Evolved)": best_adj
}
results = {name: [] for name in architectures}
for name, adj in architectures.items():
    for P in power_levels:
        results[name].append(calculate_topology_capacity(adj, traffic, total_power=P))
print("DEBUG: Comparison complete")
