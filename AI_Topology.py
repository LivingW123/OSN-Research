import random
import collections
import numpy as np
try:
    import pygad
except ImportError:
    pygad = None

from Traffic_Benchmarks import calculate_topology_capacity, generate_uniform_traffic, generate_hotspot_traffic, generate_skewed_traffic

def calculate_aspl(adj_matrix):
    """
    Calculates the Average Shortest Path Length (ASPL) of a graph using BFS.
    Returns float('inf') if graph is disconnected.
    """
    n = len(adj_matrix)
    total_distance = 0
    total_paths = 0
    
    # Run BFS from each node to find all pairs shortest paths
    for start_node in range(n):
        visited = {start_node}
        queue = collections.deque([(start_node, 0)])
        distances = {}
        
        while queue:
            curr, dist = queue.popleft()
            distances[curr] = dist
            
            neighbors = adj_matrix[curr]
            if neighbors:
                for neighbor in neighbors:
                    if neighbor is not None and neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))
        
        if len(visited) != n:
             return float('inf')
             
        for node in range(n):
            if node != start_node:
                total_distance += distances[node]
                total_paths += 1
                
    if total_paths == 0:
        return 0
        
    return total_distance / total_paths

def generate_random_topology(num_nodes, degree):
    """
    Generates a random regular graph with specified degree (approximate).
    """
    if (num_nodes * degree) % 2 != 0:
        raise ValueError("N * Degree must be even.")
        
    adj_matrix = [set() for _ in range(num_nodes)] 
    
    stubs = []
    for i in range(num_nodes):
        stubs.extend([i] * degree)
        
    random.shuffle(stubs)
    
    while len(stubs) >= 2:
        u = stubs.pop()
        v = stubs.pop()
        
        if u == v or v in adj_matrix[u]:
            continue
        
        adj_matrix[u].add(v)
        adj_matrix[v].add(u)
        
    return [list(s) for s in adj_matrix]

def genome_to_adj(genome, num_nodes, frozen_backbone=None):
    """
    Converts a flat genome (representing upper triangle of adj matrix) to adjacency list.
    If frozen_backbone is provided, those edges are always included.
    """
    adj = [set() for _ in range(num_nodes)]
    
    # Add frozen edges if they exist
    if frozen_backbone:
        for i, neighbors in enumerate(frozen_backbone):
            for n in neighbors:
                adj[i].add(n)
                adj[n].add(i)

    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if genome[idx] == 1:
                adj[i].add(j)
                adj[j].add(i)
            idx += 1
    return [list(x) for x in adj]

def evolve_topology(num_nodes, target_degree, population_size=15, generations=50, 
                    traffic_type="uniform", frozen_backbone=None):
    """
    Evolves a network topology using PyGAD.
    Optimizes for a mix of minimizing ASPL and maximizing delivered capacity.
    """

    print(f"AI Evolving Topology (PyGAD) (N={num_nodes}, D={target_degree}, Traffic={traffic_type})...")
    
    # Generate Traffic Matrices
    traffic_matrices = []
    if traffic_type == "robust":
        traffic_matrices = [
            generate_uniform_traffic(num_nodes),
            generate_skewed_traffic(num_nodes),
            generate_hotspot_traffic(num_nodes, hotspot_nodes=[0, 1])
        ]
    elif traffic_type == "uniform":
        traffic_matrices = [generate_uniform_traffic(num_nodes)]
    elif traffic_type == "hotspot":
        traffic_matrices = [generate_hotspot_traffic(num_nodes, hotspot_nodes=[0, 1])]
    elif traffic_type == "skewed":
        traffic_matrices = [generate_skewed_traffic(num_nodes)]
    else:
        traffic_matrices = [generate_uniform_traffic(num_nodes)]

    num_genes = num_nodes * (num_nodes - 1) // 2
    
    def fitness_func(ga_instance, solution, solution_idx):
        adj = genome_to_adj(solution, num_nodes, frozen_backbone=frozen_backbone)
        
        # 1. Connectivity Check
        aspl = calculate_aspl(adj)
        if aspl == float('inf'):
            return 0.0001
            
        # 2. Capacity Score (Average over all target traffic matrices)
        avg_capacity = np.mean([calculate_topology_capacity(adj, tm) for tm in traffic_matrices])
        
        # 3. Degree Distribution Penalty
        degrees = [len(neighbors) for neighbors in adj]
        # Stronger penalty for exceeding max degree or having too few links
        avg_degree = np.mean(degrees)
        degree_var = np.var(degrees)
        degree_penalty = abs(avg_degree - target_degree) * 5 + degree_var * 2
            
        # Composite Fitness: Capacity is usually 50-200. ASPL is 1.5-3.0.
        fitness = (avg_capacity * 0.1) + (10.0 / aspl) - degree_penalty
        
        return max(fitness, 0.0001)

    ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=int(population_size * 0.4),
                           fitness_func=fitness_func,
                           sol_per_pop=population_size,
                           num_genes=num_genes,
                           init_range_low=0,
                           init_range_high=2, 
                           gene_type=int,
                           mutation_percent_genes=10, # Slightly higher mutation to explore
                           suppress_warnings=True)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    print(f"Evolution Complete. Best Fitness: {solution_fitness:.4f}")
    
    best_adj = genome_to_adj(solution, num_nodes, frozen_backbone=frozen_backbone)
    actual_aspl = calculate_aspl(best_adj)
    final_cap = np.mean([calculate_topology_capacity(best_adj, tm) for tm in traffic_matrices])
    print(f"Best ASPL: {actual_aspl:.4f} | Final Avg Capacity: {final_cap:.4f}")
    
    return best_adj


