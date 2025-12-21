import random
import collections
import numpy as np

# Try importing pygad, but don't crash if it fails (though we expect it to be installed)
try:
    import pygad
except ImportError:
    pygad = None

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

def genome_to_adj(genome, num_nodes):
    """
    Converts a flat genome (representing upper triangle of adj matrix) to adjacency list.
    """
    adj = [set() for _ in range(num_nodes)]
    idx = 0
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if genome[idx] == 1:
                adj[i].add(j)
                adj[j].add(i)
            idx += 1
    return [list(x) for x in adj]

def evolve_topology(num_nodes, target_degree, population_size=10, generations=20):
    """
    Evolves a network topology using PyGAD.
    Optimizes for minimizing ASPL with a penalty for deviating from target degree.
    """
    if pygad is None:
        print("Error: PyGAD not installed. Please install it using 'pip install pygad'.")
        return None

    print(f"AI Evolving Topology (PyGAD) (N={num_nodes}, D={target_degree})...")
    
    num_genes = num_nodes * (num_nodes - 1) // 2
    
    def fitness_func(ga_instance, solution, solution_idx):
        adj = genome_to_adj(solution, num_nodes)
        
        # 1. ASPL Score
        aspl = calculate_aspl(adj)
        
        if aspl == float('inf'):
            return 0.0001 # Extremely low fitness for disconnected graphs
            
        # 2. Degree Penalty
        degree_penalty = 0
        for neighbors in adj:
            d = len(neighbors)
            degree_penalty += abs(d - target_degree)
            
        # Fitness formula: roughly 1 / (ASPL + Penalty)
        # We weigh degree penalty to ensure graph regularity is prioritized if desired
        # or balance it.
        # Let's say we want ASPL to be low (e.g. 2.5)
        # If degree is off by 1 for every node (N=20), penalty is 20.
        # We should scale penalty to be comparable or dominant if strict regularity is needed.
        
        combined_score = aspl + (degree_penalty * 0.5) 
        
        if combined_score == 0:
            return 99999
            
        return 1.0 / combined_score

    ga_instance = pygad.GA(num_generations=generations,
                           num_parents_mating=int(population_size * 0.4),
                           fitness_func=fitness_func,
                           sol_per_pop=population_size,
                           num_genes=num_genes,
                           init_range_low=0,
                           init_range_high=2, # Exclusive, so 0 or 1
                           gene_type=int,
                           mutation_percent_genes=5,
                           suppress_warnings=True)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    
    print(f"Evolution Complete. Best Fitness: {solution_fitness:.4f}")
    
    best_adj = genome_to_adj(solution, num_nodes)
    actual_aspl = calculate_aspl(best_adj)
    print(f"Best ASPL: {actual_aspl:.4f}")
    
    return best_adj


