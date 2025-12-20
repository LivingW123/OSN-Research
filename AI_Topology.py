import random
import collections

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
        
        # Check connectivity from this node
        # If we can't reach all other nodes, graph is disconnected
        # (Though for undirected graphs, checking from one node is enough, 
        # but we need distances for ASPL)
        if len(visited) != n:
             return float('inf')
             
        # Sum distances
        for node in range(n):
            if node != start_node:
                total_distance += distances[node]
                total_paths += 1
                
    if total_paths == 0:
        return 0
        
    # Since graph is undirected, we double counted each path (u->v and v->u)
    # Total distance sum is correct for N*(N-1) paths.
    return total_distance / total_paths

def generate_random_topology(num_nodes, degree):
    """
    Generates a random regular graph with specified degree (approximate).
    Tries to create a k-regular graph by randomly pairing deficient nodes.
    """
    # Simple strategy:
    # 1. Create N nodes with empty adjacency lists
    # 2. List all "sockets" (stubs) available = N * degree
    # 3. Randomly shuffle and pair them up
    
    if (num_nodes * degree) % 2 != 0:
        raise ValueError("N * Degree must be even.")
        
    adj_matrix = [set() for _ in range(num_nodes)] # Use sets to avoid dupes easily
    
    stubs = []
    for i in range(num_nodes):
        stubs.extend([i] * degree)
        
    random.shuffle(stubs)
    
    while len(stubs) >= 2:
        u = stubs.pop()
        v = stubs.pop()
        
        if u == v or v in adj_matrix[u]:
            # Self-loop or duplicate edge.
            # Put back and reshuffle/retry (simple heuristic)
            # Or just ignore this edge (which reduces degree).
            # For simplicity in this heuristic, we'll try to reconnect.
            # But deep retries can be infinite loop. 
            # We'll just ignore for now -> graph might not be perfectly regular.
            continue
        
        adj_matrix[u].add(v)
        adj_matrix[v].add(u)
        
    # Convert sets to lists
    return [list(s) for s in adj_matrix]

def mutate_topology(adj_matrix):
    """
    Mutates the topology by rewiring edges.
    Swaps two edges: (u, v) and (x, y) become (u, x) and (v, y).
    """
    n = len(adj_matrix)
    # Deep copy
    new_adj = [set(row) for row in adj_matrix]
    
    # Pick two distinct edges
    edges = []
    for u in range(n):
        for v in new_adj[u]:
            if u < v:
                edges.append((u, v))
                
    if len(edges) < 2:
        return [list(row) for row in new_adj]
        
    # Try multiple times to find valid swap
    for _ in range(5):
        edge1, edge2 = random.sample(edges, 2)
        u, v = edge1
        x, y = edge2
        
        # Ensure nodes are distinct
        if len({u, v, x, y}) != 4:
            continue
            
        # Check if new edges already exist
        if x in new_adj[u] or y in new_adj[v]:
            continue
            
        # Swap
        new_adj[u].remove(v)
        new_adj[v].remove(u)
        new_adj[x].remove(y)
        new_adj[y].remove(x)
        
        new_adj[u].add(x)
        new_adj[x].add(u)
        new_adj[v].add(y)
        new_adj[y].add(v)
        
        break
        
    return [list(row) for row in new_adj]

def evolve_topology(num_nodes, target_degree, population_size=10, generations=10):
    """
    Evolves a network topology using a Genetic Algorithm.
    Optimizes for minimizing Average Shortest Path Length (ASPL).
    """
    print(f"AI Evolving Topology (N={num_nodes}, D={target_degree})...")
    
    # Initialize Population
    population = []
    for _ in range(population_size):
        topo = generate_random_topology(num_nodes, target_degree)
        score = calculate_aspl(topo)
        population.append((score, topo))
        
    best_overall_score = float('inf')
    best_overall_topo = None
    
    for gen in range(generations):
        # Sort by score (lower ASPL is better)
        population.sort(key=lambda x: x[0])
        
        current_best = population[0]
        if current_best[0] < best_overall_score:
            best_overall_score = current_best[0]
            best_overall_topo = current_best[1]
            
        # print(f"  Gen {gen}: Best ASPL = {current_best[0]:.4f}")
        
        # Selection: Keep top 20%
        cutoff = max(1, int(population_size * 0.2))
        survivors = population[:cutoff]
        
        # Reproduction / Mutation
        next_gen = survivors[:] # Elitism
        
        tries = 0
        while len(next_gen) < population_size and tries < 100:
            parent = random.choice(survivors)[1]
            child_topo = mutate_topology(parent)
            child_score = calculate_aspl(child_topo)
            
            # Simple check to avoid catastrophic mutations (disconnected graphs)
            # If disconnected, maybe don't add? Or add with high penalty (already inf)
            next_gen.append((child_score, child_topo))
            tries += 1
            
        population = next_gen
        
    print(f"Evolution Complete. Best ASPL: {best_overall_score:.4f}")
    return best_overall_topo

