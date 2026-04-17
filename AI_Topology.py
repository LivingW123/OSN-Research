import random
import collections
import numpy as np
try:
    import pygad
except ImportError:
    pygad = None

from Traffic_Benchmarks import (
    calculate_topology_capacity,
    generate_uniform_traffic,
    generate_hotspot_traffic,
    generate_skewed_traffic,
)

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

    GA-Robust definition (traffic_type="robust")
    ─────────────────────────────────────────────
    When traffic_type="robust", candidates are evaluated against the full
    suite  {Uniform, Skewed, Hotspot}  simultaneously.  The fitness is the
    average capacity across all three, penalised by ASPL and degree spread.

    The evolved topology is a STATIC graph (no circuit reconfiguration), so:
        hw_reconfig_ratio = 0.0          — zero reconfiguration overhead
        (cf. Opera 0.04, Sirius 0.0384)

    This is the key reason GA-Robust finds a middle ground:
      • Better than Shale  under uniform/adversarial: shorter ASPL → better
        per-hop efficiency and no 2h-path bandwidth tax from VLB.
      • Better than Sirius under hotspot: topology was evolved to avoid
        single points of VLB convergence; no 2-hop mandate.
      • More flexible than Opera: not constrained by the α/1−α bulk split;
        any src-dst pair can use any available path at any time.
      • Equal or better hw_overhead: 0.0 < Opera 0.04 ≈ Sirius 0.0384.

    Fitness:
        F = 0.1 · avg_capacity(W, {Uniform, Skewed, Hotspot})
          + 10 / ASPL(W)
          − [λ₁ · |avg_degree − D| + λ₂ · Var(degrees)]
    where λ₁=5, λ₂=2, D=target_degree.
    The capacity weight (0.1) is intentionally small so ASPL dominates;
    a topology with short paths serves ALL traffic patterns well.
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


def _avg_fct_from_topology(adj, traffic_matrices, total_power=50.0):
    """
    Compute mean normalised FCT across the robust traffic suite for a given
    topology.  Used by GA-FCT and GA-Dynamic fitness functions.
    """
    total_fct = 0.0
    n_runs = 0
    for tm in traffic_matrices:
        fct, primary, _secondary = calculate_topology_capacity(
            adj, tm,
            total_power=total_power,
            architecture_type=None,      # generic expander model (no arch-specific reconfig)
            return_metrics=True,
        )
        # FCT per active flow — scale by number of non-zero demands
        active = int(np.sum(tm > 0))
        if active > 0 and np.isfinite(fct):
            total_fct += fct / active
            n_runs += 1
    if n_runs == 0:
        return float("inf")
    return total_fct / n_runs


def evolve_topology_fct(num_nodes, target_degree,
                        population_size=15, generations=50,
                        traffic_type="robust",
                        frozen_backbone=None,
                        w_c=0.15, w_l=8.0, w_f=0.5,
                        total_power=50.0):
    """
    GA-FCT: a variant of evolve_topology() that trades throughput for lower FCT.

    Fitness:
        F_FCT = w_c * avg_capacity + w_l / ASPL - w_f * avg_FCT - degree_penalty

    Relative to GA-Robust (evolve_topology):
      • Capacity weight raised (0.10 -> 0.15) to push throughput harder.
      • ASPL weight reduced (10 -> 8) because avg_FCT already captures path-length.
      • New w_f * avg_FCT term suppresses topologies with long FCT tails.

    Returns the adjacency list of the best-evolved topology.
    """
    if pygad is None:
        raise ImportError("pygad not available; install pygad to use evolve_topology_fct")

    print(f"GA-FCT Evolving (N={num_nodes}, D={target_degree}, Traffic={traffic_type}, "
          f"w_c={w_c}, w_l={w_l}, w_f={w_f})...")

    # Traffic mix (same as GA-Robust when traffic_type='robust')
    if traffic_type == "robust":
        traffic_matrices = [
            generate_uniform_traffic(num_nodes),
            generate_skewed_traffic(num_nodes),
            generate_hotspot_traffic(num_nodes, hotspot_nodes=[0, 1]),
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
        aspl = calculate_aspl(adj)
        if aspl == float("inf"):
            return 0.0001

        avg_capacity = float(np.mean([
            calculate_topology_capacity(adj, tm, total_power=total_power)
            for tm in traffic_matrices
        ]))
        avg_fct = _avg_fct_from_topology(adj, traffic_matrices, total_power=total_power)
        if not np.isfinite(avg_fct):
            avg_fct = 1e6

        degrees = [len(n) for n in adj]
        avg_deg = float(np.mean(degrees))
        deg_var = float(np.var(degrees))
        degree_penalty = abs(avg_deg - target_degree) * 5 + deg_var * 2

        fitness = (w_c * avg_capacity
                   + w_l / aspl
                   - w_f * np.log1p(avg_fct)         # log1p to match composite-score FCT compression
                   - degree_penalty)
        return max(float(fitness), 0.0001)

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=int(population_size * 0.4),
        fitness_func=fitness_func,
        sol_per_pop=population_size,
        num_genes=num_genes,
        init_range_low=0, init_range_high=2,
        gene_type=int,
        mutation_percent_genes=10,
        suppress_warnings=True,
    )
    ga_instance.run()

    solution, fitness, _idx = ga_instance.best_solution()
    best_adj = genome_to_adj(solution, num_nodes, frozen_backbone=frozen_backbone)
    print(f"GA-FCT complete. Fitness={fitness:.4f}, ASPL={calculate_aspl(best_adj):.4f}")
    return best_adj


# ──────────────────────────────────────────────────────────────────────────
# GA-DYNAMIC: topology + routing co-evolution
# ──────────────────────────────────────────────────────────────────────────

def _k_shortest_paths(adj, src, dst, K=3, max_len=None):
    """
    Yen-k style k-shortest simple paths between src and dst on an undirected
    adjacency list.  Falls back to BFS single-path if k-shortest produces fewer.
    Returns list of paths (each a list of node ids) sorted by length ascending.
    """
    n = len(adj)
    if src == dst:
        return [[src]]

    # BFS shortest path first
    def bfs_path(block_edges=None, block_nodes=None):
        block_edges = block_edges or set()
        block_nodes = block_nodes or set()
        parent = {src: None}
        q = collections.deque([src])
        while q:
            u = q.popleft()
            if u == dst:
                path = []
                cur = u
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return list(reversed(path))
            for v in adj[u]:
                if v is None or v in parent or v in block_nodes:
                    continue
                if (u, v) in block_edges or (v, u) in block_edges:
                    continue
                parent[v] = u
                q.append(v)
        return None

    paths = []
    seen = set()
    first = bfs_path()
    if first is None:
        return []
    paths.append(first)
    seen.add(tuple(first))

    # Generate alternates by penalising edges on previously-found paths
    for _ in range(K - 1):
        # Remove one edge from each existing path in turn; pick best alternate
        best_alt = None
        best_len = float("inf")
        for p in paths:
            for i in range(len(p) - 1):
                blocked = {(p[i], p[i + 1])}
                alt = bfs_path(block_edges=blocked)
                if alt is not None and tuple(alt) not in seen and len(alt) < best_len:
                    best_alt = alt
                    best_len = len(alt)
        if best_alt is None:
            break
        paths.append(best_alt)
        seen.add(tuple(best_alt))

    paths.sort(key=len)
    if max_len is not None:
        paths = [p for p in paths if len(p) - 1 <= max_len]
    return paths


def _umax_under_routing(adj, traffic, paths_cache, route_weights):
    """
    Given a topology `adj`, a demand matrix `traffic`, a dict
    `paths_cache[(i,j)]` of k candidate paths, and `route_weights[(i,j)]` =
    np.array of non-negative splits summing to 1, compute u_max = max directed
    edge load.
    """
    edge_load = collections.defaultdict(float)
    n = len(adj)
    for i in range(n):
        for j in range(n):
            if i == j or traffic[i, j] <= 0:
                continue
            paths = paths_cache.get((i, j), [])
            if not paths:
                continue
            weights = route_weights.get((i, j))
            if weights is None or len(weights) != len(paths):
                weights = np.full(len(paths), 1.0 / len(paths))
            d = float(traffic[i, j])
            for w, p in zip(weights, paths):
                share = d * w
                for a, b in zip(p[:-1], p[1:]):
                    edge_load[(a, b)] += share
    return max(edge_load.values()) if edge_load else 0.0


def evolve_topology_dynamic(num_nodes, target_degree,
                            population_size=20, generations=100,
                            K_paths=3,
                            traffic_type="robust",
                            frozen_backbone=None,
                            w_c=0.2, w_u=0.1, w_f=0.4,
                            total_power=50.0,
                            seed=None):
    """
    GA-Dynamic: co-evolve topology + per-pair multi-path routing weights.

    Genome layout:
        bits [0 : C(N,2)]                           — upper-triangle topology bits
        bits [C(N,2) : C(N,2) + N*(N-1)*K_paths]    — routing weights (raw floats in [0,1]
                                                       before renormalisation to sum-to-1)

    Fitness:
        F_Dyn = w_c * avg_capacity - w_u * u_max - w_f * log1p(avg_FCT) - degree_penalty

    The routing weights are decoded into per-pair non-negative normalised splits
    over the K shortest paths on the evolved topology.  Splits with sum-to-zero
    fall back to ECMP (equal split across paths).

    Returns:  (best_adj, best_route_weights)
        best_adj: adjacency list
        best_route_weights: dict (i,j) -> np.ndarray of weights (length ≤ K_paths)
    """
    if pygad is None:
        raise ImportError("pygad not available; install pygad to use evolve_topology_dynamic")

    rng = np.random.RandomState(seed) if seed is not None else np.random
    print(f"GA-Dynamic Evolving (N={num_nodes}, D={target_degree}, K={K_paths}, "
          f"pop={population_size}, gen={generations})...")

    if traffic_type == "robust":
        traffic_matrices = [
            generate_uniform_traffic(num_nodes),
            generate_skewed_traffic(num_nodes),
            generate_hotspot_traffic(num_nodes, hotspot_nodes=[0, 1]),
        ]
    elif traffic_type == "uniform":
        traffic_matrices = [generate_uniform_traffic(num_nodes)]
    elif traffic_type == "hotspot":
        traffic_matrices = [generate_hotspot_traffic(num_nodes, hotspot_nodes=[0, 1])]
    elif traffic_type == "skewed":
        traffic_matrices = [generate_skewed_traffic(num_nodes)]
    else:
        traffic_matrices = [generate_uniform_traffic(num_nodes)]

    n_topo_genes = num_nodes * (num_nodes - 1) // 2
    n_pairs = num_nodes * (num_nodes - 1)      # ordered pairs
    n_route_genes = n_pairs * K_paths
    num_genes = n_topo_genes + n_route_genes

    pair_index = {}
    idx = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                pair_index[(i, j)] = idx
                idx += 1

    def decode_routing(weights_flat, adj):
        """Build per-pair k-shortest paths + normalised split weights."""
        paths_cache = {}
        routes = {}
        for (i, j), pidx in pair_index.items():
            paths = _k_shortest_paths(adj, i, j, K=K_paths)
            if not paths:
                continue
            paths_cache[(i, j)] = paths
            raw = weights_flat[pidx * K_paths : pidx * K_paths + len(paths)]
            raw = np.clip(np.asarray(raw, dtype=float), 1e-6, None)
            routes[(i, j)] = raw / raw.sum()
        return paths_cache, routes

    def fitness_func(ga_instance, solution, solution_idx):
        topo_bits = [int(b) for b in solution[:n_topo_genes]]
        route_vec = np.asarray(solution[n_topo_genes:], dtype=float)

        adj = genome_to_adj(topo_bits, num_nodes, frozen_backbone=frozen_backbone)
        aspl = calculate_aspl(adj)
        if aspl == float("inf"):
            return 0.0001

        paths_cache, routes = decode_routing(route_vec, adj)

        # Capacity + FCT under generic expander model (already computed by calculate_topology_capacity)
        caps = []
        fcts = []
        for tm in traffic_matrices:
            c = calculate_topology_capacity(adj, tm, total_power=total_power)
            caps.append(c)
            f, _p, _s = calculate_topology_capacity(
                adj, tm, total_power=total_power,
                architecture_type=None, return_metrics=True,
            )
            if np.isfinite(f):
                fcts.append(f / max(1, int(np.sum(tm > 0))))
        avg_capacity = float(np.mean(caps)) if caps else 0.0
        avg_fct = float(np.mean(fcts)) if fcts else 1e6

        # u_max using evolved routing (averaged over traffic matrices)
        u_values = []
        for tm in traffic_matrices:
            u_values.append(_umax_under_routing(adj, tm, paths_cache, routes))
        u_max = float(np.mean(u_values)) if u_values else 1.0

        degrees = [len(n) for n in adj]
        avg_deg = float(np.mean(degrees))
        deg_var = float(np.var(degrees))
        degree_penalty = abs(avg_deg - target_degree) * 5 + deg_var * 2

        fitness = (w_c * avg_capacity
                   - w_u * u_max
                   - w_f * np.log1p(avg_fct)
                   - degree_penalty)
        return max(float(fitness), 0.0001)

    # Mixed-type genome: first n_topo_genes are binary, rest are floats in [0,1]
    gene_space = [[0, 1]] * n_topo_genes + [{"low": 0.0, "high": 1.0}] * n_route_genes
    init_range_low = [0.0] * num_genes
    init_range_high = [1.0] * num_genes

    ga_instance = pygad.GA(
        num_generations=generations,
        num_parents_mating=int(population_size * 0.4),
        fitness_func=fitness_func,
        sol_per_pop=population_size,
        num_genes=num_genes,
        gene_space=gene_space,
        mutation_percent_genes=10,
        mutation_type="random",
        suppress_warnings=True,
    )
    ga_instance.run()

    solution, fitness, _idx = ga_instance.best_solution()
    topo_bits = [int(b) for b in solution[:n_topo_genes]]
    route_vec = np.asarray(solution[n_topo_genes:], dtype=float)
    best_adj = genome_to_adj(topo_bits, num_nodes, frozen_backbone=frozen_backbone)
    _paths_cache, best_routes = ( {}, {} )
    # Decode once more for return
    best_paths, best_routes = (None, None)
    try:
        best_paths, best_routes = decode_routing(route_vec, best_adj)
    except Exception:
        pass

    print(f"GA-Dynamic complete. Fitness={fitness:.4f}, ASPL={calculate_aspl(best_adj):.4f}")
    return best_adj, best_routes, best_paths
