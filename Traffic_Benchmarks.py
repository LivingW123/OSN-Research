import numpy as np

def generate_uniform_traffic(num_nodes):
    """All pairs communicate equally."""
    traffic = np.ones((num_nodes, num_nodes))
    np.fill_diagonal(traffic, 0)
    return traffic

def generate_hotspot_traffic(num_nodes, hotspot_nodes=[0], intensity=10):
    """Certain nodes receive/send much more traffic than others."""
    traffic = np.ones((num_nodes, num_nodes))
    for node in hotspot_nodes:
        traffic[:, node] *= intensity
        traffic[node, :] *= intensity
    np.fill_diagonal(traffic, 0)
    return traffic

def generate_skewed_traffic(num_nodes, skew_factor=2):
    """Traffic follows a power-law or zipf-like distribution."""
    traffic = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            # Simplified skew: closer indices have more traffic
            traffic[i, j] = 1.0 / (abs(i - j) ** skew_factor + 1)
    return traffic

def calculate_topology_capacity(adj_list, traffic_matrix, total_power=50):
    """
    Simulates network capacity for a given topology and traffic demand.
    Uses path-length as noise and Waterfilling for power allocation.
    """
    from Waterfilling_Alg import waterfilling
    import collections
    
    num_nodes = len(adj_list)
    
    # 1. Find all-pairs shortest paths to determine 'noise' (hops = noise)
    def get_all_pairs_dist(adj):
        dist_matrix = np.full((num_nodes, num_nodes), float('inf'))
        for start in range(num_nodes):
            dist_matrix[start, start] = 0
            queue = collections.deque([(start, 0)])
            visited = {start}
            while queue:
                u, d = queue.popleft()
                dist_matrix[start, u] = d
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        queue.append((v, d + 1))
        return dist_matrix

    dist_matrix = get_all_pairs_dist(adj_list)
    
    # 2. Map logical traffic flows to physical characteristics
    # Logical channels are SRC-DST pairs with traffic > 0
    channels_noise = []
    demands = []
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i == j: continue
            demand = traffic_matrix[i, j]
            if demand > 0:
                # Noise is path length. If disconnected, noise is huge.
                noise = dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
                channels_noise.append(noise)
                demands.append(demand)
    
    if not channels_noise:
        return 0
        
    # 3. Perform Waterfilling
    # Note: Traditional waterfilling assumes equal weight. 
    # Here we can weight by demand or adjust noise. 
    # Let's simplify: Capacity = sum(log2(1 + P_i / Noise_i)) weighted by demand.
    
    allocations = waterfilling(channels_noise, total_power)
    
    capacity = 0
    for i in range(len(channels_noise)):
        p = allocations[i]
        n = channels_noise[i]
        if p > 0:
            # Shannon capacity: W * log2(1 + SNR)
            capacity += demands[i] * np.log2(1 + p/n)
            
    return capacity
