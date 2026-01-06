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

def calculate_topology_capacity(adj_list, traffic_matrix, total_power=50, architecture_type=None):
    """
    Simulates network capacity for a given topology and traffic demand.
    Uses path-length as noise and Waterfilling for power allocation.
    
    Rules implemented:
    - Weight Normalization: For each source node, the sum of path-lengths (noise) 
      to all destinations is normalized to a constant 'Power Level' target.
    - Opera-specific: If architecture_type="opera", the number of timeslots equals the 
      number of hops. This results in the capacity being divided by the path length for each flow.
    """
    from Waterfilling_Alg import waterfilling
    import collections
    
    num_nodes = len(adj_list)
    weight_sum_target = num_nodes * 10.0 # Standard 'Power Level' target for sum of weights per node
    
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
    
    # 2. Map logical traffic flows and Normalize per-node weights
    channels_noise = []
    demands = []
    flow_metas = [] # Store (src, dst, original_hops)
    
    for i in range(num_nodes):
        # Calculate current sum of weights for source i
        current_sum = 0
        for j in range(num_nodes):
            if i == j: continue
            hops = dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
            current_sum += hops
        
        # Scale factor to hit the 'Power Level' target
        scale_factor = weight_sum_target / current_sum if current_sum > 0 else 1.0
        
        for j in range(num_nodes):
            if i == j: continue
            demand = traffic_matrix[i, j]
            if demand > 0:
                original_hops = dist_matrix[i, j] if dist_matrix[i, j] != float('inf') else 100
                normalized_noise = original_hops * scale_factor
                channels_noise.append(normalized_noise)
                demands.append(demand)
                flow_metas.append((i, j, original_hops))
    
    if not channels_noise:
        return 0
        
    # 3. Perform Waterfilling
    allocations = waterfilling(channels_noise, total_power)
    
    # 4. Sum up Weighted Capacity
    capacity = 0
    for idx in range(len(channels_noise)):
        p = allocations[idx]
        n = channels_noise[idx]
        demand = demands[idx]
        src, dst, hops = flow_metas[idx]
        
        if p > 0:
            # Shannon capacity: demand * log2(1 + SNR)
            rate = demand * np.log2(1 + p/n)
            
            if architecture_type == "opera":
                # Opera Hybrid Model (92% Bulk, 8% Short)
                f_bulk = 0.92
                f_short = 0.08
                reconfig_eff = 0.98
                capacity += (f_bulk * rate * reconfig_eff) + (f_short * rate / max(1, hops))
            elif architecture_type == "shale":
                # Shale VLB Model: Every flow is sprayed across h intermediate nodes
                # Effective capacity is divided by (h + 1) hops. 
                # For our base analysis, we'll assume h=2 (2 spraying hops + 1 deliver).
                h_shale = 2
                capacity += rate / (h_shale + 1)
            elif architecture_type == "sirius":
                # Sirius Model: Direct 1-hop vs Spraying 2-hop
                # Efficiency 0.95 covers insertion loss
                eff = 0.95
                if hops <= 1:
                    capacity += rate * eff
                elif hops <= 2:
                    # 2-hop spraying: Bandwidth tax = 2 (consumes two logical slots)
                    capacity += (rate * eff) / 2
                else:
                    # > 2 hops in Sirius is inefficient (not standard VLB)
                    capacity += (rate * eff) / (hops ** 2)
            else:
                # Default (e.g. GA)
                capacity += rate
            
    return capacity
