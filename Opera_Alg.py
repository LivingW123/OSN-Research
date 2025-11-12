import collections
import heapq  # We need heapq for the priority queue, not collections.deque
import numpy as np # Using numpy for the latin square generator

from Common_Alg import(
    generate_random_latin_square
)

def find_optimal_path_broken_racks(A, broken_racks, start_tor, end_tor):
    """
    Finds the shortest path (least hops) given a set of broken *racks* (columns).
    
    Args:
        A (list[list[int]]): 
            The N x M matrix where A[i][j] = k means node 'i'
            connects to node 'k' (1-indexed) via rack 'j'.
            
        broken_racks (set[int] or list[int]): 
            A set or list of 0-indexed *rack indices* (columns) that are broken.
            
        start_tor (int): The 0-indexed starting ToR.
        end_tor (int): The 0-indexed destination ToR.
        
    Returns:
        list[int]: The list of ToR indices in the shortest path,
                   or None if no path exists.
    """
    
    num_nodes = len(A)
    num_racks = len(A[0])
    
    # --- 1. Build the Adjacency List (NOW WITH FILTERING) ---
    
    adj_list = {i: set() for i in range(num_nodes)}
    
    # Convert to a set for fast O(1) lookups
    broken_racks_set = set(broken_racks)
    
    for i in range(num_nodes):  # i is the 0-indexed current node
        for j in range(num_racks):  # j is the rack (column)
            
            # --- THIS IS THE NEW LOGIC ---
            # If the rack 'j' is in our broken set, skip this connection
            if j in broken_racks_set:
                continue
            # --- END OF NEW LOGIC ---
            
            # Get the neighbor's value (which is 1-indexed)
            neighbor_val = A[i][j]
            
            # Convert to a 0-indexed neighbor
            neighbor_idx = neighbor_val - 1
            
            if neighbor_idx != i:
                adj_list[i].add(neighbor_idx)

    # --- 2. Handle Invalid Inputs ---
    if start_tor == end_tor:
        return [start_tor]

    # --- 3. Initialize for BFS ---
    # (Note: We no longer have a 'broken_tors' set for nodes)
    visited = set()
    queue = collections.deque()
    
    # Check if start/end nodes are even reachable (e.g. isolated)
    # This isn't strictly necessary but good practice.
    if start_tor not in adj_list:
        return None

    queue.append((start_tor, [start_tor])) # (current_node, path_to_this_node)
    visited.add(start_tor)
    
    # --- 4. Run BFS (This part is identical to before) ---
    while queue:
        current_tor, path = queue.popleft()
        
        for neighbor in adj_list[current_tor]:
            if neighbor not in visited:
                if neighbor == end_tor:
                    return path + [neighbor]
                    
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
                
    # --- 5. No Path Found ---
    return None

# The 8x8 matrix A from your image
A = generate_random_latin_square(8)

start = 0
end = 7 # (Which is node value 8 in the matrix)

# --- Case 1: No broken racks ---
path = find_optimal_path_broken_racks(A, set(), start, end)
print(f"Case 1: No broken racks")
print(f"Path from {start} to {end}: {path}")
# Expected: [0, 7]

print("-" * 20)

# --- Case 2:  ---
broken = {3,4,5} 
path = find_optimal_path_broken_racks(A, broken, start, end)
print(f"Case 2: Rack 3,4,5 is broken")
print(f"Path from {start} to {end}: {path}")

print("-" * 20)

# --- Case 3 ---
broken = {3,4,5,6,7} 
path = find_optimal_path_broken_racks(A, broken, start, end)
print(f"Case 3: Rack 1,2,5,6,7 is broken")
print(f"Path from {start} to {end}: {path}")

print("------------------------------------------------------------")




# --- Weighted Graph ---


def find_least_cost_path_weighted_racks(A, rack_weights, start_tor, end_tor):
    """
    Finds the *least cost* path given weighted racks.
    Uses Dijkstra's algorithm.
    
    Args:
        A (list[list[int]]): 
            The N x M matrix. A[i][j] = k means node 'i'
            connects to node 'k' (1-indexed) via rack 'j'.
            
        rack_weights (list[float]): 
            A list of costs for each rack. 
            Use float('inf') to represent a "broken" rack.
            
        start_tor (int): The 0-indexed starting ToR.
        end_tor (int): The 0-indexed destination ToR.
        
    Returns:
        tuple (float, list[int]): 
            A tuple of (total_cost, path_list)
            or (float('inf'), None) if no path exists.
    """
    
    num_nodes = len(A)
    num_racks = len(A[0])
    
    # --- 1. Build a WEIGHTED Adjacency List ---
    # The format will be: {node: [(cost, neighbor), (cost, neighbor), ...]}
    adj_list = {i: [] for i in range(num_nodes)}
    
    for i in range(num_nodes):  # i is the 0-indexed current node
        for j in range(num_racks):  # j is the rack (column)
            
            cost = rack_weights[j]
            
            # If the rack is "broken" (infinite cost), skip it
            if cost == float('inf'):
                continue
                
            neighbor_val = A[i][j]
            neighbor_idx = neighbor_val - 1
            
            # Don't add self-loops if they don't help
            if neighbor_idx != i:
                adj_list[i].append((cost, neighbor_idx))

    # --- 2. Handle Invalid Inputs ---
    if start_tor == end_tor:
        return (0, [start_tor]) # Cost is 0

    # --- 3. Initialize for Dijkstra's Algorithm ---
    
    # Priority queue stores: (current_total_cost, current_node, path_so_far)
    # heapq always pops the tuple with the *smallest* first element (total_cost)
    pq = [(0, start_tor, [start_tor])] 
    
    # 'visited' stores nodes for which we have found the *final* cheapest path
    visited = set()
    
    # --- 4. Run Dijkstra's ---
    while pq:
        # Get the node with the lowest cost from the start
        current_cost, current_tor, path = heapq.heappop(pq)
        
        # If we've already found a cheaper path to this node, skip
        if current_tor in visited:
            continue
            
        visited.add(current_tor)

        # --- 5. Goal Check ---
        if current_tor == end_tor:
            # Found the cheapest path!
            return (current_cost, path)

        # --- 6. Explore Neighbors ---
        for edge_cost, neighbor in adj_list[current_tor]:
            if neighbor not in visited:
                new_cost = current_cost + edge_cost
                new_path = path + [neighbor]
                # Add the new path to the priority queue
                heapq.heappush(pq, (new_cost, neighbor, new_path))

    # --- 7. No Path Found ---
    return (float('inf'), None)

# --- EXAMPLE USAGE ---

A = generate_random_latin_square(8)

start = 0
end = 7

# --- Case 1: All racks have a simple cost ---
rack_weights_1 = [10, 10, 10, 15, 10, 10, 10, 10]
print("--- Case 1: Rack 3 is expensive ---")
cost, path = find_least_cost_path_weighted_racks(A, rack_weights_1, start, end)
print(f"Path from {start} to {end}: {path}")
print(f"Total cost: {cost}\n")


print("-" * 20)

# --- Case 2: Make the 1-hop path "broken" ---
# A "broken" rack just has infinite cost
rack_weights_2 = [1, 1, 1, float('inf'), 1, 1, 1, 1]

print("--- Case 2: Rack 3 is 'broken' (cost=inf) ---")
cost, path = find_least_cost_path_weighted_racks(A, rack_weights_2, start, end)
print(f"Path from {start} to {end}: {path}")
print(f"Total cost: {cost}\n")

print("-" * 20)

# --- Case 3: A more complex cost scenario ---
rack_weights_3 = [5, 20, 5, 5, 20, 20, 5, 20]
# Racks 0, 2, 3, 6 are cheap (cost 5)
# Racks 1, 4, 5, 7 are expensive (cost 20)
print("--- Case 3: Mixed cheap/expensive racks ---")
cost, path = find_least_cost_path_weighted_racks(A, rack_weights_3, start, end)
print(f"Path from {start} to {end}: {path}")
print(f"Total cost: {cost}\n")