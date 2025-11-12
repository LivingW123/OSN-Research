import collections

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
A = [
    [3, 7, 1, 8, 5, 2, 6, 4],  # Node 0
    [7, 8, 4, 5, 3, 1, 2, 6],  # Node 1
    [1, 4, 5, 3, 2, 6, 7, 8],  # Node 2
    [5, 3, 2, 4, 6, 7, 8, 1],  # Node 3
    [4, 6, 3, 2, 1, 8, 1, 7],  # Node 4
    [6, 5, 8, 7, 4, 3, 3, 2],  # Node 5
    [2, 1, 7, 6, 8, 4, 4, 5],  # Node 6
    [8, 2, 6, 1, 7, 5, 5, 3]   # Node 7
]

start = 0
end = 7 # (Which is node value 8 in the matrix)

# --- Case 1: No broken racks ---
# A[0][3] = 8. This is a 1-hop path from 0 to 7 via rack 3.
path = find_optimal_path_broken_racks(A, set(), start, end)
print(f"Case 1: No broken racks")
print(f"Path from {start} to {end}: {path}")
# Expected: [0, 7]

print("-" * 20)

# --- Case 2: The 1-hop path is broken ---
# Let's break rack 3
broken = {3} 
path = find_optimal_path_broken_racks(A, broken, start, end)
print(f"Case 2: Rack 3 is broken")
print(f"Path from {start} to {end}: {path}")

# --- Trace ---
# 1. The 1-hop path [0, 7] (via rack 3) is now impossible.
# 2. BFS checks other neighbors of 0. Let's say it finds node 6
#    (A[0][6] = 6, via rack 6).
# 3. BFS then checks neighbors of 6.
#    (A[6][4] = 8). This is node 7, via rack 4.
# 4. Rack 4 is not broken, so this path is valid.
# Expected: [0, 6, 7] (or another valid 2-hop path)