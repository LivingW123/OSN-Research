import collections
import heapq  # We need heapq for the priority queue, not collections.deque
import numpy as np # Using numpy for the latin square generator

from Common_Alg import(
    generate_random_latin_square
)

def find_path_2d(A, start_node, end_node, max_hops=100):
    """
    Finds a path in a 2D (time-slotted) Opera network where the number of hops 
    required is exactly the number of timeslots elapsed.
    
    Args:
        A (list[list[int]]): N x M matrix where A[i][t % M] is the neighbor 
                             of node i at timeslot t.
        start_node (int): 0-indexed start node.
        end_node (int): 0-indexed destination node.
        max_hops (int): Maximum number of hops/timeslots to search.
        
    Returns:
        tuple (list[int], int): (path, num_hops) or (None, None)
    """
    num_nodes = len(A)
    num_racks = len(A[0])
    
    # State: (current_node, timeslot)
    # Use BFS to find the path with fewest hops (and thus fewest timeslots)
    queue = collections.deque([(start_node, 0, [start_node])])
    visited = set([(start_node, 0)])
    
    while queue:
        u, t, path = queue.popleft()
        
        if u == end_node:
            return path, t
            
        if t >= max_hops:
            continue
            
        # At timeslot 't', the available neighbor is determined by column 't % num_racks'
        neighbor_idx = A[u][t % num_racks] - 1
        
        # Move to the neighbor in the next timeslot
        new_state = (neighbor_idx, t + 1)
        if new_state not in visited:
            visited.add(new_state)
            queue.append((neighbor_idx, t + 1, path + [neighbor_idx]))
            
    return None, None


def find_path_2d_grid(B, start_node, end_node, max_hops=100):
    """
    Two-Dimensional Opera on a B x B Grid.
    Number of hops required is exactly the number of timeslots elapsed.
    At each timeslot, we alternate between jumping in the row dimension 
    and the column dimension.
    
    Args:
        B (int): The side length of the grid (Total nodes N = B*B).
        start_node (int): 0-indexed start node.
        end_node (int): 0-indexed destination node.
        max_hops (int): Maximum timeslots/hops.
        
    Returns:
        tuple (list[int], int): (path, num_hops)
    """
    num_nodes = B * B
    
    # State: (current_node, timeslot)
    queue = collections.deque([(start_node, 0, [start_node])])
    visited = set([(start_node, 0)])
    
    while queue:
        u, t, path = queue.popleft()
        
        if u == end_node:
            return path, t
            
        if t >= max_hops:
            continue
            
        r, c = divmod(u, B)
        
        # Determine available neighbors at timeslot t
        # In a 2D Grid Opera, we alternate dimensions.
        if t % 2 == 0:
            # Allow jumping to ANY node in the current row (Expanders/Rotor behavior)
            for c_new in range(B):
                if c_new != c:
                    v = r * B + c_new
                    if (v, t + 1) not in visited:
                        visited.add((v, t+1))
                        queue.append((v, t+1, path + [v]))
        else:
            # Odd timeslot: Column dimension neighbors
            for r_new in range(B):
                if r_new != r:
                    v = r_new * B + c
                    if (v, t + 1) not in visited:
                        visited.add((v, t + 1))
                        queue.append((v, t + 1, path + [v]))
    return None, None



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
            
            # If the rack 'j' is in our broken set, skip this connection
            if j in broken_racks_set:
                continue
            
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
    visited = set()
    queue = collections.deque()
    
    if start_tor not in adj_list:
        return None

    queue.append((start_tor, [start_tor])) # (current_node, path_to_this_node)
    visited.add(start_tor)
    
    # --- 4. Run BFS ---
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

def find_least_cost_path_weighted_racks(A, rack_weights, start_tor, end_tor):
    """
    Finds the *least cost* path given weighted racks.
    Uses Dijkstra's algorithm.
    """
    
    num_nodes = len(A)
    num_racks = len(A[0])
    
    adj_list = {i: [] for i in range(num_nodes)}
    
    for i in range(num_nodes):  # i is the 0-indexed current node
        for j in range(num_racks):  # j is the rack (column)
            cost = rack_weights[j]
            if cost == float('inf'):
                continue
            neighbor_val = A[i][j]
            neighbor_idx = neighbor_val - 1
            if neighbor_idx != i:
                adj_list[i].append((cost, neighbor_idx))

    if start_tor == end_tor:
        return (0, [start_tor])

    pq = [(0, start_tor, [start_tor])] 
    visited = set()
    
    while pq:
        current_cost, current_tor, path = heapq.heappop(pq)
        if current_tor in visited:
            continue
        visited.add(current_tor)

        if current_tor == end_tor:
            return (current_cost, path)

        for edge_cost, neighbor in adj_list[current_tor]:
            if neighbor not in visited:
                new_cost = current_cost + edge_cost
                new_path = path + [neighbor]
                heapq.heappush(pq, (new_cost, neighbor, new_path))

    return (float('inf'), None)

def check_guard_band(arrival_times, slot_duration, guard_band_duration):
    """
    Checks if packet arrival times drift into the forbidden guard band region.
    """
    violations = []
    is_synced = True
    safe_limit = slot_duration - guard_band_duration
    for t in arrival_times:
        offset = t % slot_duration
        if offset > safe_limit:
            is_synced = False
            violations.append(t)
    return is_synced, violations

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    A = generate_random_latin_square(8)
    start = 0
    end = 7

    print("--- Testing Shortest Path (Static) ---")
    path = find_optimal_path_broken_racks(A, set(), start, end)
    print(f"Path: {path}\n")

    print("--- Testing Weighted Path (Static) ---")
    rack_weights = [5, 20, 5, 5, 20, 20, 5, 20]
    cost, path = find_least_cost_path_weighted_racks(A, rack_weights, start, end)
    print(f"Cost: {cost}, Path: {path}\n")

    print("=== Testing 2D Opera (Hops = Timeslots) ===")
    path_2d, hops_2d = find_path_2d(A, start, end)
    if path_2d:
        print(f"Space-Time Path: {path_2d}, Hops/Timeslots: {hops_2d}")
    else:
        print("No Space-Time path found.")

    print("\n=== Testing 2D Grid Opera (B=4, N=16) ===")
    B_grid = 4
    s_grid = 0  # (0,0)
    d_grid = 15 # (3,3)
    path_grid, hops_grid = find_path_2d_grid(B_grid, s_grid, d_grid)
    if path_grid:
        print(f"Grid Path: {path_grid}")
        print(f"Hops/Timeslots: {hops_grid}")

