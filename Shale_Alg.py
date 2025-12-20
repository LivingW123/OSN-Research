import itertools

from mathfunc import(
    base10_conversion
)

from Common_Alg import(
    generate_simple_latin_square,
    create_constrained_matrix
)



import heapq

def dijkstra(adj_matrix, weights, start_node, end_node):
    """
    Standard Dijkstra's algorithm to find shortest path.
    """
    # Priority queue: (cost, current_node, path_associated)
    queue = [(weights.get(start_node, 0), start_node, [start_node])]
    visited = {} # node -> cost
    
    while queue:
        cost, u, path = heapq.heappop(queue)
        
        if u == end_node:
            return path, cost
            
        if u in visited and visited[u] <= cost:
            continue
        visited[u] = cost
        
        if u < len(adj_matrix):
            neighbors = adj_matrix[u]
            if neighbors is not None:
                for v in neighbors:
                    if v is not None:
                        new_cost = cost + weights.get(v, 0)
                        if v not in visited or new_cost < visited[v]:
                            heapq.heappush(queue, (new_cost, v, path + [v]))
    return None, float('inf')


def spray_short(adj_matrix, weights, start_node, end_node, k=3, penalty_factor=2.0):
    """
    Finds k paths that are short but diverse (sprayed) to avoid congestion.
    
    Iteratively finds the shortest path, then penalizes the nodes used in that path
    (multiplying their weight by penalty_factor) to encourage subsequent paths
    to choose different routes.
    
    Args:
        adj_matrix (list): Adjacency matrix.
        weights (dict): Node weights.
        start_node (int): Start node.
        end_node (int): End node.
        k (int): Number of paths to find.
        penalty_factor (float): Multiplier for used node weights.
        
    Returns:
        list[tuple]: List of (path, original_cost) tuples.
    """
    
    # Work with a copy of weights so we can penalize without affecting original global state permanently
    # But for multiple calls we need a local working copy.
    current_weights = weights.copy()
    
    found_paths = []
    
    for _ in range(k):
        path, cost = dijkstra(adj_matrix, current_weights, start_node, end_node)
        
        if path is None:
            break
            
        # Calculate ORIGINAL cost for reporting
        original_cost = sum(weights.get(n, 0) for n in path)
        found_paths.append((tuple(path), original_cost))
        
        # Penalize nodes in this path (except start and end which are mandatory)
        for node in path:
            if node != start_node and node != end_node:
                current_weights[node] = current_weights.get(node, 0) * penalty_factor
                
    return found_paths

def RR1(node):
    return create_constrained_matrix(generate_simple_latin_square(node))

def RR2_path(adj_matrix, weights, start_node, end_node):
    """
    Finds the top 10 simple paths with the lowest costs between start_node and end_node.
    
    Args:
        adj_matrix (list[list]): Adjacency matrix where adj_matrix[i] contains neighbors of node i.
                                  May contain None values.
        weights (dict): A dictionary mapping node index to its weight/cost.
        start_node (int): The starting node index.
        end_node (int): The ending node index.
        
    Returns:
        dict: A dictionary where keys are tuples representing paths and values are their total costs.
              Only the 10 paths with the lowest costs are returned.
    """
    
    # Store all found simple paths and their costs
    # List of tuples: (cost, path_tuple)
    all_paths = []
    
    # Stack for DFS: (current_node, current_path, current_cost)
    # path includes start_node
    initial_cost = weights.get(start_node, 0)
    stack = [(start_node, [start_node], initial_cost)]
    
    while stack:
        curr, path, cost = stack.pop()
        
        if curr == end_node:
            all_paths.append((cost, tuple(path)))
            continue
            
        # Get neighbors
        if curr < len(adj_matrix):
            neighbors = adj_matrix[curr]
            if neighbors is not None:
                for neighbor in neighbors:
                    if neighbor is not None and neighbor not in path:
                         # Calculate new cost
                        new_cost = cost + weights.get(neighbor, 0)
                        
                        # Create new path list
                        new_path = path + [neighbor]
                        
                        stack.append((neighbor, new_path, new_cost))
                        
    # Sort by cost
    all_paths.sort(key=lambda x: x[0])
    
    # Keep only top 10
    top_10 = all_paths[:10]
    
    # Convert to expected dictionary format {path: cost}
    result = {}
    for cost, path in top_10:
        result[path] = cost
        
    return result



def RR2(base, dimension):
    """
    Generates the adjacency sets for all tuples in a base-'base', 'dimension'-dimensional space
    where each neighbor differs by exactly one coordinate.

    For each tuple `big` (of length = dimension):
      - Iterates through every position.
      - Replaces the value at that position with all possible base values.
      - Collects all unique resulting tuples except the original itself.
      - Converts all neighbor tuples into base-10 integers.

    Args:
        base (int): The numerical base (e.g., 2 for binary, 3 for ternary).
        dimension (int): The number of positions per tuple.

    Returns:
        list[set[int]]: Each index corresponds to a tuple in lexicographic order,
                        containing a set of neighbors (in base-10 form).
    """
    choice = [x for x in range(base)]

    bigger = list(itertools.product(choice, repeat=dimension))
    smaller = list(itertools.product(choice, repeat=dimension - 1))
    
    mat = []
    for big in bigger:
        temp = set()
        for indb, b in enumerate(big):
            for small in smaller:
                new = list(small)
                new.insert(indb, b)
                temp.add(tuple(new))
        
        # Exclude the original tuple itself
        temp.remove(big)
        
        # Convert tuples to base-10 integers
        final_set = {base10_conversion(base, t, dimension) for t in temp}
        mat.append(final_set)
    return create_constrained_matrix(mat)


def RR3(base, dimension, t):
    """
    Generalized adjacency generator.

    Produces all tuples that differ from a given 'big' tuple in exactly 't' positions.
    Each of those differing positions can take any other value in the given base.

    Uses:
        - itertools.product for all possible tuples
        - itertools.combinations for choosing which positions to change

    Args:
        base (int): The numerical base (e.g., 3 for ternary, 4 for quaternary).
        dimension (int): The length of each tuple.
        t (int): Number of positions that differ from the original tuple.

    Returns:
        list[set[int]]: Each entry corresponds to one base-'base' tuple,
                        containing all neighbors differing in 't' positions,
                        expressed in base-10 form.
    """
    choice = [x for x in range(base)]
    bigger = list(itertools.product(choice, repeat=dimension))
    mat = []

    # All combinations of positions that can differ
    position_combinations = list(itertools.combinations(range(dimension), t))
    
    # All possible value combinations for those differing positions
    new_values_list = list(itertools.product(choice, repeat=t))

    for big in bigger:
        temp = set()
        for positions in position_combinations:
            for new_values in new_values_list:
                new = list(big)
                all_t_are_different = True
                
                for i, pos in enumerate(positions):
                    if new_values[i] == big[pos]:
                        all_t_are_different = False
                        break
                    new[pos] = new_values[i]
                
                if all_t_are_different:
                    temp.add(tuple(new))
        
        # Convert tuples to base-10 integers
        final_set = {base10_conversion(base, t, dimension) for t in temp}
        mat.append(final_set)
    return create_constrained_matrix(mat)


# --- Example Outputs ---
print("### Optimized Functions ###")

print(f"RR1(4):\n{RR1(4)}")

print(f"RR2(3, 2):\n{RR2(3, 2)}")
print("\n---\n")


print(f"RR3(3, 2, 1):\n{RR3(3,2,1)}")
print("\n---\n")

RR3_out = RR3(4, 3, 2)
print(f"RR3(4, 3, 2):\n{RR3_out}")
print("\n---\n")

rr3_opt_output = RR3(4, 3, 2)
print(f"RR3(4, 3, 2) [Neighbors of (0,0,0)]:\n{rr3_opt_output[0]}")
print("\n---\n")
# print(RR3_out)
# for x in RR3_out:
#     print(len(x))