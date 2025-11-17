import numpy as np
import collections
import random

def generate_simple_latin_square(N):
    """
    Constructs a simple round-robin matrix.

    Each element i (0 ≤ i < node) generates a list that:
    - Contains all indices from 0 to node - 1,
    - But starts the list at index i (wrapping around).

    Example:
        RR1(3) → [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1]
        ]

    Args:
        node (int): Number of nodes (or positions).

    Returns:
        list[list[int]]: The generated round-robin structure.
    """
    first_row = np.arange(1, N + 1)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        A[i] = np.roll(first_row, -i)
        
    return A.tolist()

# --- Generate the 8x8 matrix ---
N = 8
A_simple = generate_simple_latin_square(N)

print(f"--- Simple {N}x{N} Latin Square ---")
print(A_simple)


def generate_random_latin_square(N):
    """
    Generates a randomized N x N Latin Square.
    """
    base_row = np.arange(1, N + 1)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        A[i] = np.roll(base_row, -i)
        
    A = np.random.permutation(A)
    A = np.random.permutation(A.T).T

    shuffled_values = np.random.permutation(np.arange(1, N + 1))
    
    value_map = {original: new for original, new in enumerate(shuffled_values, 1)}

    for i in range(N):
        for j in range(N):
            A[i, j] = value_map[A[i, j]]
            
    return A.tolist()

# --- Generate the 8x8 matrix ---
N = 8
A_random = generate_random_latin_square(N)

print(f"--- Randomized {N}x{N} Latin Square ---")
print(A_random)



def is_solvable(list_of_sets):
    """
    Checks the Pigeonhole Principle constraint.
    (This helper function is unchanged)
    """
    if not list_of_sets:
        return True
    
    k = len(list_of_sets[0])
    
    counts = collections.Counter()
    for s in list_of_sets:
        if len(s) != k:
            raise ValueError("Sets do not have an equal number of elements.")
        counts.update(s)
        
    for num, count in counts.items():
        if count > k:
            print(f"Impossible: Number {num} appears in {count} sets, but k is only {k}.")
            return False
            
    return True

def create_constrained_matrix(list_of_sets):
    """
    Finds a matrix arrangement that satisfies the column constraint.
    Iterative backtracking without recursion.
    """
    if not list_of_sets:
        return []

    if not is_solvable(list_of_sets):
        return None

    n = len(list_of_sets) 
    k = len(list_of_sets[0])
    
    matrix = [[None for _ in range(k)] for _ in range(n)]
    rows_used = [set() for _ in range(n)]
    cols_used = [set() for _ in range(k)]
    
    pos = 0  # Current position (0 to n*k - 1)
    attempts = [0] * (n * k)  # Track which number index we're trying at each position
    
    while pos >= 0:
        if pos == n * k:
            # Found a solution
            return matrix
        
        row = pos // k
        col = pos % k
        current_set = list(list_of_sets[row])
        
        # Undo previous placement at this position
        if matrix[row][col] is not None:
            rows_used[row].remove(matrix[row][col])
            cols_used[col].remove(matrix[row][col])
            matrix[row][col] = None
        
        # Try numbers starting from the current attempt index
        found = False
        for i in range(attempts[pos], len(current_set)):
            num = current_set[i]
            if num not in rows_used[row] and num not in cols_used[col]:
                matrix[row][col] = num
                rows_used[row].add(num)
                cols_used[col].add(num)
                attempts[pos] = i + 1  # Next time, try the next number
                found = True
                break
        
        if found:
            # Move forward
            pos += 1
        else:
            # Backtrack
            attempts[pos] = 0  # Reset attempts for this position
            pos -= 1
    
    return None

print(create_constrained_matrix([{0,1},{2,1},{0,3}]))

def create_random_constrained_matrix(list_of_sets):
    """
    Finds a random matrix arrangement that satisfies the column constraint.
    Iterative backtracking without recursion.
    """
    if not list_of_sets:
        return []

    if not is_solvable(list_of_sets):
        return None

    n = len(list_of_sets) 
    k = len(list_of_sets[0])
    
    matrix = [[None for _ in range(k)] for _ in range(n)]
    rows_used = [set() for _ in range(n)]
    cols_used = [set() for _ in range(k)]
    
    # Pre-shuffle the numbers for each row to randomize
    shuffled_sets = [list(s) for s in list_of_sets]
    for s in shuffled_sets:
        random.shuffle(s)
    
    pos = 0  # Current position (0 to n*k - 1)
    attempts = [0] * (n * k)  # Track which number index we're trying at each position
    
    while pos >= 0:
        if pos == n * k:
            # Found a solution
            return matrix
        
        row = pos // k
        col = pos % k
        current_set = shuffled_sets[row]
        
        # Undo previous placement at this position
        if matrix[row][col] is not None:
            rows_used[row].remove(matrix[row][col])
            cols_used[col].remove(matrix[row][col])
            matrix[row][col] = None
        
        # Try numbers starting from the current attempt index
        found = False
        for i in range(attempts[pos], len(current_set)):
            num = current_set[i]
            if num not in rows_used[row] and num not in cols_used[col]:
                matrix[row][col] = num
                rows_used[row].add(num)
                cols_used[col].add(num)
                attempts[pos] = i + 1  # Next time, try the next number
                found = True
                break
        
        if found:
            # Move forward
            pos += 1
        else:
            # Backtrack
            attempts[pos] = 0  # Reset attempts for this position
            pos -= 1
    
    return None
    

print(create_random_constrained_matrix([{0,1},{2,1},{0,3}]))