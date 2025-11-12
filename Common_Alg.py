import numpy as np
import random

def generate_simple_latin_square(N):
    # Create the first row: [1, 2, 3, ..., N]
    first_row = np.arange(1, N + 1)
    # Create an empty N x N matrix
    A = np.zeros((N, N), dtype=int)
    # Fill each row by 'rolling' the first row
    for i in range(N):
        A[i] = np.roll(first_row, -i)
        
    return A

# --- Generate the 8x8 matrix ---
N = 8
A_simple = generate_simple_latin_square(N)

print(f"--- Simple {N}x{N} Latin Square ---")
print(A_simple)


def generate_random_latin_square(N):
    """
    Generates a randomized N x N Latin Square.
    """
    # 1. Start with the simple cyclic square
    base_row = np.arange(1, N + 1)
    A = np.zeros((N, N), dtype=int)
    for i in range(N):
        A[i] = np.roll(base_row, -i)
        
    # 2. Randomly shuffle the rows
    # np.random.permutation(A) shuffles the rows
    A = np.random.permutation(A)
    
    # 3. Randomly shuffle the columns
    # To shuffle columns, we transpose, shuffle rows, and transpose back
    A = np.random.permutation(A.T).T

    # 4. (Optional) Randomly "relabel" the numbers
    # e.g., all 1s -> 5s, all 2s -> 3s, etc.
    shuffled_values = np.random.permutation(np.arange(1, N + 1))
    
    # Create a mapping: 1->shuffled[0], 2->shuffled[1], ...
    value_map = {original: new for original, new in enumerate(shuffled_values, 1)}

    # Apply the mapping
    # (This is a bit more advanced, but shows how to get a
    # truly random-looking result)
    for i in range(N):
        for j in range(N):
            A[i, j] = value_map[A[i, j]]
            
    return A

# --- Generate the 8x8 matrix ---
N = 8
A_random = generate_random_latin_square(N)

print(f"--- Randomized {N}x{N} Latin Square ---")
print(A_random)