from mathfunc import(
    find_closest_factors_positive
)

def SiriusGen(nodes):
    delta, port = find_closest_factors_positive(nodes)
    source = {}
    for i in range(nodes):
        for j in range(port):
            source[(nodes, port)] = []
            for k in range(delta):
                source[(nodes, port)].append(j+k)
    return source
            
print(SiriusGen(4))

def matrix_multiply(A, B):
    """
    Helper function to multiply two matrices A and B.
    """
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
        raise ValueError("Cannot multiply matrices: dimensions do not match.")

    # Create result matrix of size rows_A x cols_B
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result

def create_matrix_A(wavelengths, ports, nodes):
    """
    Generates the initial Matrix A (A1).
    """
    matrix_A = [[0] * ports for _ in range(nodes)]
    current_val = 1
    rows_per_block = wavelengths
    num_blocks = nodes // wavelengths
    
    for b in range(num_blocks):
        row_start = b * rows_per_block
        for col in range(ports):
            for r_offset in range(rows_per_block):
                row_idx = row_start + r_offset
                matrix_A[row_idx][col] = current_val
                current_val += 1
    return matrix_A

def create_matrix_W(wavelengths, nodes):
    """
    Generates the initial Matrix W (W1) using stride permutation.
    """
    perm_order = []
    for start_idx in range(wavelengths):
        sequence = range(start_idx, nodes, wavelengths)
        perm_order.extend(list(sequence))
        
    matrix_W = [[0] * nodes for _ in range(nodes)]
    for col_idx, target_row in enumerate(perm_order):
        matrix_W[target_row][col_idx] = 1
        
    return matrix_W

def create_matrix_P(wavelengths, nodes):
    """
    Generates the Transformation Matrix P.
    P consists of block-diagonal submatrices that perform a cyclic shift.
    (It moves Row 2 to Row 1, Row 3 to Row 2, Row 1 to Row 3, etc.)
    """
    P = [[0] * nodes for _ in range(nodes)]
    num_blocks = nodes // wavelengths

    for b in range(num_blocks):
        block_offset = b * wavelengths
        for r in range(wavelengths):
            # We want New Row 'r' to come from Old Row '(r + 1)'
            # So we place a 1 at (row=r, col=(r+1)%w) within the block
            
            row_idx = block_offset + r
            col_idx = block_offset + ((r + 1) % wavelengths)
            
            P[row_idx][col_idx] = 1
            
    return P

def generate_full_system(wavelengths, ports, nodes):
    """
    Generates all As and Ws in the cycle.
    """
    # 1. Create Initial State
    A_list = [create_matrix_A(wavelengths, ports, nodes)]
    W_list = [create_matrix_W(wavelengths, nodes)]
    
    # 2. Create Transformation Matrix P
    P = create_matrix_P(wavelengths, nodes)
    
    # 3. Generate the full cycle
    # The cycle length is equal to the number of wavelengths.
    # We already have the 1st one, so we generate w-1 more.
    for _ in range(wavelengths - 1):
        # A_next = P * A_current
        next_A = matrix_multiply(P, A_list[-1])
        A_list.append(next_A)
        
        # W_next = P * W_current
        next_W = matrix_multiply(P, W_list[-1])
        W_list.append(next_W)
        
    return A_list, W_list, P

# --- Execution & Verification ---

def print_matrix(name, matrix):
    print(f"{name} = [")
    for row in matrix:
        print(f"  {row},")
    print("]")

# Test Case 1: Example 4.11 (2 Wavelengths -> Should generate A1, A2)
print("--- Generating System for Example 4.11 (w=2) ---")
As_11, Ws_11, P_11 = generate_full_system(wavelengths=2, ports=3, nodes=6)

for i, mat in enumerate(As_11):
    print_matrix(f"A{i+1}", mat)

for i, mat in enumerate(Ws_11):
    print_matrix(f"W{i+1}", mat)

print("\n" + "="*30 + "\n")

# Test Case 2: Example 4.12 (3 Wavelengths -> Should generate A1, A2, A3)
print("--- Generating System for Example 4.12 (w=3) ---")
As_12, Ws_12, P_12 = generate_full_system(wavelengths=3, ports=2, nodes=6)

for i, mat in enumerate(As_12):
    print_matrix(f"A{i+1}", mat)

for i, mat in enumerate(Ws_12):
    print_matrix(f"W{i+1}", mat)