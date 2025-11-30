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

def create_matrix_A(wavelengths, ports, nodes):
    """
    Generates Matrix A based on the block-filling pattern seen in the examples.
    """
    # Initialize an empty matrix with zeros
    matrix_A = [[0] * ports for _ in range(nodes)]
    
    current_val = 1
    
    # We fill the matrix in chunks of rows determined by 'wavelengths'
    # For Example 4.11: 6 nodes / 2 wavelengths = 3 blocks
    rows_per_block = wavelengths
    num_blocks = nodes // wavelengths
    
    for b in range(num_blocks):
        row_start = b * rows_per_block
        
        # Inside each block, we fill column by column
        for col in range(ports):
            for r_offset in range(rows_per_block):
                row_idx = row_start + r_offset
                matrix_A[row_idx][col] = current_val
                current_val += 1
                
    return matrix_A

def create_matrix_W(wavelengths, nodes):
    """
    Generates Matrix W based on the stride permutation pattern.
    (Columns are reordered by taking every w-th index).
    """
    # 1. Generate the target permutation order
    # Example 4.11 (w=2): Start at 1, jump 2 -> [1, 3, 5]. Then start 2, jump 2 -> [2, 4, 6]
    # Example 4.12 (w=3): Start at 1, jump 3 -> [1, 4]. Then [2, 5]. Then [3, 6].
    
    perm_order = []
    for start_idx in range(wavelengths):
        # We use range(start, end, step)
        # We use 0-based indexing for calculation, then map to matrix positions
        sequence = range(start_idx, nodes, wavelengths)
        perm_order.extend(list(sequence))
        
    # 2. Build the Identity matrix based on this column order
    matrix_W = [[0] * nodes for _ in range(nodes)]
    
    for col_idx, target_row in enumerate(perm_order):
        # In the examples, the columns of W are the basis vectors e_i
        # in the permuted order. 
        # So column 0 has a 1 at row 'perm_order[0]'
        matrix_W[target_row][col_idx] = 1
        
    return matrix_W

# --- Verification with your specific examples ---

def print_matrix(name, matrix):
    print(f"{name} = [")
    for row in matrix:
        print(f"  {row},")
    print("]")
    print()

# Example 4.11: 2 Wavelengths, 3 Ports, 6 Nodes
print("--- Example 4.11 (2 Wavelengths, 3 Ports, 6 Nodes) ---")
A_11 = create_matrix_A(wavelengths=2, ports=3, nodes=6)
W_11 = create_matrix_W(wavelengths=2, nodes=6)

print_matrix("A1", A_11)
print_matrix("W_1,3,5", W_11)

# Example 4.12: 3 Wavelengths, 2 Ports, 6 Nodes
print("--- Example 4.12 (3 Wavelengths, 2 Ports, 6 Nodes) ---")
A_12 = create_matrix_A(wavelengths=3, ports=2, nodes=6)
W_12 = create_matrix_W(wavelengths=3, nodes=6)

print_matrix("A1", A_12)
print_matrix("W_1,4", W_12)