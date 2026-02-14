from mathfunc import find_closest_factors_positive
import numpy as np


def SiriusGen(nodes):
    """
    Generates the Sirius source matrix mapping for a given number of nodes.
    Uses mod arithmetic to transfer AWGR routing labels to actual node indices.

    For N nodes factored as N = wavelengths * ports:
        source[node] = list of destination nodes across all ports and wavelengths
        dest(s, k, i) = (A^i_{s,k} - 1) mod N

    Returns:
        dict: Mapping from each node index to its list of reachable destination nodes
              across all timeslots.
    """
    wavelengths, ports = find_closest_factors_positive(nodes)
    As, Ws, P = generate_full_system(wavelengths, ports, nodes)

    source = {}
    for node_idx in range(nodes):
        destinations = set()
        for A in As:
            for val in A[node_idx]:
                dest = (val - 1) % nodes  # Mod-based transfer to node index
                if dest != node_idx:
                    destinations.add(dest)
        source[node_idx] = sorted(destinations)
    return source


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
    Generates the initial Source Matrix A (A^1).
    This is an r x l matrix (nodes x ports) containing AWGR routing labels.
    Labels are assigned in blocks of size 'wavelengths'.
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
    Generates the initial Connectivity Matrix W (W^1) using stride permutation.
    This is an r x r (nodes x nodes) binary adjacency matrix derived from
    the source matrix via modular arithmetic:
        W^i_{(A^i_{s,k} - 1) mod r, s} = 1
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
    Used to generate successive timeslots: A^{i+1} = P * A^i, W^{i+1} = P * W^i
    """
    P = [[0] * nodes for _ in range(nodes)]
    num_blocks = nodes // wavelengths

    for b in range(num_blocks):
        block_offset = b * wavelengths
        for r in range(wavelengths):
            row_idx = block_offset + r
            col_idx = block_offset + ((r + 1) % wavelengths)
            P[row_idx][col_idx] = 1

    return P


def generate_full_system(wavelengths, ports, nodes):
    """
    Generates all source matrices A^i and connectivity matrices W^i in the cycle.

    Returns:
        A_list: List of j source matrices [A^1, ..., A^j], each r x l
        W_list: List of j connectivity matrices [W^1, ..., W^j], each r x r
        P: The transformation matrix (r x r)
    """
    # 1. Create Initial State
    A_list = [create_matrix_A(wavelengths, ports, nodes)]
    W_list = [create_matrix_W(wavelengths, nodes)]

    # 2. Create Transformation Matrix P
    P = create_matrix_P(wavelengths, nodes)

    # 3. Generate the full cycle: j = wavelengths timeslots
    for _ in range(wavelengths - 1):
        # A^{i+1} = P * A^i
        next_A = matrix_multiply(P, A_list[-1])
        A_list.append(next_A)

        # W^{i+1} = P * W^i
        next_W = matrix_multiply(P, W_list[-1])
        W_list.append(next_W)

    return A_list, W_list, P


def generate_adjacency_from_source(A_list, nodes):
    """
    Converts a list of source matrices to an adjacency list using mod-based
    node transfer: dest(s, k, i) = (A^i_{s,k} - 1) mod r

    Args:
        A_list: List of source matrices [A^1, ..., A^j]
        nodes: Number of nodes r

    Returns:
        adj_list: List of lists, adj_list[s] = sorted unique destination nodes for node s
    """
    union_adj = [set() for _ in range(nodes)]
    for A in A_list:
        for s, row in enumerate(A):
            for val in row:
                dest = (val - 1) % nodes
                if dest != s:
                    union_adj[s].add(dest)
    return [sorted(dests) for dests in union_adj]


def generate_traffic_demand_matrix(A_list, nodes, demand_matrix=None):
    """
    Computes the Sirius Traffic Demand Matrix:
        T^Sir = sum_{i=1}^{j} D o W^i
    where D is the offered demand matrix and o is the Hadamard product.

    This represents the aggregate demand that can be delivered across one full
    scheduling cycle of j timeslots.

    Args:
        A_list: List of source matrices [A^1, ..., A^j]
        nodes: Number of nodes r
        demand_matrix: Optional r x r numpy array of offered demand.
                       If None, uses uniform demand (1 everywhere, 0 on diagonal).

    Returns:
        traffic_demand: r x r numpy array of aggregate delivered demand
    """
    if demand_matrix is None:
        demand_matrix = np.ones((nodes, nodes))
        np.fill_diagonal(demand_matrix, 0)

    traffic_demand = np.zeros((nodes, nodes))

    for A in A_list:
        # Build W^i from A^i using mod-based transfer
        W_i = np.zeros((nodes, nodes))
        for s, row in enumerate(A):
            for val in row:
                dest = (val - 1) % nodes
                if dest != s:
                    W_i[dest, s] = 1  # W^i_{dest, s} = 1

        # T^Sir += D o W^i (Hadamard product)
        traffic_demand += demand_matrix * W_i

    return traffic_demand


# --- Utility ---

def print_matrix(name, matrix):
    print(f"{name} = [")
    for row in matrix:
        print(f"  {row},")
    print("]")


if __name__ == "__main__":
    # Test Case 1: Example 4.11 (2 Wavelengths -> Should generate A1, A2)
    print("--- Generating System for Example 4.11 (w=2) ---")
    As_11, Ws_11, P_11 = generate_full_system(wavelengths=2, ports=3, nodes=6)

    for i, mat in enumerate(As_11):
        print_matrix(f"A{i+1}", mat)

    for i, mat in enumerate(Ws_11):
        print_matrix(f"W{i+1}", mat)

    print("\n--- Adjacency (mod-based) for Example 4.11 ---")
    adj_11 = generate_adjacency_from_source(As_11, 6)
    for i, neighbors in enumerate(adj_11):
        print(f"Node {i}: -> {neighbors}")

    print("\n--- Traffic Demand Matrix for Example 4.11 ---")
    T_11 = generate_traffic_demand_matrix(As_11, 6)
    print(T_11)

    print("\n" + "=" * 30 + "\n")

    # Test Case 2: Example 4.12 (3 Wavelengths -> Should generate A1, A2, A3)
    print("--- Generating System for Example 4.12 (w=3) ---")
    As_12, Ws_12, P_12 = generate_full_system(wavelengths=3, ports=2, nodes=6)

    for i, mat in enumerate(As_12):
        print_matrix(f"A{i+1}", mat)

    for i, mat in enumerate(Ws_12):
        print_matrix(f"W{i+1}", mat)

    print("\n--- Adjacency (mod-based) for Example 4.12 ---")
    adj_12 = generate_adjacency_from_source(As_12, 6)
    for i, neighbors in enumerate(adj_12):
        print(f"Node {i}: -> {neighbors}")

    print("\n--- Traffic Demand Matrix for Example 4.12 ---")
    T_12 = generate_traffic_demand_matrix(As_12, 6)
    print(T_12)

    # Test Case 3: SiriusGen convenience function
    print("\n" + "=" * 30)
    print("\n--- SiriusGen(6) ---")
    source = SiriusGen(6)
    for node, dests in source.items():
        print(f"Node {node}: -> {dests}")

    print("\n--- SiriusGen(16) ---")
    source16 = SiriusGen(16)
    for node, dests in source16.items():
        print(f"Node {node}: -> {dests}")