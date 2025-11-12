import itertools

from mathfunc import(
    base10_conversion
)

def RR1(node):
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
    temp = []
    for i in range(node):
        temp.append([x for x in range(i)] + [y for y in range(i, node)])
    return temp


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
    return mat


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
    return mat


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