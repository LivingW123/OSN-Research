def base10_conversion(base, tup, length):
    """
    Converts a tuple of digits (in a given base) to its base-10 integer representation.

    Args:
        base (int): The number base (e.g., 2 for binary, 3 for ternary).
        tup (tuple): A tuple of digits in the given base.
        length (int): The number of digits in the tuple.

    Returns:
        int: The equivalent base-10 integer.
    """
    ans = 0
    for digit in range(length):    
        ans += (base ** digit) * tup[length - digit - 1]
    return ans