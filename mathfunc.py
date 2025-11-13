import math

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

def find_closest_factors_positive(n):
    """
    Finds two positive integer factors of n (a * b = n) that are
    closest to each other in value.

    Assumes n is a positive integer (n > 0).
    Returns a tuple (factor1, factor2) where factor1 <= factor2.
    """
    start = int(math.sqrt(n))
    for i in range(start, 0, -1):
        if n % i == 0:
            return (i, n // i)
    return (1, n)