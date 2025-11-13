def ends(D, W):
    """
    Multiplies two matrices D and W (represented as nested lists).

    Args:
        D (list[list[float]]): Left matrix of size (m x n)
        W (list[list[float]]): Right matrix of size (n x p)

    Returns:
        list[list[float]]: The resulting matrix of size (m x p)
    """
    WT = list(zip(*W))
    new = []
    for row in D:
        new_row = []
        for col in WT:
            value = sum(row[i] * col[i] for i in range(len(col)))
            new_row.append(value)
        new.append(new_row)
    return new