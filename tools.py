import math
import numpy as np


def comb(n, k):
    return math.factorial(n) / math.factorial(k) / math.factorial(n - k)


def shifted_irwin_hall(start, end, n, x):
    """
    Let X_1, .., X_n iid with X_1 ~ U[start, end]
    Let Z = sum(i=1, n, X_i)
    Sample the pdf of Z
    """
    if end <= start:
        raise ValueError

    if n == 0:
        raise ValueError

    # Shift so X_i ~ U[0,1]
    x_shift = (x - n * start) / (end - start)

    # Irwin-Hall-pdf (https://en.wikipedia.org/wiki/Irwin–Hall_distribution)
    y = sum([1 / (2 * math.factorial(n - 1)) * (-1)**k * comb(n, k) *
             (x_shift - k)**(n-1) * np.sign(x_shift - k) for k in range(n + 1)])
    y[(x < start * n) | (x > end * n)] = 0

    return y
