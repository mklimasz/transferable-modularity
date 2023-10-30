import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_max_sum_assignment(cost_matrix: np.ndarray):
    # Higher cost - better.
    # https://en.wikipedia.org/wiki/Hungarian_algorithm
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    row_indices, col_indices = linear_sum_assignment(-cost_matrix)
    assignment = [(row, col) for row, col in zip(row_indices, col_indices)]
    cost = sum([cost_matrix[r, c] for r, c in assignment])
    return cost, assignment


if __name__ == '__main__':
    m = np.array([[3, 4, 4, 4], [1, 3, 4, 4], [3, 2, 3, 4], [4, 4, 4, 4]])
    cost, _ = linear_max_sum_assignment(m)
    assert cost == 16

    m = np.array([[1, 2, 3], [6, 5, 4], [7, 9, 8]])
    cost, _ = linear_max_sum_assignment(m)
    assert cost == 18

    m = np.array([[1, 2, 3], [6, 5, 4], [7, 9, 8], [-1, -1, -1]])
    cost, _ = linear_max_sum_assignment(m)
    assert cost == 18
