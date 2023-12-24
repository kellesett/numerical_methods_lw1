import numpy as np
from math import *

a_option8_ex1 = np.array([[2, -1, -6, 3],
                          [1, 3, -6, 2],
                          [3, -2, 2, -2],
                          [2, -1, 2, 0]], dtype=float)

b_option8_ex1 = np.array([-1, 3, 8, 4], dtype=float)

option8_ex1 = (a_option8_ex1, b_option8_ex1)

a_option8_ex2 = np.array([[1, 1, -3, 1],
                          [2, 1, -2, -1],
                          [1, 1, 1, -1],
                          [2, 4, -6, 14]], dtype=float)

b_option8_ex2 = np.array([-1, 1, 3, 12], dtype=float)

option8_ex2 = (a_option8_ex2, b_option8_ex2)

a_option8_ex3 = np.array([[1, 4, 5, 2],
                          [2, 9, 8, 3],
                          [3, 7, 7, 0],
                          [5, 7, 9, 2]], dtype=float)

b_option8_ex3 = np.array([2, 7, 12, 20], dtype=float)

option8_ex3 = (a_option8_ex3, b_option8_ex3)

n = 30
m = 20.

a_option5 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            a_option5[i][j] = (i + 1 + j + 1) / (m + n)
        else:
            a_option5[i][j] = n + m ** 2 + (j + 1) / m + (i + 1) / n

b_option5 = np.zeros((n, ))
for i in range(n):
    b_option5[i] = m * (i + 1) + n

option_5 = (a_option5, b_option5)

n = 70
m = 3.
q = 1.001 - 2 * m * 10 ** -3

a_option4 = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            a_option4[i][j] = q ** (i + 1 + j + 1) + 0.1 * (j - i)
        else:
            a_option4[i][j] = (q - 1) ** (i + j)

b_option4 = np.zeros((n, ))
for i in range(n):
    b_option4[i] = n * exp(m / (i + 1)) * cos(m)

option_4 = (a_option4, b_option4)
