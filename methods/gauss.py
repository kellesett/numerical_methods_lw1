import numpy as np
import equations
from math import *
import matplotlib.pyplot as plt

def gauss(input_matrix :np.array, input_b: np.array) -> np.array:
    # Классический метод Гаусса
    matrix = input_matrix.copy()
    b = input_b.copy()
    size = matrix.shape[0]
    for i in range(size):
        if np.abs(matrix[i][i]) < 1.e-6:
            idx_arr = np.where(np.abs(matrix.T[i][i:]) < 1.e-6)
            if not len(idx_arr):
                print("ERROR")
                return
            idx = idx_arr[0][0]
            matrix[idx], matrix[i] = matrix[i], matrix[idx]

        for j in range(i + 1, size):
            b[j] = b[j] - b[i] / matrix[i][i] * matrix[j][i]
            matrix[j] = matrix[j] - matrix[i] / matrix[i][i] * matrix[j][i]

    return solve_right_triangular(matrix, b)

def gauss_modified(input_matrix :np.array, input_b: np.array) -> np.array:
    # Метод Гаусса с выбором главного элемента
    matrix = input_matrix.copy()
    b = input_b.copy()
    size = matrix.shape[0]
    for i in range(size):
        # Выбор строки
        idx = np.abs(matrix[i:, i]).argmax() + i
        matrix[idx], matrix[i] = matrix[i].copy(), matrix[idx].copy()
        b[idx], b[i] = b[i], b[idx]
        # Обниуление оставшихся строк
        for j in range(i + 1, size):
            b[j] = b[j] - b[i] / matrix[i][i] * matrix[j][i]
            matrix[j] = matrix[j] - matrix[i] / matrix[i][i] * matrix[j][i]

    return solve_right_triangular(matrix, b)

def solve_right_triangular(matrix: np.array, b: np.array) -> np.array:
    # Решение СЛАУ для верхнетреугольной матрицы
    length = matrix.shape[0]
    x = np.zeros_like(b, dtype=float)
    for i in range(length - 1, -1, -1):
        x[i] = (b[i] - np.sum(np.dot(matrix[i][i + 1:], x[i + 1:]))) / matrix[i][i]
    return x

def cond(matrix: np.array):
    # Подсчет числа обусловленности
    return np.linalg.cond(matrix)


def decompose_to_lu(matrix: np.array) -> np.array:
    # Разложение матрицу коэффициентов на матрицы L и U.
    # Создаём пустую LU-матрицу
    lu_matrix = np.matrix(np.zeros_like(matrix))
    size = matrix.shape[0]

    for k in range(size):
        # Вычисляем все остаточные элементы k-ой строки
        for j in range(k, size):
            lu_matrix[k, j] = matrix[k, j] - lu_matrix[k, :k] * \
                              lu_matrix[:k, j]
        # Вычисляем все остаточные элементы k-го столбца
        for i in range(k + 1, size):
            lu_matrix[i, k] = (matrix[i, k] - lu_matrix[i, : k] * lu_matrix[: k, k]) / lu_matrix[k, k]

    return lu_matrix

def get_l(matrix: np.array):
    # Получение треугольной матрицы L из представления LU-матрицы
    l = matrix.copy()
    for i in range(l.shape[0]):
            l[i, i] = 1
            l[i, i + 1:] = 0
    return l


def get_u(matrix: np.array):
    # Получение треугольной матрицы U из представления LU-матрицы
    u = matrix.copy()
    for i in range(1, u.shape[0]):
        u[i, :i] = 0
    return u

def determinant(matrix: np.array):
    # Вычисление определителя
    u = get_u(decompose_to_lu(matrix))
    size = u.shape[0]
    det = 1
    for i in range(size):
        det *= u[i, i]
    return det

def inv(matrix: np.array):
    # Подсчет обратной матрицы
    size = matrix.shape[0]
    # Создание единичной матрицы
    identity_matrix = np.identity(size)
    inv_matrix = np.zeros_like(matrix)
    # Использование метода Гаусса
    for i in range(size):
        inv_matrix.T[i] = gauss(matrix, identity_matrix[i])

    return inv_matrix


if __name__ == "__main__":
    a0, b0 = equations.option_4
    m = 30.
    q = 1.001 - 2 * m * 10 ** -3
    fig, ax = plt.subplots()
    start, end = 2, 100

    x = [i for i in range(start, end)]
    y = np.zeros_like(x)
    for n in x:
        a_option5 = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    a_option5[i][j] = (i + 1 + j + 1) / (m + n)
                else:
                    a_option5[i][j] = n + m ** 2 + (j + 1) / m + (i + 1) / n
        det = determinant(a_option5)
        y[n - start] = log(1 / abs(det))
    ax.set_xlabel(r'n', fontsize=12, loc="right")
    ax.set_ylabel(r'f(det(A))', fontsize=12, loc="top", rotation=0)
    ax.plot(x, y, c='green', marker='o')
    y = np.ones_like(x)
    # ax.plot(x, y, c='blue', label='const = 1', linewidth=3)
    ax.set_title('determinant research')
    ax.grid()
    # ax.legend()
    fig.canvas.manager.set_window_title("Result")
    fig.tight_layout()
    plt.show()
