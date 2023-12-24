import numpy as np
import equations

def relax(input_matrix :np.array, input_b: np.array, omiga: float, max_iter: int) -> np.array:
    # Реализация метода верхней релаксации
    matrix = input_matrix.copy()
    b = input_b.copy()
    x = np.ones_like(b)
    size = matrix.shape[0]

    # Преобразования матрицы
    for i in range(size):
        idx = np.argmax(np.abs(matrix[i:, i])) + i
        tmp = matrix[idx].copy()
        matrix[idx] = matrix[i].copy()
        matrix[i] = tmp

    err = 100.
    iterations = 0
    # Установка критериев останова
    while err > 1.e-6 and iterations < max_iter:
        # Вычисление новой итерации
        for i in range(size):
            if matrix[i, i] == 0:
                print("ERROR")
                return None
            x[i] = omiga * (b[i] - np.dot(matrix[i], x)) / matrix[i,i] + x[i]
        iterations += 1
        # Пересчет ошибки
        err = norm(matrix, b, x)
    return x, iterations


def norm(a, b, x):
    # Подсчет второй нормы разности b - Ax
    norm_value = np.linalg.norm(np.dot(a, x) - b, ord=2)
    return norm_value


def read_equation(file_name: str):
    with open(file_name, 'r', encoding="utf-8") as file:
        equation = np.array([list(map(float, line.split())) for line in file])

    return equation.T[:-1].T, equation.T[-1]


if __name__ == "__main__":
    a, b = read_equation("../equations/test.txt")

    system = '\\\\\n'.join([' + '.join(
        ["{}x_{{{}}}".format(int(a[i, j]), j + 1) if a[i, j] else "" for j in range(a.shape[1])]) + " = {}".format(
        int(b[i])) for i in range(a.shape[0])])
    print(system)
    print('-' * 10)
    system = '\n'.join([' '.join(
        ["{}".format(int(a[i, j])) for j in range(a.shape[1])]) + " {}".format(
        int(b[i])) for i in range(a.shape[0])])
    print(system)
    print('-' * 10)
    package = relax(a, b, 0.2, 2500)
    if package is not None:
        ans, iterations = package
        ans = ans.round(10)
        print("RELAX ANS:\n{}\nITERATIONS: {}\n".format(ans, iterations) + "-" * 10)
        format_ans = ', '.join(["x_{{{}}} = {}".format(i + 1, ans[i]) for i in range(len(ans))])
        print('$$' + format_ans + '$$')
