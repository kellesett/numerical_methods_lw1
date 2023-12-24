import numpy as np
import methods
import equations

PRECISION = 10

def read_equation(file_name: str):
    with open(file_name, 'r', encoding="utf-8") as file:
        equation = np.array([list(map(float, line.split())) for line in file])

    return equation.T[:-1].T, equation.T[-1]

if __name__ == "__main__":
    mode = input("Введите (yes) для ввода из файла, (Enter) для подсчета по умолчанию: ")
    if mode == "yes":
        name = input("Введите имя файла, для выбора по умолчанию (Enter): ")
        a, b = read_equation("equations/test.txt")
    else:
        a, b = equations.option8_ex3

    system = '\\\\\n'.join([' + '.join(["{}x_{{{}}}".format(int(a[i, j]), j + 1) if a[i, j] else "" for j in range(a.shape[1])]) + " = {}".format(int(b[i])) for i in range(a.shape[0])])
    print(system)
    print('-' * 10)
    system = '\n'.join([' '.join(
        ["{}".format(int(a[i, j])) for j in range(a.shape[1])]) + " {}".format(
        int(b[i])) for i in range(a.shape[0])])
    print(system)
    print('-' * 10)
    det = methods.determinant(a)
    print("DETERMINANT ({}): {}\n".format(np.isclose(det, np.linalg.det(a)), det) + "-" * 10)
    if np.isclose(det, 0):
        print("Выроженная система, единственное решение отсутствует")
        # exit(0)

    print("CONDITION NUMBER: {:.10f}\n".format(methods.cond(a)) + "-" * 10)
    ans = methods.gauss(a, b)
    print("GAUSS ANS:\n{}\n".format(ans) + "-" * 10)
    format_ans = ', '.join(["$x_{{{}}} = {}$".format(i + 1, ans[i]) for i in range(len(ans))])
    print("$$" + format_ans + "$$")

    ans = methods.gauss_modified(a, b)
    print("MODIFIED GAUSS ANS:\n{}\n".format(ans) + "-" * 10)
    format_ans = ', '.join(["$x_{{{}}} = {}$".format(i + 1, ans[i]) for i in range(len(ans))])
    print("$$" + format_ans + "$$")

    inv = methods.inv(a)
    print("INV MATRIX ({}):\n{}\n".format(
        np.allclose(inv, np.linalg.inv(a)),
        inv.round(4)) + "-" * 10)
    inv = inv.round(4)
    format_ans = '\\\\\n'.join([' & '.join(["{}".format(inv[i, j]) for j in range(inv.shape[1])]) for i in range(inv.shape[0])])
    print(format_ans)

    package = methods.relax(a, b, 1, 2500)
    if package is not None:
        ans, iterations = package
        ans = ans.round(PRECISION)
        print("RELAX ANS:\n{}\nITERATIONS: {}\n".format(ans, iterations) + "-" * 10)
        format_ans = ', '.join(["x_{{{}}} = {}".format(i + 1, ans[i]) for i in range(len(ans))])
        print("$$" + format_ans + "$$")
