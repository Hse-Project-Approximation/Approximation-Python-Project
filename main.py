# from scipy.optimize import curve_fit
import numpy as np
import sympy as sp
from numpy import array, exp, sin, cos, tan, linalg
import matplotlib.pyplot as plt
import csv

# Парсинг координат из CSV файла
x_arr, y_arr = np.array(list()), np.array(list())
with open('shel.csv', 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for line in csv_reader:
        t = ''.join(line).split(';')
        x_arr = np.append(x_arr, t[0]).astype('f')
        y_arr = np.append(y_arr, t[1]).astype('f')


def ed(i):
    return 1


def sq(i):
    return i ** 2


def kub(i):
    return i ** 3


def ob(i):
    return i ** (-1)


def pokv2(i):
    return 2 ** i


def pokn2(i):
    return 2 ** (-i)


def ns(i):
    return i


def log(i):
    return np.log(i + 1)


# funlist = [np.cos, np.sin, ed, sq, ns, pokn2] #улевой минор
funlist = [np.cos, np.sin, ed, sq, ns, log]
x = list()
for i in range(len(funlist)):
    x.append(sp.symbols("C" + str(i + 1)))

su = 0
for i in range(len(y_arr)):
    kv = 0
    for j in range(len(x)):
        kv = kv + x[j] * funlist[j](i + 1)
    su = su + (kv - y_arr[i]) ** 2
print(su, '\n')

di = []
for i in range(len(x)):
    di.append(sp.diff(su, x[i]))
print(di, '\n')
resh = sp.solve(di, x)

mat = np.zeros((len(x), len(x)))
for i in range(len(x)):
    for j in range(len(x)):
        for g in range(len(y_arr)):
            mat[i, j] = mat[i, j] + 2 * funlist[i](y_arr[g]) * funlist[j](y_arr[g])

print(mat, '\n')


# Функция для проверки матрицы на детерминанты нулевых миноров
def check_minors(matrix):
    status = True
    for k in range(len(matrix)):
        lst = [[matrix[i][j] for j in range(k + 1)] for i in range(k + 1)]
        if linalg.det(lst) == 0:
            print('Детерминант минора:', lst, 'равен нулю, решение невозможно')
            status = False
        else:
            continue
    if status:
        print('Нет нулевых миноров, выполняем аппроксимцию\n')
    else:
        exit()


# Проверяем матрицу
check_minors(mat)


# Функция для поиска доверительного интервала
def find_trusted_interval(y_arr):
    m = sum(y_arr) / len(y_arr)
    s = 0
    for i in y_arr:
        s += (i - m) ** 2
    D = s / (len(y_arr) + 1)
    return m + 3 * D ** (1 / 2), m - 3 * D ** (1 / 2)


# Считаем доверительный интервал
print("Доверительный интервал: ", find_trusted_interval(y_arr), '\n')
mmin, mmax = find_trusted_interval(y_arr)

# Задаём интервал графика
graph_range = int(
    input('Введите промежуток (одно число, т.к. начинаем отрисовку с нуля), на котором вы хотите рассмотреть график: '))
x_arr_new = np.array(list())
for i in range(graph_range):
    x_arr_new = np.append(x_arr_new, i).astype('f')

progn = graph_range - len(y_arr)
status = False
results = np.zeros(progn)
for i in range(progn):
    for j in range(len(resh)):
        results[i] = results[i] + resh[x[j]] * funlist[j](i + len(y_arr))
    if (results[i] < mmin or results[i] > mmax):
        status = True
# print(results, '\n')
# print(status, '\n')
#if (status):
    #print("predicted meanings are in normal distribution", '\n')
#else:
    #print("predicted meanings are in normal distribution", '\n')

status = False
preresults = np.zeros(len(y_arr) + progn)
for i in range(len(preresults)):
    for j in range(len(resh)):
        preresults[i] = preresults[i] + resh[x[j]] * funlist[j](i)
    if (preresults[i] < mmin or preresults[i] > mmax):
        status = True
# print(results, '\n')
# print(status, '\n')
#if (status):
    #print("predicted meanings are in normal distribution", '\n')
#else:
    #print("predicted meanings are in normal distribution", '\n')

# Выводим график получившейся функции, с подставленными значениями аргументов и начальные данные
plt.plot(x_arr_new, preresults)
plt.plot(x_arr, y_arr, 'bo')  # чтобы вернуть отрисовку линией, убрать 'bo'
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
