from scipy.optimize import curve_fit
import numpy as np
from numpy import array, exp, sin, cos, tan, log, linalg
import matplotlib.pyplot as plt
import csv

# Парсинг координат из CSV файла
x_arr, y_arr = np.array(list()), np.array(list())
FileStatus = True
try:
    with open('shel.csv', 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            t = ''.join(line).split(';')
            x_arr = np.append(x_arr, t[0]).astype('f')
            y_arr = np.append(y_arr, t[1]).astype('f')

except FileNotFoundError:
    print('Ошибка при открытии файла')
    FileStatus = False
    exit()

except:
    print('Ошибка при работе с файлом')
    FileStatus = False
    exit()

finally:
    if FileStatus:
        print('Файл закрыт после чтения:', csv_file.closed, '\n')
    else:
        exit()


# Функция для нахождения аргументов при помощи curve_fit
def function(values_x, x1, x2, x3, x4, x5, x6):
    return x1 * np.cos(values_x) + x2 * np.sin(values_x) + x3 * 1 + x4 * (
            values_x ** 2) + x5 * values_x + x6 * np.log(values_x + 1)


# Используем curve_fit, а выходже получаем значения аргументов и ковариационную матрицу в виде двумерного списка
args, covar = curve_fit(function, x_arr, y_arr)
print("Аргументы: ", args, '\n')
print("Ковариационная матрица: \n", covar, '\n')


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
check_minors(covar)


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

# Получаем значения аргументов
arguments = list(map(float, args))

# Задаём интервал графика
graph_range = int(
    input('Введите промежуток (одно число, т.к. начинаем отрисовку с нуля), на котором вы хотите рассмотреть график: '))
x_arr_new = np.array(list())
for i in range(graph_range):
    x_arr_new = np.append(x_arr_new, i).astype('f')

# Выводим график получившейся функции, с подставленными значениями аргументов и начальные данные
y_fit = function(x_arr_new, *args)
plt.plot(x_arr_new, y_fit)
plt.plot(x_arr, y_arr, 'bo')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()
