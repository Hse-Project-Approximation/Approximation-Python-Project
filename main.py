from scipy.optimize import curve_fit
import numpy as np
from numpy import array, exp, sin, cos, tan, log, linalg
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog
from contextlib import redirect_stdout


class TextWrapper:
    text_field: tk.Text

    def __init__(self, text_field: tk.Text):
        self.text_field = text_field

    def write(self, text: str):
        self.text_field.insert(tk.END, text)

    def flush(self):
        self.text_field.update()


window = tk.Tk()
window.geometry("700x500")
window.title("Approximation app")
text = tk.Text(window, height=500, width=500)
text.pack()


def openFile():
    filepath = filedialog.askopenfilename(initialdir="C:\PythonProjects\SciPy")

    # Парсинг координат из CSV файла
    x_arr, y_arr = np.array(list()), np.array(list())
    FileStatus = True
    try:
        with open(filepath, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for line in csv_reader:
                t = ''.join(line).split(';')
                x_arr = np.append(x_arr, t[0]).astype('f')
                y_arr = np.append(y_arr, t[1]).astype('f')

    except FileNotFoundError:
        with redirect_stdout(TextWrapper(text)):
            print('Ошибка при открытии файла')
        FileStatus = False
        exit()

    except:
        with redirect_stdout(TextWrapper(text)):
            print('Ошибка при работе с файлом')
        FileStatus = False
        exit()

    finally:
        if FileStatus:
            with redirect_stdout(TextWrapper(text)):
                print('Файл закрыт после чтения:', csv_file.closed, '\n')
        else:
            exit()

    # Функция для нахождения аргументов при помощи curve_fit
    def function(values_x, x1, x2, x3, x4, x5, x6):  # укажите необходимое количество аргументов
        return x1 * np.cos(values_x) + x2 * np.sin(values_x) + x3 * 1 + x4 * (  # задайте функцию
                values_x ** 2) + x5 * values_x + x6 * np.log(values_x + 1)

    # Используем curve_fit, а выходже получаем значения аргументов и ковариационную матрицу в виде двумерного списка
    args, covar = curve_fit(function, x_arr, y_arr)
    with redirect_stdout(TextWrapper(text)):
        print("Аргументы: ", args, '\n')
    with redirect_stdout(TextWrapper(text)):
        print("Ковариационная матрица: \n", covar, '\n')

    # Функция для проверки матрицы на детерминанты нулевых миноров
    def check_minors(matrix):
        status = True
        for k in range(len(matrix)):
            lst = [[matrix[i][j] for j in range(k + 1)] for i in range(k + 1)]
            if linalg.det(lst) == 0:
                with redirect_stdout(TextWrapper(text)):
                    print('Детерминант минора:', lst, 'равен нулю, решение невозможно')
                status = False
            else:
                continue
        if status:
            with redirect_stdout(TextWrapper(text)):
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
    with redirect_stdout(TextWrapper(text)):
        print("Доверительный интервал: ", find_trusted_interval(y_arr), '\n')

    # Получаем значения аргументов
    arguments = list(map(float, args))

    # Function to read and print the value in Entry widget
    def print_graph():
        graph_range = int(entry.get())
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

    label = tk.Label(window, text="Введите длину графика:")
    label.place(x=300, y=400)

    # Поле ввода длины графика
    entry = tk.Entry(window)
    entry.place(x=300, y=430)

    # Кнопка "Ввод"
    button = tk.Button(window, text="Ввод", command=print_graph)
    button.place(x=300, y=455)

# Кнопка "Открыть файл"
button = tk.Button(text="Открыть файл", command=openFile)
button.place(x=10, y=450)

window.mainloop()
