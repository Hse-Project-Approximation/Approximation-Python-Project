from sympy import diff, symbols, cos, sin, Function, Derivative, solve, Matrix, plot, expand
import matplotlib.pyplot as plt
import numpy as np
import csv

#read coordinates
x_arr, y_arr = np.array(list()), np.array(list())
with open('coord.csv', 'r') as csv_file:
  csv_reader = csv.reader(csv_file)
  for line in csv_reader:
    t = ''.join(line).split(';')
    x_arr = np.append(x_arr, t[0])
    y_arr = np.append(y_arr, t[1])

#show the raw graphics
#plt.plot(x_arr, y_arr)
#plt.show()

#creating necessary symbols and functions
n = 6
x, y = symbols('x y')

symb_arr = list()
func_arr = list()
for i in range((n-1)*n + 1):
  symb_arr.append(symbols('c' + str(i)))

for i in range(1, n+1):
  func_arr.append(sin(x**(i+1)))
  func_arr.append(sin(i*x))
  func_arr.append(cos(x**(i+1)))
  func_arr.append(cos(i*x))
  #func_arr.append(1/x)
  func_arr.append(x**i)
func_arr.append(1)

#creating fucntions
r_func = 1
for i in range(len(symb_arr)):
  r_func += symb_arr[i]*func_arr[i]
i_func = expand((y - r_func)**2)
print(i_func)

#solving df
c_df_arr = list()
for i in range(len(symb_arr)):
  c_df_arr.append(diff(r_func, symb_arr[i]))
print(c_df_arr)
slu = solve(c_df_arr, symb_arr, dict=True, check=False)
print(slu)

'''
#checking if koefs are correct
det_min_arr = list()
matrx = None
for i in range(n):
  t_arr = list()
  for j in range(n):
    t_arr.append(diff(c_df_arr[i], symb_arr[j]))
  matrx.append(t_arr)
for z in range(n):
  temp_matrx = list()
  for i in range(n-1, -1, -1):
    t_arr = list()
    for j in range(n-1, -1, -1):
      t_arr.append(diff(c_df_arr[i-z], symb_arr[j]))
    temp_matrx.append(t_arr)[::-1]
  det_min_arr.append(Matrix(temp_matrx).det())
res = 0
t_arr = list()
for j in range(len(slu)):
  t_arr.append(slu[symb_arr[i]])
for i in range(len(x_arr)):
  t1_func = i_func(y_arr[i])
  t2_func = t1_func(t_arr)
  t_res = t2_func(x_arr[i])
  if t_res < 0:
    print('ERROR!')
    break
  else:
    res += t_res
#show the graphics
plot(r_func(t_arr))
'''