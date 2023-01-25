from sympy import diff, symbols, cos, sin, Function, Derivative, solve, Matrix, plot
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

#creating necessary functions
c1, c2, x, y = symbols('c1 c2 x y')
r_func = c1*cos(x) + c2*sin(x)
i_func = y - r_func

#solving df
c1_df = diff(r_func, c1)
print(c1_df)
c2_df = diff(r_func, c2)
print(c2_df)
slu = solve([c1_df, c2_df], c1, c2, dict=True)
print(slu)

#checking if correct
det_min2 = Matrix([[diff(c1_df, c1), diff(c1_df, c2)], [diff(c2_df, c1), diff(c2_df, c2)]]).det()
det_min1 = diff(c1_df, c1)

print(det_min1, det_min2)

res = 0
for i in range(len(x_arr)):
  t = i_func(slu[c1], slu[c2], x_arr[i], y_arr[i])
  if t < 0:
    print('ERROR!')
    break
  else:
    res += t

#show the graphics
plot(r_func(slu['c1'], slu['c2']))

