import matplotlib
import sympy
import matplotlib.pyplot as plt
import numpy as np
#Графики
x=np.array([1,4,7,8,10,13])
y=np.array([0,1,2,3,7,10])

plt.plot(x,y)
#plt.show()

#Производные
from sympy import diff, symbols, cos, sin
x, y = symbols('x y')
df1=diff(cos(x),x,2)
df2=diff(cos(x) + sin(y), x,5)
df3=diff(cos(x) + sin(y), y,3)

print(df1)
print(df2)
print(df3)

#Матрицы
from sympy.matrices import Matrix
mtx=Matrix([[1,2,3],[4,5,6],[7,8,20]])
print(mtx)
print(mtx.det())

from sympy.matrices import eye
print(eye(3))

#Линейные уравнения
from sympy import *
a, b,y = symbols('a b y')
a = Matrix([[1, -1], [1, 1]])
b = Matrix([4, 1])

print(linsolve([a, b], y))