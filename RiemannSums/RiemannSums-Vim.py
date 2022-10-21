import numpy as np
import sympy as sp
import math
from matplotlib import pyplot as plt

#BEGIN EXERCISE 1

def riemann_sum_2D(f, xMin, xMax, yMin, yMax, N, method):
  delx = (xMax - xMin) / N #change in x
  dely = (yMax - yMin) / N #change in y
  sum = 0
  if method == "left":
    xstart = 0
    xstop = 2*N
    ystart = 0
    ystop = 2*N
  elif method == "right":
    xstart = 2
    xstop = 2*(N+1)
    ystart = 2
    ystop  = 2*(N+1)
  elif method == "mid":
    xstart = 1
    xstop = 2*N + 1
    ystart = 1
    ystop = 2*N + 1
  else:
    raise ValueError("Method must equal left, right, or mid")
  for j in range(ystart, ystop, 2):
    for i in range(xstart, xstop, 2):
      sum += f(xMin + (delx*i*0.5), yMin + (dely*j*0.5)) 
  return delx*dely*sum
#END EXERCISE 1

#BEGIN EXERCISE 2

#Part A
f2A = lambda x,y : x * math.sin(x*y) 
riemann_sum_2D(f2A, 0, math.pi, 0, math.pi, 10, "mid")
riemann_sum_2D(f2A, 0, math.pi, 0, math.pi, 20, "mid")

#Part B
f2B = lambda x,y : y**2 * math.e**(-x-y)
riemann_sum_2D(f2B, 0, 1, 0, 1, 10, "mid")
riemann_sum_2D(f2B, 0, 1, 0, 1, 20, "mid")

#Part C
f2C = lambda x,y : x**3 * y**2 + x*y
riemann_sum_2D(f2C, 0, 1, 1, 2, 10, "mid")
riemann_sum_2D(f2C, 0, 1, 1, 2, 20, "mid")

#END EXERCISE 2

#BEGIN EXERCISE 3

f3 = lambda x,y : x * math.sin(x+y)
fdf = lambda x,y : x*math.sin(x) - x*math.sin(x+y) - math.cos(x+y) + math.cos(x) + math.cos(y) - 1
AreaEval = fdf(math.pi/6, math.pi/3)
Nrange = np.linspace(1, 100, 100) #input N range
ErrorCalc = lambda x : AreaEval - x
Arange = [] #approximation range
for i in range(len(Nrange)):
  Arange += [riemann_sum_2D(f3, 0, math.pi/6, 0, math.pi/3, int(Nrange[i]), "mid")]
Erange = []
for i in range(len(Nrange)): 
  Erange += [ErrorCalc(Arange[i])] #error range
CvalRange = np.full((100,),AreaEval) #correct value range
plt.plot(Nrange, CvalRange, label="Actual Value")
plt.plot(Nrange, Arange, label="Approximation of N sections")
plt.plot(Nrange, Erange, label="Approximation Error")
plt.legend()
plt.show()

#END EXERCISE 3

#BEGIN EXERCISE 4

def riemann_sum_3D(f, xMin, xMax, yMin, yMax, zMin, zMax, N, method):
  delx = (xMin + xMax) / N #change in x
  dely = (yMin + yMax) / N #change in y
  delz = (zMin + zMax) / N #change in z
  sum = 0
  if method == "left":
    start = 0
    stop = 2*N
  elif method == "right":
    start = 2
    stop = 2*(N+1)
  elif method == "mid":
    start = 1
    stop = 2*N + 1
  else:
    raise ValueError("Method must equal left, right, or mid")
  for k in range(start, stop, 2):
    for j in range(start, stop, 2):
      for i in range(start, stop, 2):
        sum += f(xMin + (0.5*delx*i), yMin + (0.5*dely*j), zMin + (0.5*delz*k))
  return delx*dely*delz*sum

#END EXERCISE 4

#BEGIN EXERCISE 5

f5 = lambda x,y,z : x*y + z**2
riemann_sum_3D(f5, 0, 2, 0, 1, 0, 3, 10, mid)
riemann_sum_3D(f5, 0, 2, 0, 1, 0, 3, 20, mid)

#END EXERCISE 5
