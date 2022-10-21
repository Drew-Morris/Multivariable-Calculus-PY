import numpy as np
import sympy as sp
import random
import scipy.optimize as opt

#BEGIN EXERCISE 1

def newton(fp, fpp, x0, tol, maxiter):
    #Problem 1 on http://math.byu.edu/~nick/gradient-descent
    #Implement Newton's method.
  for i in range(maxiter):
    xn = x0 - (fp(x0) / fpp(x0))
    if abs(fp(xn) - fp(x0)) < tol:
      return (xn, True, i) 
    else:
      x0 = xn
  return (x0, False, i)

#END EXERCISE 1

#BEGIN EXERCISE 2

x = sp.symbols('x')
f = x**2 + sp.sin(5*x)
fp = sp.diff(f, x)
fpp = sp.diff(fp, x)
newton(fp, fpp, 0, 10**(-10), 500)

#END EXERCISE 2

#BEGIN EXERCISE 3

def descent(f, df, x0, tol = 1e-5, maxiter = 100):
  #Problem 3 on http://math.byu.edu/~nick/gradient-descent
  #Implement the method of steepest descent
  x = sp.symbols('x')
  for i in range(maxiter):
    g = lambda a : f(x0 - a*df(x0))
    a_n = opt.minimize_scalar(g).x
    h = lambda x : x - a_n*df(x)
    xn = h(x0)
    if sum(df(xn)[i]**2 for i in range(len(xn)))**(1/2) < tol:
      return (xn, True, i)
    else:
      x0 = xn
  return (x0, False, i)
#END EXERCISE 3

#BEGIN EXERCISE 4

#Part A 

#Call your descent function on 
f = lambda x : x[0]**4 + x[1]**4 + x[2]**4
x0 = [1,1,1]
tol = 1e-5
maxiter = 100
df = lambda x : np.array([4*x[0]**3, 4*x[1]**3, 4*x[2]**3])
descent(f, df, x0, tol, maxiter)

#Part B

#Then call it on the Rosenbrock function.
rosen = lambda x : (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2
drosen = lambda x : np.array([-2*(1 - x[0]) + -200*2*x[0]*(x[1] - x[0]**2), 200*(x[1] - x[0]**2)])
x0 = np.array([0,0])
