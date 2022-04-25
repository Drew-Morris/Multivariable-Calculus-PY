import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

#BEGIN EXERCISE 1

x,y = sp.symbols('x,y')
F_X = lambda x,y : y**2 - 2*x*y
F_Y = lambda x,y : 3*x*y - 6*x**2
x_tail_axis = np.linspace(-5, 5, 11)
y_tail_axis = np.linspace(-5, 5, 11)
x_tail_grid,y_tail_grid = np.meshgrid(x_tail_axis,y_tail_axis)
x_tip_grid = F_X(x_tail_grid, y_tail_grid)
y_tip_grid = F_Y(x_tail_grid, y_tail_grid)

fig, ax = plt.subplots()
ax.quiver(x_tail_grid, y_tail_grid, x_tip_grid, y_tip_grid)
ax.set_xlim(-6,6)
ax.set_ylim(-9,6)
ax.set_title("F(x,y) = <y**2 - 2*x*y, 3*x*y - 6*x**2>")
fig.show()

#END EXERCISE 1

#BEGIN EXERCISE 2

x,y = sp.symbols('x,y')
r = lambda x,y : (x**2 + y**2)**0.5
F_X = lambda x,y : (r(x,y)**2 - 2*r(x,y))*x
F_Y = lambda x,y : (r(x,y)**2 - 2*r(x,y))*y
x_tail_axis = np.linspace(-1, 1, 21)
y_tail_axis = np.linspace(-1, 1, 21)
x_tail_grid,y_tail_grid = np.meshgrid(x_tail_axis,y_tail_axis)
x_tip_grid = F_X(x_tail_grid, y_tail_grid)
y_tip_grid = F_Y(x_tail_grid, y_tail_grid)

fig, ax = plt.subplots()
ax.quiver(x_tail_grid, y_tail_grid, x_tip_grid, y_tip_grid)
ax.set_xlim(-1.1,1.1)
ax.set_ylim(-1.1,1.1)
ax.set_title("F(x,y) = (r**2 - 2*r) * <x,y>")
fig.suptitle("r = (x**2 + y**2)**0.5")
fig.show()

x_tail_axis = np.linspace(-5, 5, 21)
y_tail_axis = np.linspace(-5, 5, 21)
x_tail_grid,y_tail_grid = np.meshgrid(x_tail_axis,y_tail_axis)
x_tip_grid = F_X(x_tail_grid, y_tail_grid)
y_tip_grid = F_Y(x_tail_grid, y_tail_grid)

fig, ax = plt.subplots()
ax.quiver(x_tail_grid, y_tail_grid, x_tip_grid, y_tip_grid)
ax.set_xlim(-6.5,6.5)
ax.set_ylim(-7,7)
ax.set_title("F(x,y) = (r**2 - 2*r) * <x,y>")
fig.suptitle("r = (x**2 + y**2)**0.5")
fig.show()

#END EXERCISE 2

#BEGIN EXERCISE 3

X,Y = sp.symbols('X,Y')

#Part A

x,y = sp.symbols('x,y')
f = lambda x,y : np.log(1 + x**2 + 2*y**2)
fdx = lambda x,y : (2*x) / (1 + x**2 + 2*y**2)
fdy = lambda x,y : (4*y) / (1 + x**2 + 2*y**2)
xaxis = np.linspace(-5,5,11)
yaxis = np.linspace(-5,5,11)
xgrid,ygrid = np.meshgrid(xaxis,yaxis)
dx_tail_axis = np.linspace(-5, 5, 11)
dy_tail_axis = np.linspace(-5, 5, 11)
dx_tail_grid,dy_tail_grid = np.meshgrid(dx_tail_axis,dy_tail_axis)
zgrid = np.array(f(xgrid,ygrid))
dx_tip_grid = fdx(dx_tail_grid,dy_tail_grid)
dy_tip_grid = fdy(dx_tail_grid,dy_tail_grid)
plt.contourf(xaxis,yaxis,zgrid, levels=10, cmap = "magma")
plt.quiver(dx_tail_grid,dy_tail_grid,dx_tip_grid,dy_tip_grid)
plt.show()

#Part B

x,y = sp.symbols('x,y')
f = lambda x,y : np.cos(x) - 2*np.sin(y)
fdx = lambda x,y : -1*np.sin(x)
fdy = lambda x,y : -2*np.cos(y)
xaxis = np.linspace(-5,5,11)
yaxis = np.linspace(-5,5,11)
xgrid,ygrid = np.meshgrid(xaxis,yaxis)
dx_tail_axis = np.linspace(-5, 5, 11)
dy_tail_axis = np.linspace(-5, 5, 11)
dx_tail_grid,dy_tail_grid = np.meshgrid(dx_tail_axis,dy_tail_axis)
zgrid = np.array(f(xgrid,ygrid))
dx_tip_grid = fdx(dx_tail_grid,dy_tail_grid)
dy_tip_grid = fdy(dx_tail_grid,dy_tail_grid)
plt.contourf(xaxis,yaxis,zgrid, levels=10, cmap = "magma")
plt.quiver(dx_tail_grid,dy_tail_grid,dx_tip_grid,dy_tip_grid)
plt.show()

#END EXERCISE 3

#BEGIN EXERCISE 4

#Part A

x,y,z = sp.symbols('x,y,z')
Fx = lambda x,y,z : 1
Fy = lambda x,y,z : 2
Fz = lambda x,y,z : 3
xaxis = np.linspace(-5,5,11)
yaxis = np.linspace(-5,5,11)
zaxis = np.linspace(-5,5,11)
xtail,ytail,ztail = np.meshgrid(xaxis,yaxis,zaxis)
xtip = Fx(xtail,ytail,ztail)
ytip = Fy(xtail,ytail,ztail)
ztip = Fz(xtail,ytail,ztail)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.quiver(xtail,ytail,ztail,xtip,ytip,ztip)
plt.show()

#Part B

x,y,z = sp.symbols('x,y,z')
Fx = lambda x,y,z : 1
Fy = lambda x,y,z : 2
Fz = lambda x,y,z : z
xaxis = np.linspace(-5,5,11)
yaxis = np.linspace(-5,5,11)
zaxis = np.linspace(-5,5,11)
xtail,ytail,ztail = np.meshgrid(xaxis,yaxis,zaxis)
xtip = Fx(xtail,ytail,ztail)
ytip = Fy(xtail,ytail,ztail)
ztip = Fz(xtail,ytail,ztail)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.quiver(xtail,ytail,ztail,xtip,ytip,ztip)
plt.show()

#Part C

x,y,z = sp.symbols('x,y,z')
Fx = lambda x,y,z : x
Fy = lambda x,y,z : y
Fz = lambda x,y,z : z
xaxis = np.linspace(-5,5,11)
yaxis = np.linspace(-5,5,11)
zaxis = np.linspace(-5,5,11)
xtail,ytail,ztail = np.meshgrid(xaxis,yaxis,zaxis)
xtip = Fx(xtail,ytail,ztail)
ytip = Fy(xtail,ytail,ztail)
ztip = Fz(xtail,ytail,ztail)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.quiver(xtail,ytail,ztail,xtip,ytip,ztip)
plt.show()

#Part D

x,y,z = sp.symbols('x,y,z')
Fx = lambda x,y,z : np.sin(y)
Fy = lambda x,y,z : x*np.cos(y) + np.cos(z)
Fz = lambda x,y,z : -1*y*np.sin(z)
xaxis = np.linspace(-5,5,11)
yaxis = np.linspace(-5,5,11)
zaxis = np.linspace(-5,5,11)
xtail,ytail,ztail = np.meshgrid(xaxis,yaxis,zaxis)
xtip = Fx(xtail,ytail,ztail)
ytip = Fy(xtail,ytail,ztail)
ztip = Fz(xtail,ytail,ztail)
fig = plt.figure()
ax = fig.gca(projection="3d")
ax.quiver(xtail,ytail,ztail,xtip,ytip,ztip)
plt.show()

#END EXERCISE 4
