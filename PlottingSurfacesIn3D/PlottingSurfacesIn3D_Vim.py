import numpy as np
from matplotlib import pyplot as pplt
from mpl_toolkits.mplot3d import axes3d


x, y = [0, 1, 2], [3, 4, 5]
X, Y = np.meshgrid(x, y)
for xrow, yrow in zip(X, Y):
  print(xrow, yrow, sep='\t')

#heat maps : plt.pcolormesh()
#contour maps : plt.contour()
#wireframe maps : ax.plot_wireframe()
#surface plots : ax.plot_surface()
#giving color scale : plt.colorbar()

#Create a 2-D domain with np.meshgrid().
x = np.linspace(-np.pi, np.pi, 100)
y = x.copy()
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.sin(Y)       # Calculate g(x,y) = sin(x)sin(y).
# Plot the heat map of f over the 2-D domain.
plt.subplot(131)
plt.pcolormesh(X, Y, Z, cmap="viridis")
plt.colorbar()
plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)
# Plot a contour map of f with 10 level curves.
plt.subplot(132)
plt.contour(X, Y, Z, 10, cmap="coolwarm")
plt.colorbar()
#plot a wireframe map, specifying the strides
fig = plt.figure()
ax = fig.add_subplot(133, projection='3d')
# Plot a basic wireframe.
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
plt.show()

#BEGIN EXERCISE 1

#PART A

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X**2 - Y**2) / (X**2 + Y**2)
plt.pcolormesh(X, Y, Z, cmap="coolwarm")
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)

#PART B

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X * Y**2) - X**3
plt.pcolormesh(X, Y, Z, cmap="magma")
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)

#PART C

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (2*X**2 + 3*X*Y + 4*Y**2) / (3*X**2 + 5*Y**2)
plt.pcolormesh(X, Y, Z, cmap="magma")
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)

#END EXERCISE 1

#BEGIN EXERCISE 2

#PART A

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X**2 - Y**2) / (X**2 + Y**2)
plt.contour(X, Y, Z, 10, cmap="coolwarm")
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)
x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X**2 - Y**2) / (X**2 + Y**2)
plt.contour(X, Y, Z, 10, cmap="coolwarm")
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)

#PART B

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X * Y**2) - X**3
plt.contour(X, Y, Z, 10, cmap="coolwarm")
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)

#PART C

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (2*X**2 + 3*X*Y + 4*Y**2) / (3*X**2 + 5*Y**2)
plt.contour(X, Y, Z, 10, cmap="coolwarm")
plt.colorbar()
plt.xlim(-5,5)
plt.ylim(-5,5)

#END EXERCISE 2

#BEGIN EXERCISE 3

#PART A

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X**2 - Y**2) / (X**2 + Y**2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
plt.show()

#PART B

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X * Y**2) - X**3
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
plt.show()

#PART C

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (2*X**2 + 3*X*Y + 4*Y**2) / (3*X**2 + 5*Y**2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
plt.show()

#END EXERCISE 3

#BEGIN EXERCISE 4

#PART A

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X**2 - Y**2) / (X**2 + Y**2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

#PART B

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (X * Y**2) - X**3
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

#PART C

x = np.linspace(-5, 5, 100)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = (2*X**2 + 3*X*Y + 4*Y**2) / (3*X**2 + 5*Y**2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

#END EXERCISE 4

#BEGIN EXERCISE 5

def FourierSquareWave(y,x):
  x = np.array(x)
  y = np.array(y)
  x = x[0]
  y = np.transpose(y)[0]
  Z = np.empty((np.shape(y)[0],0))
  print(x)
  print(y)
  for i in x:  
    C = np.empty((0,1))
    S = 0
    k = np.shape(y)[0]-1
    for n in range(k+1):
      S += (((-1)**n)*np.cos(((2*n+1)*np.pi*(i-1))/2))/(2*n+1)
      T = (4/np.pi)*S+2
      T = np.array([[T]])
      C = np.concatenate((C,T), axis=0)
    Z = np.concatenate((Z,C), axis=1)
  return Z

x = np.linspace(0,4,1001)
y = np.linspace(0,25,26, dtype=int)
X,Y = np.meshgrid(x,y)
Z = FourierSquareWave(Y,X)
#This is the graph of the first 25 Fourier approximations of the
#discontinuous piece-wise function, 3 : [0,2), 2 : 2, 1 : (2,4]

plt.pcolormesh(X, Y, Z, cmap='magma')
plt.colorbar()
plt.xlim(0,4)
plt.ylim(0,25)
plt.show()

plt.contour(X, Y, Z, 5, cmap="coolwarm")
plt.colorbar()
plt.xlim(0,4)
plt.ylim(0,25)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=50)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
my_col = cm.jet(Z/np.amax(Z))
ax.plot_surface(X, Y, Z, facecolors=my_col)
plt.show()

#END EXERCISE 5

#BEGIN EXERCISE 6

x = np.linspace(0,2851,10000)
y = x.copy()
X,Y = np.meshgrid(x,y)
Z = ( 5000-0.005*(X**2+Y**2+X*Y)+12.5*(X+Y) ) * np.exp( -np.abs(0.000001*(X**2+Y**2)-0.0015*(X+Y)+0.7) )

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
plt.show()

fig = plt.figure()
ax = fig.gca(projection='3d')
my_col = cm.jet(Z/np.amax(Z))
ax.plot_surface(X, Y, Z, facecolors=my_col)
plt.show()

#END EXERCISE 6
