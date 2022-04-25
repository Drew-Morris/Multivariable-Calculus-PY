import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation, rc
from IPython.display import HTML
#impot matplotlib.animation as animation

%matplotlib inline

t = np.linspace(-5, 5, 50)
#creates values for parameter, t
x = np.sin(t)
y = np.cos(t)
z = t
#finds functional values based on t
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-1.2,1.2])
ax.set_ylim3d([-1.2,1.2])
ax.set_zlim3d([-5.2,5.2])
#creates figure, axes, and axis limits
ax.plot(x,y,z)
plt.title("Plot of $< \sin(t), \cos(t), t >$")
plt.show()
#Plots the curve

t = np.linspace(0, 2*np.pi, 500)
x = (4+np.sin(20*t))*np.cos(t)
y = (4+np.sin(20*t))*np.sin(t)
z = np.cos(20*t)
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-5,5])
ax.set_ylim3d([-5,5])
ax.set_zlim3d([-1.2,1.2])
#
ax.plot(x,y,z)
#
ax.view_init(elev=30)
plt.title("Torroidal Spiral")
plt.show()

#BEGIN EXERCISE 1

#Changing the elevation changes the angle at which you are viewing the 3d model

#END EXERCISE 1

#BEGIN EXERCISE 2

t = np.linspace(0, 2*np.pi, 500)
x = (np.cos(t)*np.sin(2*t))
y = (np.sin(t)*np.sin(2*t))
z = (np.cos(2*t))
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-1,1])
ax.set_ylim3d([-1,1])
ax.set_zlim3d([-1,1])
#
ax.plot(x,y,z)
#
ax.view_init(elev=30)
plt.title("4 Petaled Sphere")
plt.show()

#END EXERCISE 2

#BEGIN EXERCISE 3

t = np.linspace(-5,5,500)
x = t*math.e**t
y = math.e**(-t)
z = t
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-5,750])
ax.set_ylim3d([-5,150])
ax.set_zlim3d([-5,5])
#
ax.plot(x,y,z)
#
ax.view_init(elev=45)
plt.title("Exp: Decay on Y and Growth on X")
plt.show()

#END EXERCISE 3

#BEGIN EXERCISE 4

t = np.linspace(0, 2*np.pi, 500)
x = np.sin(3*t)*np.cos(t)
y = t/4
z = np.sin(3*t)*np.sin(t)
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-1,1])
ax.set_ylim3d([0,2])
ax.set_zlim3d([-1,1])
#
ax.plot(x,y,z)
#
ax.view_init(elev=15)
plt.title("Oscillating Petals")
plt.show()

#END EXERCISE 4

#BEGIN EXERCISE 5

t = np.linspace(0, 2*np.pi, 500)
x = (1+np.cos(16*t))*np.cos(t)
y = (1+np.cos(16*t))*np.sin(t)
z = 1+np.cos(16*t)
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-2,2])
ax.set_ylim3d([-2,2])
ax.set_zlim3d([0,2])
#
ax.plot(x,y,z)
#
ax.view_init(elev=50)
plt.title("16 Petal Cone")
plt.show()

#END EXERCISE 5

t = np.linspace
x = (4+np.sin(20*t))*np.cos(t)
y = (4+np.sin(20*t))*np.sin(t)
z = np.cos(20*t)
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-5,5])
ax.set_ylim3d([-5,5])
ax.set_zlim3d([-1.2,1.2])
#
ln, = ax.plot([], [])
points = ax.plot([], [], marker='o', color='blue')[0]
#Creates empty Line#D objects for the trajectories and planet positions.
plt.title("Animation of Torroidal Spiral")
#
def update(i):
  ax.view_init(elev=30*np.cos(0.005*i), azim=45) #Change view angle each frame
  ln.set_data(x[:i+1],y[:i+1]) #set_data on each line object for (x,y)
  ln.set_3d_properties(z[:i+1]) #set_3d properties on each line for z.
  points.set_data(x[i],y[i]) #do the same for each plot.
  points.set_3d_properties(z[i])
  return ln,points
#Define update function
ani = animation.FuncAnimation(fig,update,len(x),interval=20)
plt.show()
ani.save('BrownianMotion.gif', writer = "pillow", fps=10)
#Creates and Saves Animation, close figure
rc('animation', html='jshtml')
ani
#Enables Colab Support

#BEGIN EXERCISE 6

t = np.linspace(0, 2*np.pi, 500)
x = np.cos(2*t)
y = np.cos(3*t)
z = np.cos(4*t)
#
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim3d([-1,1])
ax.set_ylim3d([-1,1])
ax.set_zlim3d([-1,1])
#
ln, = ax.plot([], [])
points = ax.plot([], [], marker='o', color='blue')[0]
#
plt.title("Loopy Boi")
#
ani = animation.FuncAnimation(fig,update,len(x),interval=20)
plt.show()
ani.save('.gif', writer = "pillow", fps=10)
#
rc('animation', html='jshtml')
ani
