import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt
import random
import math

#BEGIN EXERCISE 1

def openBallVolume(n, N = 10**4):
  points = np.random.uniform(-1,1,(n,N))
  lengths = la.norm(points, axis=0)
  num_within = np.count_nonzero(lengths < 1)
  volume = 2**n * (num_within / N)
  return volume

Y = []
x = np.linspace(1,20,20)
for i in range(20):
  y = []
  for j in range(len(x)):
    y += [openBallVolume(i)]
  Y += [y]
for i in range(20):
  Y[i] = np.sum(i)/20
plt.plot(x,Y)
plt.show()

#The Volume of the unit sphere as n approaches infinity is 0 
#The Volume of the unit sphere is the highest at n=6

#END EXERCISE 1

#BEGIN EXERCISE 2

x = np.linspace(1,30,30)
y = 2**(-1/x)
plt.plot(x,y)
plt.show()

#END EXERCISE 2

import numpy as np
#A function to generate n points on a d-dimensional unit sphere
def generate(n, d):
  list_of_points = []
  point = []
  for i in range(n):
    point = []
    for j in range(d):
      point.append(random.uniform(-1,1))
    norm = np.sqrt(np.sum([s**2 for s in point]))
    norm_point = point/norm
    list_of_points.append(tuple(norm_point))
  return list_of_points

#BEGIN EXERCISE 3

test_Points = generate(500,50)
pole_Points = generate(5,50)
band_Points = []
all_Points = 0
for i in range(len(pole_Points)):
  band_Points += [0]
  for j in range(len(test_Points)):
    if -50**-0.5 <= np.dot(pole_Points[i],test_Points[j]) <= 50*-0.5:
      band_Points[i] += 1
for i in range(len(test_Points)):
  for j in range(len(pole_Points)):
    if np.dot(pole_Points[j],test_Points[i]) < -50**-0.5 or np.dot(pole_Points[j],test_Points[i]) > 50**-0.5:
      break
    elif j == len(test_Points) - 1:
      all_Points += 1
print(band_Points)
print(all_Points)

#END EXERCISE 3

#BEGIN EXERCISE 4

#Part A

def Gauss_generate(n, d):
  list_of_points = []
  point = []
  for i in range(n):
    point = []
    for j in range(d):
      point.append(random.uniform(-1,1))
    list_of_points.append(tuple(point))
  return list_of_points

#Part B

child_Points = Gauss_generate(200,50)
parent_Point = Gauss_generate(1,50)[0]
angles_List = []
parent_Norm = (np.dot(parent_Point,parent_Point))**0.5
for i in range(len(child_Points)):
  angles_List += [(180/np.pi)*np.arccos(np.dot(parent_Point,child_Points[i])/(parent_Norm*np.dot(child_Points[i], child_Points[i])**0.5]
ortho_Count = 0
for i in range(len(angles_List)):
  if 87 <= angles_List[i] <= 93:
    ortho_Count += 1
print(ortho_Count)
edges = np.linspace(0, 180, 181)
plt.hist(angles_List, edges)
plt.show()
