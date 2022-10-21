import math
import numpy as np
import sympy as sp

#BEGIN EXERCISE 1

def PolygonArea(points):
  assert np.shape[1] == 2, "points must be two-dimensional"
  L = len(points)
  Area = 0
  for i in range(1,L+1):
    Area += (points[i%L]*points[(i+1)%L] - points[(i+1)%L]*points[i%L])

#END EXERCISE 1

#BEGIN EXERCISE 2

apoints = [(0, 0), (2, 1), (1, 3), (0, 2), (-1, 1)]
PolygonArea(apoints)
bpoints = [(3,0), (5,3), (1,7), (-1,4), (-5,7), (-5,-2), (-2,-6), (5,-6), (2,-3), (5, -2)]
PolygonArea(bpoints)

#END EXERCISE 2

#BEGIN EXERCISE 3

def PlaneFinder(points):
  assert np.shape(points) == (3,3), "input must be 3 points in R3"
  p1 = np.array(points[0])
  p2 = np.array(points[1])
  p3 = np.array(points[2])
  v1 = p2 - p1
  v2 = p3 - p1
  n = []
  for i in range(2,5):
    n += [(-1)**i*v1[i%3]*v2[(i+1)%3] - (-1)**i*v2[i%3]*v1[(i+1)%3]]
  N = np.array(n)
  d = np.dot(N,p1)
  n += d
  return n

#END EXERCISE 3

#BEGIN EXERCISE 4

def FaceArea(points):
  L = len(points)
  if np.shape(points)[1] == 2:
    for i in range(L):
      points[i] += [0]
  planepoints = []
  for i in range(3):
    planepoints += [points[i]]
  coef_list = PlaneFinder(planepoints)
  n = []
  for i in range(3):
    n += [coef_list[i]]
  points1 = []
  points2 = []
  points3 = []
  for i in range(L):
    points1 += [(points[i][0],points[i][1])]
    points2 += [(points[i][1],points[i][2])]
    points3 += [(points[i][2],points[i][0])]
  A1 = PolygonArea(points1) / (n[2])
  A2 = PolygonArea(points2) / (n[0])
  A3 = PolygonArea(points3) / (n[1])
  Area = (1/3) * (A1 + A2 - A3)
  return Area

#END EXERCISE 4

verticies1 = [[-0.5571,-0.3714,-0.7428],[-0.7636,-1.1758,-0.1857],[-1.5680,-0.7120,0.1857],[-1.3614,0.0924,-0.3714]]
print(FaceArea(verticies1))
verticies2 = [[-0.6667, -0.6667, -0.3333], [3.6667, -3.3333, 5.3333], [-1.6667, -8.6667, 2.6667], [-5.6667, -0.6667, -5.3333], [1.0000, 6.0000, -2.0000], [0.3333, 5.3333, -2.3333]]
print(FaceArea(verticies2))
verticies3 = [[1.2113, 1.2113, 2.4226],[-4.9434, 5.0566, 0.1132],[-3.2528, -1.2528, -4.5056],[6.2113, -3.7887, 2.4226],[4.0981, 4.0981, 8.1962]]
print(FaceArea(verticies3))
