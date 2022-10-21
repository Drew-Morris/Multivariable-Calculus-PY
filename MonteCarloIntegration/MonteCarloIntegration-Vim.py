import numpy as np
from scipy import linalg as la

#samples 2000 random points in [-1,1]x[-1,1]
points = np.random.uniform(-1,1,(2,2000))
#determines how many points are within the unit circle
lengths = la.norm(points, axis=0)
num_within = np.count_nonzero(lengths < 1)
#estimates the circles's area
area = 4 * (num_within / 2000)
#this is an estimation of pi

#BEGIN EXERCISE 1

def openBallVolume(n, N = 10**4):
  points = np.random.uniform(-1,1,(n,N))
  lengths = la.norm(points, axis=0)
  num_within = np.count_nonzero(lengths < 1)
  volume = 2**n * (num_within / N)
  return volume

#END EXERCISE 1

#BEGIN EXERCISE 2

def montecarlo_1D(f, a, b, N = 10**4):
  points = np.random.uniform(a,b,(1,N))
  approx = ((b-a)/N) * np.sum(f(points))
  return approx

#END EXERCISE 2

#BEGIN EXERCISE 3

def montecarlo_nD(f, a, b, N = 10**4):
  n = len(a)
  points = np.random.uniform(0,1,(n,N))
  volume = []
  for i in range(n):
    B = b[i]
    A = a[i]
    volume += [B-A]
    for j in range(N):
      points[i][j] = (B-A)*points[i][j] + A
  approx = (np.prod(volume)/N) * np.sum(f(points))
  return approx

#END EXERCISE 3
