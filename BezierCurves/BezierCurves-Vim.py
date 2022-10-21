import numpy as np
import sympy as sp
import math
from matplotlib import pyplot as plt

#BEGIN EXERCISE 1

def DeCasteljau(M,t):
  assert 0 <= t
  assert 1 >= t
  M = np.array(M)
  #port M to numpy
  dimPoint = np.shape(M)[1]
  #point dimension
  arrCurr = np.copy(M)
  #current point array
  numPoints = np.shape(arrCurr)[0]
  #current number of points
  arrEmpt = np.empty([0,dimPoint])
  #empty array of appropriate dimension
  pointEmpt = np.empty([0,0])
  #empty point
  arrRecur = np.copy(arrEmpt)
  #current recursive array
  while numPoints > 1:
    for n in range(numPoints-1):
      pointEval = np.copy(pointEmpt)
      #current point evaluation
      arrVec = np.array([arrCurr[n],arrCurr[n+1]])
      arrVec = np.transpose(arrVec)
      for i in range(dimPoint):
        vecComp = np.array([((-t+1)*arrVec[i][0])+(t*arrVec[i][1])])
        #vector component
        pointEval = np.concatenate((pointEval,vecComp),axis=None)
      pointEval = np.array([pointEval])
      arrRecur = np.concatenate((arrRecur,pointEval),axis=0)
      pointEval = np.copy(pointEmpt)
      #add pointEval to arrRecur and clear pointEval
    arrCurr = np.copy(arrRecur)
    arrRecur = np.copy(arrEmpt)
    #set arrCurr to arrRecur and clear arrRecur
    numPoints = np.shape(arrCurr)[0]
    #update numPoints
  assert numPoints == 1
  arrCurr = np.ndarray.tolist(arrCurr)
  return arrCurr

#END EXERCISE 1

#using list form instead of numpy arrays
def BezierPoint(M,t):
  numPoints = len(M)
  R = []
  #new point list
  V = []
  #point vector
  if numPoints == 1:
    return M[0]
  else:
    for i in range(numPoints-1):
      V = []
      A = M[i]
      B = M[i+1]
      for j in range(len(M[i])):
        V += [(-t+1)*(A[j]) + t*(B[j])]
        #evaluate each dimensional component of the vector from A to B
      R += [V]
    return BezierPoint(R,t)

#BEGIN EXERCISE 2

def BezierPlot2D(M):
  timeSteps = np.linspace(0,1,num=100)
  orderedPlotPoints = np.empty([0,2])
  for t in range(len(timeSteps)):
    orderedPlotPoints = np.concatenate((orderedPlotPoints,DeCasteljau(M,timeSteps[t])) 
  parametricSegments = np.transpose(orderedPlotPoints)
  x = parametricSegments[0]
  y = parametricSegments[1]
  plt.plot(x,y)
  plt.show()
  return

#END EXERCISE 2

def BezierPlot(M):
  timeSteps = np.linspace(0,1,num=100)
  curvePoints = []
  for k in range(len(timeSteps)):
    curvePoints += [BezierPoint(M,t)]
  x = []
  y = []
  for p in range(len(curvePoints)):
    x += [curvePoints[p][0]]
    y += [curvePoints[p][1]]
  plt.plot(x,y)
  plt.show()
