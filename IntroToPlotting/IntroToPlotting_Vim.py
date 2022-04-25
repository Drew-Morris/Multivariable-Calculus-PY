import numpy as np

from matplotlib import pyplot as plt

y = np.arange(-5,6)**2

plt.plot(y) #Draws Plot

plt.show() #Shows Plot

np.linspace(0,32,4) #Makes 4 evenly-spaced values between 0 and 32

x = np.linspace(-5,5,50) #Makes 50 evenly-spaced values between -5 and 5

y = x**2 #Calculates the range of f(x) = x**2

plt.plot(x,y)

plt.show()

#BEGIN EXERCISE 1

k1 = np.linspace(-2*np.pi,2*np.pi,100)

plt.plot(k1, np.sin(k1), 'r')
plt.plot(k1, np.cos(k1), 'b')
plt.plot(k1, np.arctan(k1), 'g')
#plotting all three functions on the same graph
plt.xlim(-2*np.pi,2*np.pi)
#limiting the x-axis to the correct domain
plt.show()

#END EXERCISE 1

x1 = np.linspace(-2,4,100)

plt.plot(x1,np.exp(x1),'g:', linewidth=6, label="Exponential")

plt.title("This is the title.", fontsize=18)

plt.legend(loc="upper left") #uses 'label' arg of plt.plot() to make legend

plt.show()

x2 = np.linspace(1,4,100)

plt.plot(x2, np.log(x2), 'r*', markersize=4)

plt.xlim(0,5) #sets the visible limits of x2

plt.xlabel("The x axis") #Labels the x axis

plt.show()

#BEGIN EXERCISE 2

def f(x):
  return 1/(x-1)

x3 = np.linspace(-2,6,100)

plt.plot(x3, f(x3), 'b-', linewidth=4, label="Given")

plt.legend(loc="upper left")

plt.show()

#PART A

x4 = np.linspace(-2,1,50, endpoint=False)

x5 = np.linspace(1,6,50, endpoint=False)[1:]

plt.plot(x4, f(x4), 'b', label="A")
plt.plot(x5, f(x5), 'b')
#splits the domain of f(x) to reflect the existence of its discontinuity

plt.legend(loc="upper left")

plt.show()

#PART B

plt.plot(x4, f(x4), 'm--', lw=4, label="B")
plt.plot(x5, f(x5), 'm--', lw=4)
#recolors the graph of f(x) as magenta dashed lines

plt.legend(loc="upper left")

plt.show()

#PART C

plt.plot(x4, f(x4), 'm--', lw=4, label="C")
plt.plot(x5, f(x5), 'm--', lw=4)

plt.xlim(-2,6)
plt.ylim(-6,6)
#limits the domain of f(x) to (-2,6) and the range to (-6,6)

plt.legend(loc="upper left")

plt.show()

#END EXERCISE 2

#BEGIN EXERCISE 3

xf = np.linspace(-4*np.pi, 4*np.pi, 1000)

def T(n,x):
  y = 0
  for i in range(0,n+1):
    def p(x):
      ((-1)**i*x**(2*i+1))/(np.math.factorial(2*i+1))
    y = p(x)+y
  return y
#Function that computes the nth Taylor Series of the sine function

plt.plot(xf, np.sin(xf), 'c', label="sin(x)")
plt.plot(xf, T(1,xf), 'y', label="1st Taylor Series of sin(x)")
plt.plot(xf, T(3,xf), 'g', label="3rd Taylor Series of sin(x)")
plt.plot(xf, T(5,xf), 'm', label="5th Taylor Series of sin(x)")
#Plot the 1st, 3rd, and 5th Taylor Series of sin(x) as well as sin(x)

plt.xlim(-4*np.pi,4*np.pi)
plt.ylim(-3,3)

plt.legend(loc="upper left")

plt.title("Taylor Series of sin(x)")

plt.xlabel("x-axis")
plt.ylabel("y-axis")

plt.show()

#END EXERCISE 3  
