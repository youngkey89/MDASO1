import numpy as np
import matplotlib.pyplot as plt

# Define function

def f(x):
    return np.exp(x)

x0 = 1
h_value = np.logspace(-15, 1, 17)

# Analytical Result
def df1_ana(x):
    "first derivative"
    return np.exp(x)

def df2_ana(x):
    "second derivative"
    return np.exp(x)

def df1_ff(x0, h):
    'first derivative using Finite forward-difference approximation'
    return (f(x0 + h) - f(x0))/h

def df1_cf(x0, h):
    'first derivative using Finite center-difference approximation'
    return (f(x0 + h) - f(x0-h))/(2*h)

def df1_com(x0, h):
    'first derivative using complex step approximation'
    return np.imag(f(x0 + 1j*h))/h

def df2_so(x0, h):
    'second derivative using second order finite difference approximation'
    return (f(x0 + h) - 2*f(x0) + f(x0 - h))/(h**2)

def df2_com(x0, h):
    'second derivative using second order finite difference approximation'
    return (2/(h**2))*(f(x0) - np.real(f(x0 + 1j*h)))

def error(dy, dy_ana):
    'error calculation'
    return abs(dy - dy_ana)/dy_ana

#First derivative calculation
dy11 = df1_ff(x0, h_value)
dy12 = df1_cf(x0, h_value)
dy13 = df1_com(x0, h_value)
dy1_ana = df1_ana(x0)
er11 = error(dy11, dy1_ana)
er12 = error(dy12, dy1_ana)
er13 = error(dy13, dy1_ana)

#Second derivative calculation
dy21 = df2_so(x0, h_value)
dy22 = df2_com(x0, h_value)
dy2_ana = df2_ana(x0)
er21 = error(dy21, dy2_ana)
er22 = error(dy22, dy2_ana)

plt.figure(1)
plt.plot(h_value, er11, 'b--', label='forward')
plt.plot(h_value, er12, 'r--', label='center')
plt.plot(h_value, er13, 'g--', label='complex')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("pertubation step size")
plt.ylabel("error")
plt.title("first derivative approximation")
plt.legend()

plt.figure(2)
plt.plot(h_value, er21, 'b--', label='second order')
plt.plot(h_value, er22, 'r--', label='complex')
plt.xscale("log")
plt.yscale("log")
plt.xlabel("pertubation step size")
plt.ylabel("error")
plt.title("second derivative approximation")
plt.legend()
plt.show()
