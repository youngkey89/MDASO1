import numpy as np
from scipy.optimize import minimize

#Objective Equation
def obj(x, sign=1):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    return sign*(100*x1*np.exp(-x1/100) + 150*x2*np.exp(-x2/150) + 300*x3*np.exp(-x3/300))

#Initial Value
x0 = [1,1,1]

#Constraint
def con(x):
    return 153 - 100*np.exp(-x[0]/100) - 150*np.exp(-x[1]/150) - 300*np.exp(-x[2]/300)

cons = {'type':'ineq', 'fun': con}

res = minimize(obj, x0, args=(-1.0,), method='SLSQP', constraints=cons)

D = np.zeros(3)
D[0] = 100*np.exp(-res.x[0]/100)
D[1] = 150*np.exp(-res.x[1]/150)
D[2] = 300*np.exp(-res.x[2]/300)

print("Price ($) :", res.x)
print("Seat quantity :", D)
print("Optimum Revenue ($):", obj(res.x))
