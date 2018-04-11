import numpy as np
from scipy.optimize import fmin_slsqp

x = [0.371, 0.012, 1.768, 0.374, 0.375]
i = [7900, 210*10**9, 700*10**6, 0.9]
s = [2400, 31*10**9, 70*10**6, 0.04]
n = 1

def Cost(x, i=i, s=s, n=n):
    M_Ibeam = (2*x[0]*x[1] + (x[2] - 2*x[1])*x[1])*30*i[0]*n
    M_support = x[3]*x[4]*5*s[0]
    return i[3]*M_Ibeam + s[3]*M_support,

M_Ibeam = lambda x: (2 * x[0] * x[1] + (x[2] - 2 * x[1]) * x[1]) * 30 * i[0] * n
I_Ibeam = lambda x: ((x[2] - 2*x[1])**3*x[1]/12) + 2*((x[1]**3*x[0]/12) + x[1]*x[0]*((x[2]/2) + (x[1]/2))**2)
sigma_ibeam = lambda x: ((7425*10**4) + M_Ibeam(x)*73.575)*(x[2]/2)/(8*I_Ibeam(x)*n)
tau_Ibeam = lambda x: (M_Ibeam(x)*9.81 + 99*10**5)/(4*(2*x[0]*x[1] + (x[2] - 2*x[1]))*n)
P_applied = lambda x: (M_Ibeam(x)*9.81 + 99*10**5)/2
P_Crit = lambda x: np.pi**2*s[1]*min(x[3]**3*x[4]/12, x[3]*x[4]**3/12)/100
sigma_support = lambda x: P_applied(x)/(x[3]*x[4])

g1 = lambda x: i[2] - sigma_ibeam(x)
g2 = lambda x: i[2] - tau_Ibeam(x)
g3 = lambda x: P_Crit(x) - P_applied(x)
g4 = lambda x: s[2] - sigma_support(x)
g5 = lambda x: 1 - 2 * x[1] / x[2]
g6 = lambda x: 1 - x[1] / x[0]

ineqs = [g1, g2, g3, g4, g5, g6]
bounds = [(0.1, 2), (0.01, 1), (0.1, 4), (0.2, 4), (0.3, 5)]

out, fx, its, imode, smode = fmin_slsqp(Cost, x, ieqcons=ineqs, bounds=bounds, full_output=True, iter=100)

print(out)


