import numpy as np
from scipy.optimize import differential_evolution, minimize
import timeit

# The Function
def rossenbrock(x):
    return 100*(x[1] - x[0]**2)**2 + (1 - x[0])**2

def eggcrate(x):
    return x[0]**2 + x[1]**2 + 25*((np.sin(x[0]))**2 + (np.sin(x[1]))**2)

def golinski_heu(x):

    G1 = 1 - 27 * x[0] ** (-1) * x[1] ** (-2) * x[2] ** (-1) >= 0
    G2 = 1 - 397.5 * x[0] ** (-1) * x[1] ** (-2) * x[2] ** (-2) >= 0
    G3 = 1 - 1.93 * x[1] ** (-1) * x[2] ** (-1) * x[3] ** 3 * x[5] ** (-4) >= 0
    G4 = 1 - 1.93 * x[1] ** (-1) * x[2] ** (-1) * x[4] ** 3 * x[6] ** (-4) >= 0
    G5 = 1 - ((745 * x[3] * x[1] ** (-1) * x[2] ** (-1))
              ** 2 + 16.9 * 10 ** 6) ** (0.5) / (110 * x[5] ** 3) >= 0
    G6 = 1 - ((745 * x[4] * x[1] ** (-1) * x[2] ** (-1))
              ** 2 + 157.5 * 10 ** 6) ** (0.5) / (85 * x[6] ** 3) >= 0
    G7 = 1 - x[1] * x[2] / 40 >= 0
    G8 = 1 - 5 * x[1] / x[0] >= 0
    G9 = 1 - x[0] / (12 * x[1]) >= 0
    G24 = 1 - (1.5 * x[5] + 1.9) * x[3] ** (-1) >= 0
    G25 = 1 - (1.1 * x[6] + 1.9) * x[4] ** (-1) >= 0

    if G1*G2*G3*G4*G5*G6*G7*G8*G9*G24*G25 == 1:
        return 0.7854 * x[0] * x[1] ** 2 * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934) \
               - 1.5079 * x[0] * (x[5] ** 2 + x[6] ** 2) + 7.477 * (x[5] ** 3 + x[6] ** 3) \
               + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)
    else:
        return 10000

def golinski_sqp(x):
    return 0.7854 * x[0] * x[1] ** 2 * (3.3333 * x[2] ** 2 + 14.9334 * x[2] - 43.0934) \
               - 1.5079 * x[0] * (x[5] ** 2 + x[6] ** 2) + 7.477 * (x[5] ** 3 + x[6] ** 3) \
               + 0.7854 * (x[3] * x[5] ** 2 + x[4] * x[6] ** 2)

cons = ({'type': 'ineq', 'fun': lambda x: 1 - 27 * x[0] ** (-1) * x[1] ** (-2) * x[2] ** (-1)},
        {'type': 'ineq', 'fun': lambda x: 1 - 397.5 * x[0] ** (-1) * x[1] ** (-2) * x[2] ** (-2)},
        {'type': 'ineq', 'fun': lambda x: 1 - 1.93 * x[1] ** (-1) * x[2] ** (-1) * x[3] ** 3 * x[5] ** (-4)},
        {'type': 'ineq', 'fun': lambda x: 1 - 1.93 * x[1] ** (-1) * x[2] ** (-1) * x[4] ** 3 * x[6] ** (-4)},
        {'type': 'ineq', 'fun': lambda x: 1 - ((745 * x[3] * x[1] ** (-1) * x[2] ** (-1))
                                               ** 2 + 16.9 * 10 ** 6) ** (0.5) / (110 * x[5] ** 3)},
        {'type': 'ineq', 'fun': lambda x: 1 - ((745 * x[4] * x[1] ** (-1) * x[2] ** (-1))
                                               ** 2 + 157.5 * 10 ** 6) ** (0.5) / (85 * x[6] ** 3)},
        {'type': 'ineq', 'fun': lambda x: 1 - x[1] * x[2] / 40},
        {'type': 'ineq', 'fun': lambda x: 1 - 5 * x[1] / x[0]},
        {'type': 'ineq', 'fun': lambda x: 1 - x[0] / (12 * x[1])},
        {'type': 'ineq', 'fun': lambda x: 1 - (1.5 * x[5] + 1.9) * x[3] ** (-1)},
        {'type': 'ineq', 'fun': lambda x: 1 - (1.1 * x[6] + 1.9) * x[4] ** (-1)})

# Bound
bounds1 = [(-20., 20.), (-20., 20.)]
bounds2 = [(2.6, 3.6), (0.7, 0.8), (17, 28), (7.3, 8.3), (7.3, 8.3), (2.9, 3.9), (5, 5.5)]

# Initial guess
x0 = np.array([-1., 1.])
x01 = np.array([ 2.6, 0.7, 17., 7.3, 7.3, 2.9, 5 ])


def optimize(func=object, x0=x0, bounds=None):

    start1 = timeit.default_timer()
    # Gradient based optimization
    res1 = minimize(func, x0, method='SLSQP', options={"maxiter" : 100}, bounds=bounds)
    stop1 = timeit.default_timer()

    start2 = timeit.default_timer()
    # Basinhopping optimization
    res2 = differential_evolution(func, bounds,
                                  popsize=30, maxiter=100, polish=False, mutation=(0.6, 1.5))
    stop2 = timeit.default_timer()

    t_gb = stop1 - start1
    t_heu = stop2 - start2

    print('***'+ func.__name__, 'equation***')
    print("global minimum gradient based: x = [%.4f, %.4f], f(x) = %.4f, t = %.4f s"
          % (res1.x[0], res1.x[1], res1.fun, t_gb))
    print("global minimum heuristic: x = [%.4f, %.4f], f(x) = %.4f, t = %.4f s \n"
          % (res2.x[0], res2.x[1], res2.fun, t_heu))

optimize(rossenbrock, x0, bounds1)
optimize(eggcrate, x0, bounds1)

start3 = timeit.default_timer()
# Gradient based optimization
res1 = minimize(golinski_sqp, x01, method='SLSQP',
                options={"maxiter" : 100}, bounds=bounds2, constraints=cons)
stop3 = timeit.default_timer()

while True:
    start4 = timeit.default_timer()
    # differential evolution optimization
    res2 = differential_evolution(golinski_heu, bounds2,
                                  popsize=80, maxiter=100, polish=False, mutation=(0.2, 1))
    stop4 = timeit.default_timer()
    if res2.fun != 10000:
        break

t_gb = stop3 - start3
t_heu = stop4- start4

print('***golinski equation***')
print("global minimum gradient based: x =",
      ['%.4f' % elem for elem in res1.x],", f(x) = %.4f, t = %.4f s"
      % (res1.fun, t_gb))
print("global minimum heuristic: x =",
      ['%.4f' % elem for elem in res2.x],", f(x) = %.4f, t = %.4f s \n"
      % (res2.fun, t_heu))

n_test = 20
n_failed = 0

for i in range(n_test):
    # differential evolution optimization
    res3 = differential_evolution(golinski_heu, bounds2,
                                  popsize=80, maxiter=100, polish=False, mutation=(0.2, 1))
    if res3.fun == 10000:
        n_failed += 1

prob = (n_test - n_failed)/n_test

print("probability for heuristic to reach global optimum:", prob)