import numpy as np
from scipy.optimize import fmin_bfgs
import matplotlib.pyplot as plt
import pandas as pd

# The Function
def rossenbrock(x):
    x1 = x[0]
    x2 = x[1]
    return 100*(x2 - x1**2)**2 + (1 - x1)**2

# Initial Value
x01 = np.linspace(-5, 5, 10).reshape(10,1)
x0 = np.append(x01, x01, 1)
n_iter =[]

# Plot for every starting point
for x in x0:
    y = []
    n = []
    # Find Minimum Point
    [fopt, xopt] = fmin_bfgs(rossenbrock, x, maxiter=2000, full_output=False, retall=True)
    for i, x in enumerate(xopt):
        n.append(i+1)
        y.append(rossenbrock(x))
    n_p = np.array(n)
    y_p = np.array(y)
    plt.plot(n_p, y_p)
    n_iter.append(len(xopt))

var_list = list(np.round(x0, 2))
dict = {'initial variable':var_list, 'iteration': n_iter}
df = pd.DataFrame(data=dict, index=np.arange(len(n_iter))+1)
print(df)
plt.xlabel('number of iteration')
plt.ylabel('objective point')
plt.legend(var_list, loc='upper right')

plt.show()







