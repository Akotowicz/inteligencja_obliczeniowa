import pyswarms as ps
import math
import numpy as np
from pyswarms.utils.plotters import plot_cost_history
import matplotlib.pyplot as plt
from pyswarms.utils.plotters.plotters import plot_contour
from pyswarms.utils.plotters.formatters import Mesher

def endurance(x, y, z, u, v, w):
    return -(math.exp(-2*(y-math.sin(x))**2)+math.sin(z*u)+math.cos(v*w))

def myEndurance(x):
    return endurance(x[0],x[1],x[2],x[3],x[4],x[5])

def f(x):
    n_particles = x.shape[0]
    j = [myEndurance(x[i]) for i in range(n_particles)]
    return np.array(j)

my_bounds = (np.zeros(6), np.ones(6))
options = {'c1':0.5, 'c2':0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=6, options=options, bounds=my_bounds)

cost, pos = optimizer.optimize(f, iters=1000)
print(f"Best cost {cost}, Best pos: {pos}")

cost_history = optimizer.cost_history
plot_cost_history(cost_history)
plt.show()