import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WTP import Potential, WTP

np.random.seed(seed = 42)
n = 100
l = 10
k = 1

wtps = []
for i in range(0, 3*l, 3):
    potentials = []
    weights = []

    # set i covers element i
    w_1 = np.zeros(n)
    w_1[i] = 1
    potential_1 = Potential(1, w_1)

    # set i+1 covers element i+1
    w_2 = np.zeros(n)
    w_2[i+1] = 1
    potential_2 = Potential(1, w_2)

    # set i+2 covers elements i, i+1, i+2
    w_3 = np.zeros(n)
    w_3[i] = w_3[i+1] = w_3[i+2] = 1
    potential_3 = Potential(1, w_3)

    potentials = [potential_1, potential_2, potential_3]
    weights = [1, 1, 1]

    wtps.append(WTP(potentials, weights))
    
file = './instances/coverage.pkl'
with open(file, 'wb') as file:
    pickle.dump(wtps, file)
print(f"List saved to {file}")

