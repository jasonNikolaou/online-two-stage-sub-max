import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WTP import Potential, WTP

np.random.seed(seed = 42)
n = 100
l = 20
k = 1

M = 100
m = M

# l sets. each set covers a single element with value M
potentials_high_value = []
for i in range(l):
    w_0 = np.zeros(n)
    w_0[i] = 1
    potentials_high_value.append(Potential(1, w_0))
    
wtp1 = WTP(potentials_high_value, [M] * l)


# l sets. each set covers one element with value m
wtps_low_value = []
for i in range(l):
    w_2= np.zeros(n)
    w_2[l + i] = 1
    potential_2 = Potential(1, w_2)
    wtp2 = WTP([potential_2], [m])
    wtps_low_value.append(wtp2)

wtps = [wtp1] + wtps_low_value

wtps = wtps * 20

file = './instances/coverage.pkl'
with open(file, 'wb') as file:
    pickle.dump(wtps, file)
print(f"List saved to {file}")

