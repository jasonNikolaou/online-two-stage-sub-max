import numpy as np
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from WTP import Potential, WTP

np.random.seed(seed = 42)
n = 30
max_val = 100
min_val = 0
mu = 60
sigma = 20

m = 100
inputs = []
for j in range(m):
    print(f'Creating function #{j}')
    values = np.random.normal(mu, sigma, n)
    values = np.minimum([max_val] * n, values)
    values = np.maximum([min_val] * n, values)
    h = values # value of each player

    # if j < m / 2:
    #     h[0] = h[1] = h[2] = h[3] = max_val # 4 players with maximium value in the first half
    # else:
    #     h[-1] = h[-2] = h[-3] = h[-4] = max_val # 4 players with maximum value in the second half
    h[0] = h[1] = h[2] = h[3] = max_val # 4 players with maximium value

    H = np.zeros(n**2).reshape(n, n) # pairwise complementarity
    for i in range(n-1):
        for j in range(i+1, n):
            H[i][j] = min(np.random.normal(-20, 10, 1), 0)
            H[j][i] = H[i][j]
    # if j < m / 2: 
    #     H[0] = H[1] = H[2] = H[3] = np.zeros(n)
    #     H[:,0] = H[:,1] = H[:,2] = H[:,3] = np.zeros(n)
    # else:
    #     H[-1] = H[-2] = H[-3] = H[-4] = np.zeros(n)
    #     H[:,-1] = H[:,-2] = H[:,-3] = H[:,-4] = np.zeros(n)
    H[0] = H[1] = H[2] = H[3] = np.zeros(n)
    H[:,0] = H[:,1] = H[:,2] = H[:,3] = np.zeros(n)

    # shrink rows and cols of H until we achieve submodularity
    ones = np.ones(n)
    while True:
        vec = h + np.dot(H, ones)
        for i in range(len(vec)):
            if h[i] == 0:
                H[i] = list(np.zeros(n))
                H[:,i] = list(np.zeros(n))

            if vec[i] < 0:
                H[i] = H[i] / 1.1 # shrink row
                H[:,i] = H[:,i] / 1.1 # shrink col

        if np.all(vec >= 0):
            break
    
    inputs.append((h, H))

wtps = []
for input in inputs:
    h, H = input
    potentials = []
    weights = []

    # Linear term
    potentials.append(Potential(np.infty, h + H @ np.ones(n)))
    weights.append(1)

    # Quadratic term
    for i in range(n-1):
        for j in range(i+1, n):
            w = np.zeros(n)
            w[i] = w[j] = 1
            potentials.append(Potential(1, w))
            assert H[i][j] <= 0, f'H[{i}][{j}] = {H[i][j]} is not negative'
            weights.append(-H[i][j])

    wtps.append(WTP(potentials, weights))
    
file = './instances/teamformation.pkl'
with open(file, 'wb') as file:
    pickle.dump(wtps, file)
print(f"List saved to {file}")


# x_1 = np.zeros(n) # Optimal team for the first half
# x_1[[0, 1, 2, 3]] = 1
# print(fs[5].eval(x_1))

# x_2 = np.zeros(n) # Optimal team for the second half
# x_2[[-1, -2, -3, -4]] = 1
# print(fs[73].eval(x_2))