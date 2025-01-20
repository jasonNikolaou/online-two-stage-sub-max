import numpy as np

class Pipage:
    def __init__(self):
        pass
    
    def clean(self, x):
        for i in range(len(x)):
            if abs(x[i]) < 1e-8:
                x[i] = 0
            elif abs(x[i] - 1) < 1e-8:
                x[i] = 1
            else:
                print(f'Rounding error: x[{i}] = {x[i]} not close to 0 or 1.')
                
        return x
    def round(self, y):
        x = y.copy()
        n = len(x)
        fractional_indices = [i for i in range(n) if 0 < x[i] < 1]
        while fractional_indices:
            if len(fractional_indices) == 1:
                i = fractional_indices[0]
                x[i] = 1
            else:
                i, j= np.random.choice(fractional_indices, size=2, replace=False)
                eps1 = min(1 - x[i], x[j])
                eps2 = min(x[i], 1 - x[j])
                
                if np.random.rand() < eps2 / (eps1 + eps2):
                    x[i] += eps1
                    x[j] -= eps1
                else:
                    x[i] -= eps2
                    x[j] += eps2

            # update fractional indices
            fractional_indices = [i for i in range(n) if 0 < x[i] < 1]

        return self.clean(x)