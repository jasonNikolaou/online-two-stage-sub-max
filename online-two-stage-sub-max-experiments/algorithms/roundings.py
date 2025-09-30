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

    def round(self, x):
        """ 
            Randomized pipage/dependent rounding for a sum (cardinality) constraint.
            Runs in O(m) where m = number of fractional entries.
        """
        x = np.asarray(x, dtype=np.float64).copy()
        n = x.size

        # Build a stack/array of fractional indices
        frac = np.flatnonzero((x > 0.0) & (x < 1.0))
        m = frac.size
        
        rand = np.random.random

        # Process last two fractional indices each time
        top = m  # length of the "stack" inside frac
        while top >= 2:
            top -= 1; i = int(frac[top])
            top -= 1; j = int(frac[top])

            a, b = x[i], x[j]

            eps1 = min(1.0 - a, b)
            eps2 = min(a, 1.0 - b)

            if rand() < (eps2 / (eps1 + eps2)):
                a2 = a + eps2
                b2 = b - eps2
            else:
                a2 = a - eps1
                b2 = b + eps1

            x[i], x[j] = a2, b2

            # Push back whichever of the two is still fractional
            if 0.0 < a2 < 1.0:
                frac[top] = i
                top += 1
            if 0.0 < b2 < 1.0:
                frac[top] = j
                top += 1

        # If one variable remains (non-integral sum in the given vector), do a final Bernoulli trial.
        if top == 1:
            i = int(frac[0])
            x[i] = 1.0 if rand() < x[i] else 0.0

        return x.astype(np.int8, copy=False)