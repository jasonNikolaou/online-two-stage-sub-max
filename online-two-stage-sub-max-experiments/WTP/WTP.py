import numpy as np

class Potential:
    def __init__(self, b, w):
        assert isinstance(b, (int, float)), f'b is not a float'
        assert isinstance(w, np.ndarray), f'w is not a np array'
        self.b = b
        self.w = w
    
    def eval(self, x):
        return min(self.b, np.dot(self.w, x))
    
    def grad(self, x):
        if np.dot(self.w, x) <= self.b:
            return self.w
        
        return np.zeros_like(x)
        

class WTP:
    def __init__(self, potentials = [], weights = []):
        assert len(potentials) == len(weights), f'len(potentials) != len(weights)'    
        self.potentials = potentials
        self.weights = weights
    
    def eval(self, x):
        potentials_val = [p.eval(x) for p in self.potentials]
        return np.dot(self.weights, potentials_val)

    def grad(self, x):
        grads = [p.grad(x) for p in self.potentials]
        return np.sum(grads, axis=0)


if __name__ == '__main__':
    P1 = Potential(1,[1, 0, 0, 1])
    P2 = Potential(1, [1, 0, 0, 0])
    WTP = WTP([P1, P2], [0.5, 1])

    x = [0.5, 0, 0, 0.5]
    print(P1.eval(x), P2.eval(x))
    print(WTP.eval(x))
    print(WTP.grad(x))