import numpy as np
from gurobipy import Model, GRB, QuadExpr


class OptimalInHindsight:
    def __init__(self, fs, l, k, setting='fractional'):
        # setting = 'fractional' or 'integral'
        self.fs = fs
        self.setting = setting
        self.l = l
        self.k = k
        self.T = len(self.fs)
    
    def solve(self):
        n = len(self.fs[0].potentials[0].w) # dimensions

        # Create a Gurobi model
        model = Model("WTP_Optimization_with_y_constraint")
        model.setParam("OutputFlag", 0)  # Suppress Gurobi output

        # Decision variables x \in R^n
        if self.setting == 'fractional':
            x = model.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'x')
        else:
            x = model.addVars(n, vtype=GRB.BINARY, name=f'x')

        # Decision variables, y_t \in \R^n for each t \in [T]
        if self.setting == 'fractional':
            ys = [model.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{t}") for t in range(self.T)]
        else:
            ys = [model.addVars(n, vtype=GRB.BINARY, name=f"y_{t}") for t in range(self.T)]

        # Auxiliary variables for potential evaluations
        zs = [model.addVars(len(f.potentials), lb=0, name=f"z_{t}") for t, f in enumerate(self.fs)]
        
        # Objective: Maximize the WTP functions
        model.setObjective(
            sum(f.weights[i] * zs[t][i] for t, f in enumerate(self.fs) for i in range(len(f.potentials))), GRB.MAXIMIZE
        )
        
        # Constraints for each function f_t
        for t, f in enumerate(self.fs):
            # Constraints for each potential
            for i, potential in enumerate(f.potentials):
                # z_i <= b
                model.addConstr(zs[t][i] <= potential.b, name=f"z_{t, i}_upper")
                # z_i <= w \cdot x
                model.addConstr(
                    zs[t][i] <= sum(potential.w[j] * ys[t][j] for j in range(n)),
                    name=f"z_{t, i}_dot"
                )
        
        # Cardinality constraint for x
        model.addConstr(sum(x[i] for i in range(n)) == self.l, name=f"cardinality_x")

        # Cardinality constraint for each function f_t: sum(x) = k
        for t in range(len(self.fs)):
            model.addConstr(sum(ys[t][i] for i in range(n)) == self.k, name=f"cardinality_y_{t}")
        
        for t in range(self.T):
            # Element-wise constraint: y[i] <= x[i]
            upper_bound_constraints = []
            for i in range(n):
                constr = model.addConstr(ys[t][i] <= x[i], name=f"y_{t, i}_leq_x_{i}")
                upper_bound_constraints.append(constr)
        
        # Optimize the model
        model.optimize()
        
        # Extract the solution
        if model.status == GRB.OPTIMAL:
            x_vals = np.array([x[i].x for i in range(n)])
            return x_vals, sum([f.eval(x_vals) for f in self.fs])
        else:
            raise ValueError("Optimization failed. Status: " + str(model.status))


class Balkanski:
    def __init__(self, x, l, k, fs):
        self.x = x # fractional solution
        self.l = l
        self.k = k
        self.fs = fs

    def isValid(self, x):
        return sum(x) <= self.l

    def round(self):
        self.x = self.x * (1 - 1/np.sqrt(self.k))
        
        x_int = []
        for p in self.x:
            if np.random.rand() < p:
                x_int.append(1)
            else:
                x_int.append(0)

        if self.isValid(x_int):
            return x_int, sum([f.eval(x_int) for f in self.fs])
        return np.zeros_like(self.x), 0

class ReplacementGreedy:
    def __init__(self, fs, l, k, n):
        self.fs = fs
        self.T  = len(self.fs)
        self.n = n
        self.l = l
        self.k = k
        self.ys = [np.zeros(n) for f in self.fs] # ys[i] = y_i = solution for the i-th function
        self.x = np.zeros(n)
    
    def grad(self, elem, i):
        f = self.fs[i]
        x_old = self.ys[i]
        x_new = x_old.copy()
        x_new[elem] = 1
        if sum(self.ys[i]) < self.k:
            return f.eval(x_new) - f.eval(x_old), None
        else:
            replacementElem = -1
            maxGain = 0
            for rep in range(self.n):
                if x_old[rep] == 0: # element rep was not used in the old solution
                    continue
                x_new[rep] = 0 # try removing the element rep
                gain = f.eval(x_new) - f.eval(x_old)
                if gain > maxGain:
                    maxGain = gain
                    replacementElem = rep
                x_new[rep] = 1 # reset x_new
        
            return maxGain, (replacementElem if replacementElem > 0 else None)


    def argmax(self, l):
        maxInd = 0
        maxVal = 0
        for i in range(len(l)):
            if maxVal < l[i]:
                maxVal = l[i]
                maxInd = i
        return maxInd


    def solve(self):
        self.Ts = [np.zeros(self.n) for f in self.fs] # initialize Ts
        self.x = np.zeros(self.n)

        for j in range(self.l):
            maxElem = self.argmax([sum([self.grad(elem, i)[0] for i in range(self.T)]) for elem in range(self.n)])
            self.x[maxElem] = 1 # add element maxElem to solution
            for i in range(self.T):
                gain_i, replacementElem = self.grad(maxElem, i)
                if gain_i > 0:
                    self.ys[i][maxElem] = 1
                    if replacementElem:
                        self.ys[i][replacementElem] = 0
        
        val = sum([f.eval(self.x) for f in self.fs])
        return self.x, val

# helper function
def eval(wtp, x, k):
    m = len(wtp.potentials)  # Number of potentials
    n = len(x) # dimensions

    # Create a Gurobi model
    model = Model("WTP_Optimization_with_y_constraint")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output

    # Decision variables (binary)
    
    y = model.addVars(n, vtype=GRB.BINARY, name="y")
    
    # Auxiliary variables for potential evaluations
    z = model.addVars(m, lb=0, name="z")
    
    # Objective: Maximize the WTP function
    model.setObjective(
        sum(wtp.weights[i] * z[i] for i in range(m)), GRB.MAXIMIZE
    )
    
    # Constraints for each potential
    for i, potential in enumerate(wtp.potentials):
        # z_i <= b
        model.addConstr(z[i] <= potential.b, name=f"z_{i}_upper")
        # z_i <= w \cdot x
        model.addConstr(
            z[i] <= sum(potential.w[j] * y[j] for j in range(n)),
            name=f"z_{i}_dot"
        )
    
    # Cardinality constraint: sum(x) = k
    model.addConstr(sum(y[i] for i in range(n)) == k, name="cardinality")
    
    # Element-wise constraint: y[i] <= x[i]
    upper_bound_constraints = []
    for i in range(n):
        constr = model.addConstr(y[i] <= x[i], name=f"y_{i}_leq_x_{i}")
        upper_bound_constraints.append(constr)
    
    # Optimize the model
    model.optimize()
    
    # Extract the solution
    if model.status == GRB.OPTIMAL:
        # x_vals = np.array([x[i].x for i in range(n)])
        obj_val = model.objVal
        return obj_val
    else:
        raise ValueError("Optimization failed. Status: " + str(model.status))

class Random:
    def __init__(self, fs, n, l, k):
        self.fs = fs
        self.n = n
        self.l = l
        self.k = k
        self.T = len(self.fs)
        self.rewards = []

    def solve(self):
        self.rewards = []
        for t in range(self.T):
            indices = np.random.choice(self.n, self.l, replace=False)
            x_t = np.zeros(self.n)
            x_t[indices] = 1
            self.rewards.append(eval(self.fs[t], x_t, self.k))
        
        return self.rewards
        

