from WTP.WTP import WTP, Potential
from algorithms.algorithms import GradientAscent
from algorithms.roundings import Pipage
import numpy as np
from gurobipy import Model, GRB, QuadExpr
from tqdm import tqdm

# === Helper function ====
def eval(wtp, x, k, setting='fractional'):
    # setting = 'fractional' or 'integral'

    m = len(wtp.potentials)  # Number of potentials
    n = len(x) # dimensions

    # Create a Gurobi model
    model = Model("WTP_Optimization_with_y_constraint")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output

    # Decision variables (binary)
    if setting == 'fractional':
        y = model.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")
    else:
        y = model.addVars(n, vtype=GRB.BINARY, name="y")
    
    # Auxiliary variables for potential evaluations
    z = model.addVars(n, lb=0, name="z")
    
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

# ===============================

class OnlineLearning:
    def __init__(self, fs, x0, k, algorithm, rounding):
        self.fs = fs
        self.T = len(fs)
        self.k = k
        self.x0 = x0
        self.algorithm = algorithm
        self.rounding = rounding
        self.frac_rewards = []
        self.int_rewards = []
        self.xs_int = [] # integral solutions
        self.xs = [] # fractional solutions
    
    
    def run(self):
        self.frac_rewards = []
        self.int_rewards = []
        x = self.x0
        for t in tqdm(range(self.T)):
            self.frac_rewards.append(eval(self.fs[t], x, self.k, 'fractional'))
            self.xs.append(x)

            x_int = self.rounding.round(x)
            self.xs_int.append(x_int)
            self.int_rewards.append(eval(self.fs[t], x_int, self.k, 'integral'))
            
            x = self.algorithm.next(self.fs[:t+1], x)
        

if __name__ == '__main__':
    P1 = Potential(1, np.array([0, 0, 0, 1]))
    P2 = Potential(1, np.array([1, 0, 0, 0]))
    WTP = WTP([P1, P2], np.array([1, 1]))
    fs = [WTP] * 20

    
    x0 = np.array([1, 1, 0, 0])
    GA = GradientAscent(l=3, k=2, n=len(x0), eta = 0.1, setting='one-stage')
    Pipage = Pipage()
    OL = OnlineLearning(fs, x0, GA, Pipage)
    OL.run()
    
    
