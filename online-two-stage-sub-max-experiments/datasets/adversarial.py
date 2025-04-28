import gurobipy as gp
from gurobipy import GRB
import itertools
import numpy as np

# Parameters
n, m = 6, 6   # number of sets (n), number of elements (m)
l, k = 4, 2   # outer (l) and inner (k) budgets
bigM = n      # safe big-M constant

# Generate all subsets up to size k and l
def all_binary_subsets(n, max_size):
    return [np.array([int(i in S) for i in range(n)], dtype=int)
            for r in range(max_size + 1)
            for S in itertools.combinations(range(n), r)]

subs_k = all_binary_subsets(n, k)
subs_l = all_binary_subsets(n, l)

# Initialize model
model = gp.Model("wtp_gap")
model.Params.OutputFlag = 0

# WTP function parameters
c = model.addVars(m, lb=0, ub=1, name="c")                      # weights
G = model.addVars(m, n, vtype=GRB.BINARY, name="G")             # incidence matrix

# Outer variable x: support for two-stage solution
x = model.addVars(n, lb=0, ub=1, name="x")
model.addConstr(gp.quicksum(x[i] for i in range(n)) <= l)

# ---------- TWO-STAGE VALUE ----------
theta = model.addVar(lb=0, name="theta")

for S in subs_k:
    z = []
    for u in range(m):
        cov = model.addVar(lb=0, ub=1, name=f"cov2s_{u}_{S}")
        model.addConstr(cov <= gp.quicksum(G[u, j] * S[j] for j in range(n)))
        model.addConstr(cov >= gp.quicksum(G[u, j] * S[j] for j in range(n)) / bigM)
        z.append(c[u] * cov)
    model.addConstr(theta >= gp.quicksum(z))

# ---------- ONE-STAGE VALUE ----------
phi = model.addVar(lb=0, name="phi")

for S in subs_l:
    for T in subs_k:
        if all(T[j] <= S[j] for j in range(n)):
            z2 = []
            for u in range(m):
                cov = model.addVar(lb=0, ub=1, name=f"cov1s_{u}_{T}")
                model.addConstr(cov <= gp.quicksum(G[u, j] * T[j] for j in range(n)))
                model.addConstr(cov >= gp.quicksum(G[u, j] * T[j] for j in range(n)) / bigM)
                z2.append(c[u] * cov)
            model.addConstr(phi >= gp.quicksum(z2))

# ---------- OBJECTIVE ----------
model.setObjective(theta - phi, GRB.MAXIMIZE)

# ---------- Solve ----------
model.optimize()

# ---------- Output ----------
if model.status == GRB.OPTIMAL and model.objVal > 1e-6:
    print(f"Counterexample found! Gap = {model.objVal:.4f}")
    print("Weights c =", [round(c[u].X, 2) for u in range(m)])
    print("Incidence matrix Γ (rows = elements, cols = sets):")
    Gamma = np.array([[int(G[u, j].X) for j in range(n)] for u in range(m)])
    print(Gamma)
else:
    print("No gap found with current (n, m, l, k). Try increasing n or m.")