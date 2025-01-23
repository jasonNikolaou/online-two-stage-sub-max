from gurobipy import Model, GRB, QuadExpr
import gurobipy as gp
import numpy as np

# helper functions
def project(z, n, k):
    '''Project z to the k-simplex: \Delta_n^k'''

    # Create a Gurobi model
    model = Model("simplex_projection")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output
    
    # Add variables x_i with bounds 0 <= x_i <= 1
    x = model.addVars(n, lb=0, ub=1, name="x")
    
    # Objective: Minimize the L2 norm (sum of squared differences)
    objective = QuadExpr()
    for i in range(n):
        objective += (x[i] - z[i]) ** 2
    model.setObjective(objective, GRB.MINIMIZE)
    
    # Add equality constraint: sum(x) = k
    model.addConstr(sum(x[i] for i in range(n)) == k, name="sum_constraint")
    
    # Optimize the model
    model.optimize()
    
    # Check optimization result
    if model.status == GRB.OPTIMAL:
        return [x[i].x for i in range(n)]  # Return the projected vector
    else:
        raise ValueError("Optimization failed. Status: " + str(model.status))

def gradient(wtp, x, k):
    m = len(wtp.potentials)  # Number of potentials
    n = len(x) # dimensions

    # Create a Gurobi model
    model = Model("WTP_Optimization_with_y_constraint")
    model.setParam("OutputFlag", 0)  # Suppress Gurobi output

    # Decision variables (binary)
    y = model.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")

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
        # obj_val = model.objVal
        lambdas = np.array([constr.Pi for constr in upper_bound_constraints])
        return lambdas
    else:
        raise ValueError("Optimization failed. Status: " + str(model.status))

# def gradient(wtp, x, k, epsilon=1e-3):
#     m = len(wtp.potentials)  # Number of potentials
#     n = len(x)  # Dimensions

#     # Create a Gurobi model
#     model = Model("WTP_Optimization_with_y_constraint")
#     model.setParam("OutputFlag", 0)  # Suppress Gurobi output

#     # Decision variables (binary)
#     y = model.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name="y")

#     # Auxiliary variables for potential evaluations
#     z = model.addVars(m, lb=0, name="z")
    
#     # Objective: Maximize the WTP function
#     model.setObjective(
#         sum(wtp.weights[i] * z[i] for i in range(m)), GRB.MAXIMIZE
#     )
    
#     # Constraints for each potential
#     for i, potential in enumerate(wtp.potentials):
#         # z_i <= b
#         model.addConstr(z[i] <= potential.b, name=f"z_{i}_upper")
#         # z_i <= w \cdot x
#         model.addConstr(
#             z[i] <= sum(potential.w[j] * y[j] for j in range(n)),
#             name=f"z_{i}_dot"
#         )
    
#     # Cardinality constraint: sum(y) = k
#     model.addConstr(sum(y[i] for i in range(n)) == k, name="cardinality")
    
#     # Element-wise constraint: y[i] <= x[i]
#     upper_bound_constraints = []
#     for i in range(n):
#         constr = model.addConstr(y[i] <= x[i], name=f"y_{i}_leq_x_{i}")
#         upper_bound_constraints.append(constr)
    
#     # Optimize the model
#     model.optimize()
    
#     # Check for optimality
#     if model.status != GRB.OPTIMAL:
#         raise ValueError("Optimization failed. Status: " + str(model.status))
    
#     # Store the original objective value
#     original_obj_val = model.objVal

#     # Initialize the result vector (grad)
#     grad = np.zeros(n)

#     # Extract the solution for warm start
#     y_solution = {i: y[i].X for i in range(n)}
#     z_solution = {i: z[i].X for i in range(m)}

#     # Compute gradients for each variable
#     for i in range(n):
#         if upper_bound_constraints[i].Pi > 0 and x[i] < 1:  # Check if tight and x[i] < 1
#             # Create a new model to re-solve the LP with x_i increased by epsilon
#             new_model = model.copy()

#             # Access the constraint directly and modify its RHS
#             constraint = upper_bound_constraints[i]
#             constraint.RHS = x[i] + epsilon  # Set the new RHS value

#             # Warm start by setting the solution from the previous model
#             for j in range(n):
#                 new_model.getVarByName(f"y[{j}]").start = y_solution[j]
#             for j in range(m):
#                 new_model.getVarByName(f"z[{j}]").start = z_solution[j]

#             # Optimize the new model
#             new_model.optimize()

#             if new_model.status == GRB.OPTIMAL:
#                 # Compute the gradient as change in objective divided by epsilon
#                 new_obj_val = new_model.objVal
#                 grad[i] = (new_obj_val - original_obj_val) / epsilon
#             else:
#                 raise ValueError(f"Re-optimization failed for variable {i}.")
#         else:
#             # If the constraint is not tight or x[i] = 1, return 0
#             grad[i] = 0
    
#     return grad



# =============================

class GradientAscent:
    def __init__(self, l, k, n, eta=0.01, setting='two-stage'):
        # setting = 'one-stage' or 'two-stage'
        self.eta = eta
        self.l = l
        self.k = k
        self.n = n
        self.setting = setting
    
    def proj(self, z):
        return project(z, self.n, self.l)

    def grad(self, f, x):
        return gradient(f, x)

    def next(self, fs, x):
        ''' 
            Given the sequence of functions fs = f_1, ..., f_{t-1} and the last iterate y = y_{t-1}
            return the next iterate y_t
        '''
        if self.setting == 'one-stage':
            z = x + self.eta * fs[-1].grad(x)
        else:
            z = x + self.eta * gradient(fs[-1], x, self.k)
        
        x = self.proj(z)
        return x

class FTRL:
    def __init__(self, l, k, n, eta=0.01, regularizer='l2', setting='two-stage'):
        # regularizer = 'l2' or 'entropy'
        # setting = 'one-stage' or 'two-stage'
        self.eta = eta
        self.l = l
        self.k = k
        self.n = n
        self.regularizer = regularizer
        self.setting = setting
        self.grads = np.zeros(n)
        
        # Initialize the Gurobi model once
        self.model = gp.Model("FTRL")
        self.model.setParam("OutputFlag", 0)  # Suppress output for cleaner logs
        
        # Precompute piecewise linear approximation for entropy
        self.xs = [0.01 * i for i in range(101)]  # x-points for approximation
        self.ys = [p * np.log(p) if p != 0 else 0 for p in self.xs]  # y-points for approximation
        
        # Precompute variables and constraints
        self._initialize_model()

    def _initialize_model(self):
        # Decision variables
        self.x_vars = self.model.addVars(self.n, lb=0, ub=1, name="x")  # 0 <= x_i <= 1
        if self.regularizer == 'entropy':
            self.z_vars = self.model.addVars(self.n, lb=-gp.GRB.INFINITY, ub=0, name="z")  # Auxiliary variables for log(x)
            
            # Add piecewise linear constraints for the entropy regularizer
            for i in range(self.n):
                self.model.addGenConstrPWL(self.x_vars[i], self.z_vars[i], self.xs, self.ys, f"pwl_constraint_{i}")  # Approximate x_i * log(x_i)
        
        # Add sum constraint (sum(x) = l)
        self.model.addConstr(gp.quicksum(self.x_vars[i] for i in range(self.n)) == self.l, "SumConstraint")

    def next(self, fs, x):
        if self.setting == 'two-stage':
            self.grads += gradient(fs[-1], x, self.k)
        else:
            self.grads += fs[-1].grad(x)
        
        # Recalculate linear term and regularizer term
        linear_term = gp.quicksum(self.grads[i] * self.x_vars[i] for i in range(self.n))
        
        if self.regularizer == 'l2':
            # L2 regularization term
            regularizer_term = gp.quicksum(self.x_vars[i] * self.x_vars[i] for i in range(self.n))
        elif self.regularizer == 'entropy':
            # Entropy regularization using piecewise linear approximation
            regularizer_term = gp.quicksum(self.z_vars[i] for i in range(self.n))
        
        # Set objective function: linear_term - regularizer_term / (2 * eta)
        self.model.setObjective(linear_term - 1 / (2 * self.eta) * regularizer_term, gp.GRB.MAXIMIZE)

        # Warm-start: Set the initial values of decision variables
        for i in range(self.n):
            self.x_vars[i].start = x[i]  # Initialize with the previous solution

        # Optimize the model
        self.model.optimize()

        # Extract solution
        if self.model.status == gp.GRB.OPTIMAL:
            return np.array([self.x_vars[i].X for i in range(self.n)])
        else:
            raise RuntimeError("Optimization did not converge")


class OMA:
    pass
