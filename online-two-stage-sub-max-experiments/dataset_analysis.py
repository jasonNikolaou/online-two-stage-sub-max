import numpy as np
import pickle
from algorithms import GradientAscent, FTRL, Pipage, OptimalInHindsight, Balkanski, ReplacementGreedy, Random
from online_learning import OnlineLearning
from WTP import Potential, WTP
import random
from gurobipy import Model, GRB, QuadExpr
import itertools
import matplotlib.pyplot as plt

settings = [{'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 1, 'sample': True}]
# settings = [{'dataset': 'images', 'T': 250, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True}]
# settings = [{'dataset': 'teamformation', 'l': 8, 'k': 4, 'eta': 10, 'sample': False}]
# settings = [{'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True}]
# settings = [{'dataset': 'influence', 'T': 100, 'l': 8, 'k': 3, 'eta': 10, 'sample': True}]
# settings = [{'dataset': 'coverage', 'T':100, 'l': 10, 'k': 1, 'eta': 0.1, 'sample': True}]
# settings = [{'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True}]

settings = [{'dataset': 'teamformation', 'l': 10, 'k': 4, 'eta': 0.1, 'sample': False}, 
            {'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True},
            {'dataset': 'images', 'T': 200, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True},
            {'dataset': 'influence', 'T': 100, 'l': 8, 'k': 3, 'eta': 10, 'sample': True},
            {'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True}
            ]
def b_stats(wtps):
    """
    Given a list of objects wtps,
    compute the min, max, and average of all b values of the potentials
    """
    b_values = [potential.b for wtp in wtps for potential in wtp.potentials]
    
    if not b_values:
        return None, None, None  # handle case where there are no potentials

    b_min = min(b_values)
    b_max = max(b_values)
    b_avg = sum(b_values) / len(b_values)
    
    return b_min, b_max, b_avg

def average_num_potentials(wtps):
    """
    Given a list of wtps, compute the average number of potentials.
    """
    total = sum(len(wtp.potentials) for wtp in wtps)
    return total / len(wtps) if wtps else 0.0

def jaccard_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the Jaccard distance between two k-hot vectors.
    Jaccard distance = 1 - (|A ∩ B| / |A ∪ B|)
    """
    intersection = np.sum(np.logical_and(vec1, vec2))
    union = np.sum(np.logical_or(vec1, vec2))
    if union == 0:
        return 0.0  # convention: distance between two empty sets is 0
    return 1 - intersection / union

def average_jaccard_distance(xs: np.ndarray) -> float:
    """
    Compute the average Jaccard distance between all pairs in xs.
    xs is an array of shape (num_vectors, vector_length) of k-hot vectors.
    """
    total_distance = 0.0
    num_pairs = 0
    for i, j in itertools.combinations(range(len(xs)), 2):
        total_distance += jaccard_distance(xs[i], xs[j])
        num_pairs += 1
    return total_distance / num_pairs if num_pairs > 0 else 0.0

def solve(f, k, setting='integral'):
        n = len(f.potentials[0].w) # dimensions

        # Create a Gurobi model
        model = Model("WTP_Optimization")
        model.setParam("OutputFlag", 0)  # Suppress Gurobi output

        # Decision variables x \in R^n
        if setting == 'fractional':
            x = model.addVars(n, lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f'x')
        else:
            x = model.addVars(n, vtype=GRB.BINARY, name=f'x')

        # Auxiliary variables for potential evaluations
        z = model.addVars(len(f.potentials), lb=0, name=f"z")
        
        # Objective: Maximize the WTP functions
        model.setObjective(
            sum(f.weights[i] * z[i] for i in range(len(f.potentials))), GRB.MAXIMIZE
        )
        
        # Constraints for each potential
        for i, potential in enumerate(f.potentials):
            # z_i <= b
            model.addConstr(z[i] <= potential.b, name=f"z_{i}_upper")
            # z_i <= w \cdot x
            model.addConstr(
                z[i] <= sum(potential.w[j] * x[j] for j in range(n)),
                name=f"z_{i}_dot"
            )
        
        # Cardinality constraint for x
        model.addConstr(sum(x[i] for i in range(n)) <= k, name=f"cardinality_x")

        # Optimize the model
        model.optimize()
        
        # Extract the solution
        if model.status == GRB.OPTIMAL:
            x_vals = np.array([x[i].x for i in range(n)])
            return x_vals #, model.objVal
        else:
            raise ValueError("Optimization failed. Status: " + str(model.status))

for setting in settings:
    dataset = setting['dataset']
    print(f'\n=== Running analysis for dataset = {dataset} ===')

    k = setting['k']
    l = setting['l']
    with open(f'./instances/{dataset}.pkl', 'rb') as file:
        print(f'dataset = {dataset}')
        wtps = pickle.load(file)
    
    print(f'number of WTPs m = {len(wtps)}')
    print(f'dimension of the action space n = {len(wtps[0].potentials[0].w)}')
    
    avg_jaccard_list = []

    for k in range(1, l + 1):
        xs = []
        for wtp in wtps:
            x_sol = solve(wtp, k, 'integral')
            xs.append(x_sol)
        avg_jaccard = average_jaccard_distance(xs)
        avg_jaccard_list.append(avg_jaccard)
        print(f'JD for k={k}: {avg_jaccard}')

    # Pickle the list
    filename = f'./plots/analysis/{dataset}/k_vs_JD.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(avg_jaccard_list, f)

    print(f"Saved average Jaccard distances to {filename}")
    
    # Plot k vs avg_JD(k)
    ks = np.arange(1, len(avg_jaccard_list) + 1)

    plt.plot(ks, avg_jaccard_list, marker='o')
    plt.title('k vs Average Jaccard Distance of Optimal Solutions')
    plt.xlabel('k')
    plt.ylabel('Average Jaccard Distance')
    plt.grid(True)
    plt.savefig(f'./plots/analysis/{dataset}/JD.pdf', bbox_inches='tight')
    plt.close()  # Close the plot to avoid display

    avg_potentials = np.round(average_num_potentials(wtps), 2)
    print(f'Average number of potentials |C| = {avg_potentials}')

    b_min, b_max, b_avg = b_stats(wtps)
    print(f'b_min = {b_min}, b_max = {b_max}, b_avg = {b_avg}')

    # Distribution of weights of the potentials
    all_weights = [weight for wtp in wtps for weight in wtp.weights]

    # Now you can plot separately
    plt.hist(all_weights, bins=10, edgecolor='black')
    plt.title("Distribution of Potential Weights")
    plt.xlabel("Weight")
    plt.ylabel("# Potentials")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'./plots/analysis/{dataset}/wtp_weights_dist.pdf', bbox_inches='tight')
    plt.close()  # Close the plot to avoid display

    # You can also compute stats separately
    w_min = np.round(min(all_weights), 2)
    w_max = np.round(max(all_weights), 2)
    w_avg = np.round(sum(all_weights) / len(all_weights), 2)

    print(f"Min weight: {w_min}, Max weight: {w_max}, Average weight: {w_avg}")
    