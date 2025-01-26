import numpy as np
import pickle
from algorithms import GradientAscent, FTRL, Pipage, OptimalInHindsight, Balkanski, ReplacementGreedy, Random
from online_learning import OnlineLearning
from WTP import Potential, WTP
import random

random.seed(42)
np.random.seed(42)

settings = [{'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 1, 'sample': True}]
settings = [{'dataset': 'images', 'T': 250, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True}]
settings = [{'dataset': 'teamformation', 'l': 10, 'k': 4, 'eta': 10, 'sample': False}]
# settings = [{'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True}]
# settings = [{'dataset': 'influence', 'T': 100, 'l': 8, 'k': 3, 'eta': 10, 'sample': True}]


# settings = [{'dataset': 'teamformation', 'l': 10, 'k': 4, 'eta': 0.1, 'sample': False}, 
#             {'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True},
#             {'dataset': 'images', 'T': 200, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True}]
algorithms = ['FTRL-l2', 'FTRL-entropy', 'GA', 'one-stage GA', 'balkanski', 'repGreedy', 'OPT', 'Random']

etas = [0.0001, 0.001, 0.01, 0.1, 1, 10]

for setting in settings:
    dataset = setting['dataset']
    print(f'=== Running experiments for dataset = {dataset} ===')

    with open(f'./instances/{dataset}.pkl', 'rb') as file:
        print(f'dataset = {dataset}')
        wtps = pickle.load(file)

    # Input
    if setting['sample']:
        T = setting['T']
        fs = random.choices(wtps, k=T)
    else:
        fs = wtps
        T = len(fs)

    n = len(wtps[0].potentials[0].w)
    l = setting['l']
    k = setting['k']
    eta = setting['eta']

    # Initial solution
    x0 = np.zeros(n, dtype=int)
    x0[:l] = 1
    np.random.shuffle(x0)
    results = dict()
    if 'GA' in algorithms:
        print(f'Running Gradient Ascent')
        pipage = Pipage()
        rewards = []
        for eta in etas:
            print(f'eta = {eta}')
            ga = GradientAscent(l=l, k=k, n=n, eta=eta)
            OL = OnlineLearning(fs, x0, k, ga, pipage)
            OL.run()

            if sum(OL.int_rewards) > sum(rewards):
                rewards = OL.int_rewards
                results['frac_rewards_GA'] = OL.frac_rewards
                results['int_rewards_GA'] = OL.int_rewards
                results['frac_sol_GA'] = OL.xs[-1]
                results['int_sol_GA'] = OL.xs_int[-1]
                results['eta_GA'] = eta

    if 'one-stage GA' in algorithms:
        print(f'Running one-stage Gradient Ascent')
        pipage = Pipage()
        rewards = []
        for eta in etas:
            print(f'eta = {eta}')
            ga = GradientAscent(l=l, k=k, n=n, eta=eta, setting='one-stage')
            OL = OnlineLearning(fs, x0, k, ga, pipage)
            OL.run()
            if sum(OL.int_rewards) > sum(rewards):
                rewards = OL.int_rewards
                results['one_stage_frac_rewards_GA'] = OL.frac_rewards
                results['one_stage_int_rewards_GA'] = OL.int_rewards
                results['frac_sol_one_stage_GA'] = OL.xs[-1]
                results['int_sol_one_stage_GA'] = OL.xs_int[-1]
                results['eta_one_stage_GA'] = eta

    if 'FTRL-l2' in algorithms:
        print(f'Running FTRL with L2 regularizer')
        pipage = Pipage()
        rewards = []
        for eta in etas:
            print(f'eta = {eta}')
            ftrl = FTRL(l=l, k=k, n=n, eta=eta, regularizer='l2', setting='two-stage')
            OL = OnlineLearning(fs, x0, k, ftrl, pipage)
            OL.run()
            
            if sum(OL.int_rewards) > sum(rewards):
                rewards = OL.int_rewards
                results['frac_rewards_FTRL_l2'] = OL.frac_rewards
                results['int_rewards_FTRL_l2'] = OL.int_rewards
                results['frac_sol_FTRL_l2'] = OL.xs[-1]
                results['int_sol_FTRL_l2'] = OL.xs_int[-1]
                results['eta_FTRL_l2'] = eta

    if 'FTRL-entropy' in algorithms:
        print(f'Running FTRL with Entropic regularizer')
        pipage = Pipage()
        rewards = []
        for eta in etas:
            print(f'eta = {eta}')
            ftrl = FTRL(l=l, k=k, n=n, eta=eta, regularizer='entropy', setting='two-stage')
            OL = OnlineLearning(fs, x0, k, ftrl, pipage)
            OL.run()

            if sum(OL.int_rewards) > sum(rewards):
                rewards = OL.int_rewards
                results['frac_rewards_FTRL_entropy'] = OL.frac_rewards
                results['int_rewards_FTRL_entropy'] = OL.int_rewards
                results['frac_sol_FTRL_entropy'] = OL.xs[-1]
                results['int_sol_FTRL_entropy'] = OL.xs_int[-1]
                results['eta_FTRL_entropy'] = eta

    if 'Random' in algorithms:
        iterations = 10
        print(f'Running random for {iterations} iterations')
        rewards = np.zeros(len(fs))
        for t in range(iterations):
            randomAlg = Random(fs, n, l, k)
            vals = randomAlg.solve()
            rewards += np.array(vals)

        rewards /= iterations
        results['random'] = randomAlg.solve()


    # === Offline algorithms ===
    if 'OPT' in algorithms:
        print(f'Calculating optimal solution in hindsight')
        FracOpt = OptimalInHindsight(fs, l, k, 'fractional')
        IntOpt = OptimalInHindsight(fs, l, k, 'integral')

        fracSol, fracVal = FracOpt.solve()
        intSol, intVal = IntOpt.solve()

        results['frac_opt'] = fracVal
        results['int_opt'] = intVal
        results['frac_sol'] = fracSol
        results['int_sol'] = intSol


    if 'balkanski' in algorithms:
        print(f"Balkasnki's rounding algorithm")
        if not 'OPT' in algorithms:
            print(f'Before running balkasnki algorithm, run OPT.')
        else:
            balkanskiVal = 0
            iterations = 10
            print(f"Running Balkanski's algorithm {iterations} times.")
            for t in range(iterations):
                balkanski = Balkanski(fracSol, l, k, fs)
                sol, val = balkanski.round()
                balkanskiVal += val
            balkanskiVal /= iterations

            results[f'balkanski_val'] = balkanskiVal
            results['balkanski_sol'] = sol

    if 'repGreedy' in algorithms:
        print(f'Running Replacement-Greedy algorithm')
        repGreedy = ReplacementGreedy(fs, l, k, n)
        repGreedySol, repGreedyVal = repGreedy.solve()

        results[f'repGreedy_val'] = repGreedyVal
        results['repGreedy_sol'] = repGreedySol

    with open(f'./results/{dataset}.pkl', 'wb') as file:
        pickle.dump(results, file) 

    

    



