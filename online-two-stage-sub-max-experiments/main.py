import numpy as np
import pickle
from algorithms import GradientAscent, FTRL, Pipage, OptimalInHindsight, Balkanski, ReplacementGreedy
from online_learning import OnlineLearning
from WTP import Potential, WTP
import random

random.seed(42)

settings = [{'dataset': 'wikipedia', 'T': 100, 'l': 5, 'k': 2}]
algorithms = {'FTRL-l2', 'FTRL-entropy', 'GA', 'one-stage GA', 'balkanski', 'repGreedy', 'OPT'}

for setting in settings:
    dataset = setting['dataset']
    print(f'=== Running experiments for dataset = {dataset} ===')
    # Reading the list back from the file
    with open(f'./instances/{dataset}.pkl', 'rb') as file:
        wtps = pickle.load(file)

    # Input
    T = setting['T']
    fs = random.choices(wtps, k=T)
    n = len(wtps[0].potentials[0].w)
    l = setting['l']
    k = setting['k']
    eta = 0.1

    # Initial solution
    x0 = np.zeros(n, dtype=int)
    x0[:l] = 1
    np.random.shuffle(x0)
    results = dict()
    if 'GA' in algorithms:
        print(f'Running Gradient Ascent')
        pipage = Pipage()
        ga = GradientAscent(l=l, k=k, n=n, eta=eta)
        OL = OnlineLearning(fs, x0, k, ga, pipage)
        OL.run()

        results['frac_rewards_GA'] = OL.frac_rewards
        results['int_rewards_GA'] = OL.int_rewards
        results['frac_sol_GA'] = OL.xs[-1]
        results['int_sol_GA'] = OL.xs_int[-1]

    if 'one-stage GA' in algorithms:
        print(f'Running one-stage Gradient Ascent')
        pipage = Pipage()
        ga = GradientAscent(l=l, k=k, n=n, eta=eta, setting='one-stage')
        OL = OnlineLearning(fs, x0, k, ga, pipage)
        OL.run()

        results['one_stage_frac_rewards_GA'] = OL.frac_rewards
        results['one_stage_int_rewards_GA'] = OL.int_rewards
        results['frac_sol_one_stage_GA'] = OL.xs[-1]
        results['int_sol_one_stage_GA'] = OL.xs_int[-1]

    if 'FTRL-l2' in algorithms:
        print(f'Running FTRL with L2 regularizer')
        pipage = Pipage()
        ftrl = FTRL(l=l, k=k, n=n, eta=eta, regularizer='l2', setting='two-stage')
        OL = OnlineLearning(fs, x0, k, ftrl, pipage)
        OL.run()

        results['frac_rewards_FTRL_l2'] = OL.frac_rewards
        results['int_rewards_FTRL_l2'] = OL.int_rewards
        results['frac_sol_FTRL_l2'] = OL.xs[-1]
        results['int_sol_FTRL_l2'] = OL.xs_int[-1]

    if 'FTRL-entropy' in algorithms:
        print(f'Running FTRL with Entropic regularizer')
        pipage = Pipage()
        ftrl = FTRL(l=l, k=k, n=n, eta=eta, regularizer='entropy', setting='two-stage')
        OL = OnlineLearning(fs, x0, k, ftrl, pipage)
        OL.run()

        results['frac_rewards_FTRL_entropy'] = OL.frac_rewards
        results['int_rewards_FTRL_entropy'] = OL.int_rewards
        results['frac_sol_FTRL_entropy'] = OL.xs[-1]
        results['int_sol_FTRL_entropy'] = OL.xs_int[-1]

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
            iterations = 100
            print(f"Running Balkanski's algorithm {iterations} times.")
            for t in range(T):
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

    with open('./results/wikipedia.pkl', 'wb') as file:
        pickle.dump(results, file) 



    



