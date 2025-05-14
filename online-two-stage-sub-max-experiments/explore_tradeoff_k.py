import numpy as np
import pickle
import random
from collections import defaultdict
from algorithms import GradientAscent, FTRL, Pipage, OptimalInHindsight, Balkanski, ReplacementGreedy, Random
from online_learning import OnlineLearning
from WTP import Potential, WTP

random.seed(42)
np.random.seed(42)

settings = [
    {'dataset': 'teamformation', 'l': 10, 'k': 4, 'eta': 10, 'sample': False},
    {'dataset': 'coverage', 'T': 100, 'l': 20, 'k': 1, 'eta': 0.1, 'sample': False},
    {'dataset': 'influence', 'T': 100, 'l': 8, 'k': 3, 'eta': 10, 'sample': False},
    {'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True},
    {'dataset': 'images', 'T': 200, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True},
    {'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True}
]

# settings = [{'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True}]
# settings = [{'dataset': 'images', 'T': 200, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True}]
# settings = [{'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True}]

algorithms = ['GA', 'one-stage GA', 'OPT']
etas = [0.001, 0.01, 0.1, 1]
runs = 5

for setting in settings:
    dataset = setting['dataset']
    print(f'=== Running experiments for dataset = {dataset} ===')

    with open(f'./instances/{dataset}.pkl', 'rb') as file:
        wtps = pickle.load(file)

    n = len(wtps[0].potentials[0].w)
    l = setting['l']
    ks = range(1, l+1) if l <= 10 else range(1, l+1, 2)

    results = defaultdict(lambda: defaultdict(list))

    for run in range(runs):
        print(f'\n=== Run {run + 1}/{runs} ===')

        # Sample function sequence once per run
        if setting['sample']:
            T = 4 * len(wtps)
            fs = random.choices(wtps, k=T)
        else:
            fs = wtps
            T = len(fs)

        # Initial x0 used across all k values in this run
        x0 = np.zeros(n, dtype=int)
        x0[:l] = 1
        np.random.shuffle(x0)

        for k in ks:
            print(f'--- k = {k} ---')

            if 'GA' in algorithms:
                pipage = Pipage()
                best_rewards = []
                for eta in etas:
                    ga = GradientAscent(l=l, k=k, n=n, eta=eta)
                    OL = OnlineLearning(fs, x0.copy(), k, ga, pipage)
                    OL.run()
                    if sum(OL.int_rewards) > sum(best_rewards):
                        best_rewards = OL.int_rewards
                        results[k]['GA_int_rewards'].append(OL.int_rewards)
                        results[k]['GA_frac_rewards'].append(OL.frac_rewards)
                        results[k]['GA_frac_sol'].append(OL.xs[-1])
                        results[k]['GA_int_sol'].append(OL.xs_int[-1])
                        results[k]['GA_eta'].append(eta)

            if 'one-stage GA' in algorithms:
                pipage = Pipage()
                best_rewards = []
                for eta in etas:
                    ga = GradientAscent(l=l, k=k, n=n, eta=eta, setting='one-stage')
                    OL = OnlineLearning(fs, x0.copy(), k, ga, pipage)
                    OL.run()
                    if sum(OL.int_rewards) > sum(best_rewards):
                        best_rewards = OL.int_rewards
                        results[k]['one_stage_GA_int_rewards'].append(OL.int_rewards)
                        results[k]['one_stage_GA_frac_rewards'].append(OL.frac_rewards)
                        results[k]['one_stage_GA_frac_sol'].append(OL.xs[-1])
                        results[k]['one_stage_GA_int_sol'].append(OL.xs_int[-1])
                        results[k]['one_stage_GA_eta'].append(eta)

            if 'OPT' in algorithms:
                print(f'Computing OPT for run {run + 1}')
                FracOpt = OptimalInHindsight(fs, l, k, 'fractional')
                IntOpt = OptimalInHindsight(fs, l, k, 'integral')
                fracSol, fracVal = FracOpt.solve()
                intSol, intVal = IntOpt.solve()
                results[k]['OPT_frac_val'].append(fracVal / T)
                results[k]['OPT_int_val'].append(intVal / T)
                results[k]['OPT_frac_sol'].append(fracSol)
                results[k]['OPT_int_sol'].append(intSol)

        # Save run results
        with open(f'./results/{dataset}_{run}_tradeoff_k.pkl', 'wb') as file:
            pickle.dump(dict(results), file)
            
    # Save full results
    with open(f'./results/{dataset}_tradeoff_k.pkl', 'wb') as file:
        pickle.dump(dict(results), file)
