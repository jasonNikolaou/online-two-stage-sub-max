import numpy as np
import pickle
from algorithms import GradientAscent, FTRL, Pipage, OptimalInHindsight, Balkanski, ReplacementGreedy, Random
from online_learning import OnlineLearning
from WTP import Potential, WTP
import random

random.seed(42)
np.random.seed(42)

# settings = [{'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 1, 'sample': True}]
# settings = [{'dataset': 'images', 'T': 250, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True}]
settings = [{'dataset': 'teamformation', 'l': 8, 'k': 4, 'eta': 10, 'sample': False}]
# settings = [{'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True}]
# settings = [{'dataset': 'influence', 'T': 100, 'l': 8, 'k': 3, 'eta': 10, 'sample': False}]
# settings = [{'dataset': 'coverage', 'T':100, 'l': 20, 'k': 1, 'eta': 0.1, 'sample': False}]


# settings = [{'dataset': 'wikipedia', 'T': 100, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True},
#             {'dataset': 'images', 'T': 200, 'l': 20, 'k': 5, 'eta': 0.1, 'sample': True},
#             {'dataset': 'influence', 'T': 100, 'l': 8, 'k': 3, 'eta': 10, 'sample': True},
#             {'dataset': 'movies', 'T': 50, 'l': 10, 'k': 3, 'eta': 10, 'sample': True},
#             {'dataset': 'coverage', 'T':100, 'l': 20, 'k': 1, 'eta': 0.1, 'sample': False}]

algorithms = ['FTRL-l2', 'FTRL-entropy', 'GA', 'one-stage GA', 'balkanski', 'repGreedy', 'OPT', 'Random']
# algorithms = ['GA', 'one-stage GA', 'balkanski', 'repGreedy', 'OPT', 'Random']
etas = [0.001, 0.01, 0.1, 1, 10]
# etas = [0.01]

for setting in settings:
    dataset = setting['dataset']
    print(f'=== Running experiments for dataset = {dataset} ===')

    with open(f'./instances/{dataset}.pkl', 'rb') as file:
        wtps = pickle.load(file)

    num_repeats = 5
    all_results = []
    best_etas = {}  # Store best eta per algorithm

    for repeat in range(num_repeats):
        print(f'--- Repeat {repeat+1}/{num_repeats} ---')
        if setting['sample']:
            T = 4 * len(wtps)
            fs = random.choices(wtps, k=T)
        else:
            fs = wtps
            T = len(fs)

        n = len(wtps[0].potentials[0].w)
        l = setting['l']
        k = setting['k']

        x0 = np.ones(n, dtype=int)
        x0 = x0 * k / n
        np.random.shuffle(x0)

        results = dict()
        pipage = Pipage()

        def run_with_eta_selection(name, constructor):
            rewards = []
            if repeat == 0:
                etas_ = etas
                if dataset == 'movies':
                    etas_ = [0.001, 0.01, 1]
                elif dataset == 'wikipedia':
                    etas_ = [0.1, 1]
                elif dataset == 'influence':
                    etas_ = [0.1]

                for eta_val in etas_:
                    print(f'{name}: eta = {eta_val}')
                    learner = constructor(eta_val)
                    OL = OnlineLearning(fs, x0, k, learner, pipage)
                    OL.run()
                    if sum(OL.frac_rewards) > sum(rewards):
                        rewards[:] = OL.frac_rewards
                        results[f'frac_rewards_{name}'] = OL.frac_rewards
                        results[f'int_rewards_{name}'] = OL.int_rewards
                        results[f'frac_sol_{name}'] = OL.xs
                        results[f'int_sol_{name}'] = OL.xs_int
                        results[f'eta_{name}'] = eta_val
                        best_etas[name] = eta_val
            else:
                eta_val = best_etas[name]
                print(f'{name}: using best eta = {eta_val}')
                learner = constructor(eta_val)
                OL = OnlineLearning(fs, x0, k, learner, pipage)
                OL.run()
                results[f'frac_rewards_{name}'] = OL.frac_rewards
                results[f'int_rewards_{name}'] = OL.int_rewards
                results[f'frac_sol_{name}'] = OL.xs
                results[f'int_sol_{name}'] = OL.xs_int
                results[f'eta_{name}'] = eta_val

        if 'GA' in algorithms:
            run_with_eta_selection('GA', lambda eta, *_: GradientAscent(l=l, k=k, n=n, eta=eta, setting='two-stage'))

        if 'one-stage GA' in algorithms:
            run_with_eta_selection('one_stage_GA', lambda eta, *_: GradientAscent(l=l, k=k, n=n, eta=eta, setting='one-stage'))

        if 'FTRL-l2' in algorithms:
            run_with_eta_selection('FTRL_l2', lambda eta, *_: FTRL(l=l, k=k, n=n, eta=eta, regularizer='l2', setting='two-stage'))

        if 'FTRL-entropy' in algorithms:
            run_with_eta_selection('FTRL_entropy', lambda eta, *_: FTRL(l=l, k=k, n=n, eta=eta, regularizer='entropy', setting='two-stage'))

        if 'Random' in algorithms:
            iterations = 5
            print(f'Running Random for {iterations} iterations')
            results['int_rewards_all_random'] = []
            for _ in range(iterations):
                alg = Random(fs, n, l, k)
                rewards = alg.solve()
                results['int_rewards_all_random'].append(rewards)

        if 'OPT' in algorithms:
            print(f'Calculating OPT')
            FracOpt = OptimalInHindsight(fs, l, k, 'fractional')
            IntOpt = OptimalInHindsight(fs, l, k, 'integral')
            fracSol, fracVal = FracOpt.solve()
            intSol, intVal = IntOpt.solve()
            results['frac_opt'] = fracVal
            results['int_opt'] = intVal
            results['frac_sol'] = fracSol
            results['int_sol'] = intSol

        if 'balkanski' in algorithms and 'frac_sol' in results:
            print(f'Running Balkanski')
            vals = []
            for _ in range(5):
                balk = Balkanski(results['frac_sol'], l, k, fs)
                _, val = balk.round()
                vals.append(val)
            results['int_rewards_all_balkanski'] = vals

        if 'repGreedy' in algorithms:
            print(f'Running ReplacementGreedy')
            rep = ReplacementGreedy(fs, l, k, n)
            sol, val = rep.solve()
            results['int_rewards_repGreedy'] = val
            results['repGreedy_sol'] = sol

        all_results.append(results)

    with open(f'./results/{dataset}.pkl', 'wb') as f:
        pickle.dump(all_results, f)

    

    



