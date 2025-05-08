import numpy as np
import matplotlib.pyplot as plt
import pickle

settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'},
            {'dataset': 'images', 'title': 'Image summarization'},
            {'dataset': 'teamformation', 'title': 'Team formation'},
            {'dataset': 'influence', 'title': 'Influence maximization'},
            {'dataset': 'movies', 'title': 'Movie recommendation'},
            {'dataset': 'coverage', 'title': 'Coverage'}]

# settings = [{'dataset': 'images', 'title': 'Image summarization'}]
# settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'}]
settings = [{'dataset': 'teamformation', 'title': 'Team formation'}]
# settings = [{'dataset': 'movies', 'title': 'Movie recommendation'}]
# settings = [{'dataset': 'influence', 'title': 'Influence maximization'}]
# settings = [{'dataset': 'coverage', 'title': 'Coverage'}]



cmap = plt.get_cmap("viridis")  # colorblind palette
marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']  # Unique markers for each line

for setting in settings:
    print(f"Dataset = {setting['dataset']}")
    dataset = setting['dataset']
    title = setting['title']
    with open(f'./results/{dataset}.pkl', 'rb') as file:
        data = pickle.load(file)

    n_lines = 0
    # Retrieve the rewards array from the dictionary
    if 'frac_rewards_GA' in data[0]:
        frac_rewards_GA = [res["frac_rewards_GA"] for res in data]
        int_rewards_GA = [res["int_rewards_GA"] for res in data]
        etas_GA = [res['eta_GA'] for res in data]
        print(f'eta GA = {etas_GA}')
        T = len(frac_rewards_GA[0])
        n_lines += 1

    if 'frac_rewards_FTRL_l2' in data[0]:
        frac_rewards_FTRL_l2 = [res["frac_rewards_FTRL_l2"] for res in data]
        int_rewards_FTRL_l2 = [res["int_rewards_FTRL_l2"] for res in data]
        etas_FTRL_l2 = [res['eta_FTRL_l2'] for res in data]
        print(f'eta FTRL l2 = {etas_FTRL_l2}')
        n_lines += 1

    if 'frac_rewards_FTRL_entropy' in data[0]:   
        frac_rewards_FTRL_entropy = [res["frac_rewards_FTRL_entropy"] for res in data]
        int_rewards_FTRL_entropy = [res["int_rewards_FTRL_entropy"] for res in data]
        etas_FTRL_entropy = [res['eta_FTRL_entropy'] for res in data]
        print(f'eta FTRL entropy = {etas_FTRL_entropy}')
        n_lines += 1
        
    if 'frac_rewards_one_stage_GA' in data[0]:
        frac_rewards_one_stage_GA = [res['frac_rewards_one_stage_GA'] for res in data]
        int_rewards_one_stage_GA = [res['int_rewards_one_stage_GA'] for res in data]
        etas_one_stage_GA = [res['eta_one_stage_GA'] for res in data]
        print(f'eta one-stage GA = {etas_one_stage_GA}')
        n_lines += 1

    if 'int_rewards_all_random' in data[0]:
        int_rewards_all_random = [res['int_rewards_all_random'] for res in data]
        n_lines += 1

    if 'frac_opt' in data[0]:
        frac_opt = [res['frac_opt'] for res in data]
        int_opt = [res['int_opt'] for res in data]
        n_lines += 1

    if 'int_rewards_all_balkanski' in data[0]:
        int_rewards_all_balkanski = [res[f'int_rewards_all_balkanski'] for res in data]
        n_lines += 1

    if 'int_rewards_repGreedy' in data[0]:
        int_rewards_repGreedy = [res['int_rewards_repGreedy'] for res in data]
        n_lines += 1

    timesteps = np.arange(1, T+1)

    def compute_avg_cum_and_std(rewards_all):
        """
        Args:
            rewards_all:
                - A 2D array of shape (N, T)
                - OR a list of 2D arrays of shape (N_i, T)
                - OR a list of 1D arrays of shape (T,) to be stacked into (N, T)
        Returns:
            avg_cum: array of shape (T,)
            std_cum: array of shape (T,)
        """
        # If it's a list, validate elements and stack
        if isinstance(rewards_all, list):
            processed = []
            for r in rewards_all:
                r = np.array(r)
                if r.ndim == 1:
                    r = r.reshape(1, -1)  # shape (1, T)
                elif r.ndim != 2:
                    raise ValueError(f"List element has unexpected shape: {r.shape}")
                processed.append(r)
            rewards_all = np.concatenate(processed, axis=0)  # shape (N_total, T)

        elif isinstance(rewards_all, np.ndarray):
            if rewards_all.ndim == 1:
                rewards_all = rewards_all.reshape(1, -1)
            elif rewards_all.ndim != 2:
                raise ValueError(f"Expected 2D array, got shape {rewards_all.shape}")
        else:
            raise TypeError("rewards_all must be a list or a numpy array")

        # Now guaranteed to be (N_total, T)
        cum_rewards = np.cumsum(rewards_all, axis=1)
        T = rewards_all.shape[1]
        timesteps = np.arange(1, T + 1).reshape(1, -1)

        avg_cum = np.mean(cum_rewards / timesteps, axis=0)
        std_cum = np.std(cum_rewards / timesteps, axis=0)

        return avg_cum, std_cum

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_idx = 0

    if 'int_rewards_GA' in data[0]:
        avg_cum_int_GA, std_cum_GA = compute_avg_cum_and_std(int_rewards_GA)
        ax.plot(avg_cum_int_GA, label='RAOCO-OGA', color=cmap(plot_idx / (n_lines - 1)), marker=marker_styles[plot_idx], markevery=int(T / 10))
        ax.fill_between(timesteps, avg_cum_int_GA - std_cum_GA, avg_cum_int_GA + std_cum_GA, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    if 'int_rewards_FTRL_l2' in data[0]:
        avg_cum_FTRL_l2, std_cum_FTRL_l2 = compute_avg_cum_and_std(int_rewards_FTRL_l2)
        ax.plot(avg_cum_FTRL_l2, label='RAOCO-FTRL-$L_2$', color=cmap(plot_idx / (n_lines - 1)), marker=marker_styles[plot_idx], markevery=int(T / 10))
        ax.fill_between(timesteps, avg_cum_FTRL_l2 - std_cum_FTRL_l2, avg_cum_FTRL_l2 + std_cum_FTRL_l2, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    if 'int_rewards_FTRL_entropy' in data[0]:
        avg_cum_FTRL_entropy, std_cum_FTRL_entropy = compute_avg_cum_and_std(int_rewards_FTRL_entropy)
        ax.plot(avg_cum_FTRL_entropy, label='RAOCO-FTRL-H', color=cmap(plot_idx / (n_lines - 1)), marker=marker_styles[plot_idx], markevery=int(T / 10))
        ax.fill_between(timesteps, avg_cum_FTRL_entropy - std_cum_FTRL_entropy, avg_cum_FTRL_entropy + std_cum_FTRL_entropy, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    if 'int_rewards_one_stage_GA' in data[0]:
        avg_cum_1S, std_cum_1S = compute_avg_cum_and_std(int_rewards_one_stage_GA)
        ax.plot(avg_cum_1S, label='1S-OGA', color=cmap(plot_idx / (n_lines - 1)), marker=marker_styles[plot_idx], markevery=int(T / 10))
        ax.fill_between(timesteps, avg_cum_1S - std_cum_1S, avg_cum_1S + std_cum_1S, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    if 'int_rewards_all_random' in data[0]:
        avg_cum_rand, std_cum_rand = compute_avg_cum_and_std(int_rewards_all_random)
        ax.plot(avg_cum_rand, label='Random', color=cmap(plot_idx / (n_lines - 1)), marker=marker_styles[plot_idx], markevery=int(T / 10))
        ax.fill_between(timesteps, avg_cum_rand - std_cum_rand, avg_cum_rand + std_cum_rand, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    if 'int_rewards_all_balkanski' in data[0]:
        int_rewards_all_balkanski = [[val/T] * T for run in int_rewards_all_balkanski for val in run]
        avg_cum_balk, std_cum_balk = compute_avg_cum_and_std(int_rewards_all_balkanski)
        ax.plot(avg_cum_balk, label='OFLN - CO', color=cmap(plot_idx / (n_lines - 1)), linestyle='-.', linewidth=1.5,
                marker=marker_styles[plot_idx], markevery=int(T/10))
        ax.fill_between(timesteps, avg_cum_balk - std_cum_balk, avg_cum_balk + std_cum_balk, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    # Fixed baseline
    if 'int_opt' in data[0]:
        int_rewards_opt = [[val/T] * T for val in int_opt]
        avg_cum_opt, std_cum_opt = compute_avg_cum_and_std(int_rewards_opt)
        ax.plot(avg_cum_opt, label='OFLN - OPT', color=cmap(plot_idx / (n_lines - 1)), linestyle='-.', linewidth=1.5,
                marker=marker_styles[plot_idx], markevery=int(T/10))
        ax.fill_between(timesteps, avg_cum_opt - std_cum_opt, avg_cum_opt + std_cum_opt, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    if 'int_rewards_repGreedy' in data[0]:
        int_rewards_repGreedy = [[val/T] * T for val in int_rewards_repGreedy]
        avg_cum_repGreedy, std_cum_repGreedy = compute_avg_cum_and_std(int_rewards_repGreedy)
        ax.plot(avg_cum_repGreedy, label='OFLN - RGR', color=cmap(plot_idx / (n_lines - 1)), linestyle='-.', linewidth=1.5,
                marker=marker_styles[plot_idx], markevery=int(T/10))
        ax.fill_between(timesteps, avg_cum_repGreedy - std_cum_repGreedy, avg_cum_repGreedy + std_cum_repGreedy, color=cmap(plot_idx / (n_lines - 1)), alpha=0.2)
        plot_idx += 1

    ax.set_xlabel("Time Steps", fontsize=16)
    ax.set_ylabel("$C_t$", fontsize=16)
    ax.legend(loc='lower left')
    # ax.legend()
    ax.grid(True)
    fig.savefig(f'./plots/{dataset}.pdf', bbox_inches='tight')
    plt.close(fig)
