import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import matplotlib.cm as cm

settings = [
    {'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'},
    {'dataset': 'images', 'title': 'Image summarization'},
    {'dataset': 'teamformation', 'title': 'Team formation'},
    {'dataset': 'influence', 'title': 'Influence maximization'},
    {'dataset': 'movies', 'title': 'Movie recommendation'},
    {'dataset': 'coverage', 'title': 'Coverage'}
]

algorithms = ['GA_int_rewards', 'one_stage_GA_int_rewards', 'OPT_int_val']
# algorithms = ['OPT_int_val']
alg_names = {
    'GA_int_rewards': 'RAOCO-OGA',
    'one_stage_GA_int_rewards': '1S-OGA',
    'OPT_int_val': 'OPT'
}

cmap = cm.get_cmap('viridis', len(algorithms))
colors = [cmap(i) for i in range(len(algorithms))]
markers = ['o', 's', '^']

for setting in settings:
    dataset = setting['dataset']
    title = setting['title']

    with open(f'./results/{dataset}_tradeoff_l.pkl', 'rb') as file:
        data = pickle.load(file)

    T = len(data[1]['GA_int_rewards'][0])
    
    ls = sorted(data.keys())

    plt.figure(figsize=(10, 6))

    for i, algo in enumerate(algorithms):
        avg_cum_rewards = []
        std_cum_rewards = []

        for l in ls:
            rewards = data[l][algo]
            if algo == 'OPT_int_val':
                # List of scalar rewards
                rewards = np.array(rewards)
                mean_cum = np.mean(rewards)
                std_cum = np.std(rewards)
            else:
                # List of arrays → compute cumulative sum then take final value
                reward_runs = np.array([np.cumsum(run) / T for run in rewards])  # shape: (num_runs, T)
                final_rewards = reward_runs[:, -1]  # final cumulative reward
                mean_cum = np.mean(final_rewards)
                std_cum = np.std(final_rewards)

            avg_cum_rewards.append(mean_cum)
            std_cum_rewards.append(std_cum)

        avg_cum_rewards = np.array(avg_cum_rewards)
        std_cum_rewards = np.array(std_cum_rewards)
        

        plt.plot(
            ls,
            avg_cum_rewards,
            label=alg_names[algo],
            marker=markers[i],
            color=colors[i],
            linewidth=2
        )
        plt.fill_between(
            ls,
            avg_cum_rewards - std_cum_rewards,
            avg_cum_rewards + std_cum_rewards,
            color=colors[i],
            alpha=0.3
        )

    plt.xlabel('l', fontsize=22)
    plt.ylabel('$C_T$', fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.tight_layout()

    output_dir = f'plots/analysis/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/lineplot_l_{dataset}.pdf', bbox_inches='tight')
    plt.close()
