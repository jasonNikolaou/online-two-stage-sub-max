import numpy as np
import matplotlib.pyplot as plt
import pickle

settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'},
            {'dataset': 'images', 'title': 'Image summarization'},
            {'dataset': 'teamformation', 'title': 'Team formation'},
            {'dataset': 'influence', 'title': 'Influence maximization'},
            {'dataset': 'movies', 'title': 'Movie recommendation'}]

# settings = [{'dataset': 'images', 'title': 'Image summarization'}]
# settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'}]
# settings = [{'dataset': 'teamformation', 'title': 'Team formation'}]
# settings = [{'dataset': 'movies', 'title': 'Movie recommendation'}]
# settings = [{'dataset': 'influence', 'title': 'Influence maximization'}]



cmap = plt.get_cmap("viridis")  # colorblind palette
marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']  # Unique markers for each line

for setting in settings:
    dataset = setting['dataset']
    title = setting['title']
    with open(f'./results/{dataset}_tradeoff.pkl', 'rb') as file:
        data = pickle.load(file)

    # algorithms = ['frac_rewards_GA', 'one_stage_frac_rewards_GA', 'random', 'int_opt', 'balkanski_val', 'repGreedy_val']
    algorithms = ['frac_rewards_GA', 'one_stage_frac_rewards_GA']
    alg_names = {'frac_rewards_GA': 'GA', 'one_stage_frac_rewards_GA': 'one-stage-GA'}
    colors = {algo: cmap(i / (len(algorithms) - 1)) for i, algo in enumerate(algorithms)}

    ks = sorted(data.keys())
    marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
    markers = {algo: marker_styles[i % len(marker_styles)] for i, algo in enumerate(algorithms)}

    plt.figure(figsize=(12, 7))

    # Generate one plot per k
    for k in ks:
        plt.figure(figsize=(10, 6))
        
        for algo in algorithms:
            rewards = data[k][algo]
            timesteps = np.array(range(len(rewards)))
            cum_avg_rewards = np.cumsum(rewards) / (timesteps + 1)
            plt.plot(
                timesteps,
                cum_avg_rewards,
                label=algo,
                color=colors[algo],
                marker=markers[algo],
                markevery=max(1, len(rewards) // 10),
                linewidth=1.5
            )
        
        plt.xlabel('Time')
        plt.ylabel('Reward')
        plt.title(f'Reward over Time (k = {k})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # Ensure the output directory exists
        output_dir = f'plots/analysis/{dataset}'

        # Save the figure
        filename = f'{output_dir}/rewards_k{k}.pdf'
        plt.savefig(filename, bbox_inches='tight')
        plt.close()


    # Plot cumulative reward of each algorithm vs avg jaccard distance (k varies)
    jd_filename = f'./plots/analysis/{dataset}/k_vs_JD.pkl'

    # Load avg_jaccard_list
    with open(jd_filename, 'rb') as f:
        avg_jaccard_list = pickle.load(f)
    
    algo_points = {algo: [] for algo in algorithms}

    for k in ks:
        jd = avg_jaccard_list[k - 1]
        for algo in algorithms:
            avg_reward = sum(data[k][algo]) / len(data[k][algo])
            algo_points[algo].append((jd, avg_reward))
    
    plt.figure(figsize=(8,6))
    for idx, algo in enumerate(algorithms):
        points = np.array(algo_points[algo])
        x = points[:, 0]  # avg_jaccard distances
        y = points[:, 1]  # rewards
        plt.plot(x, y, marker='o', label=alg_names[algo], color=colors[algo])
        
        # Annotate each point with k
        for i, (x_coord, y_coord) in enumerate(zip(x, y)):
            k_value = i + 1  # because k starts from 1
            plt.text(x_coord, y_coord, str(k_value), fontsize=8, ha='right', va='bottom')

    plt.xlabel('Average Jaccard Distance')
    plt.ylabel('Average Reward')
    plt.title(f'Average Reward vs Average Jaccard Distance ({dataset})')
    plt.legend()
    plt.grid(True)
    # plt.tight_layout()
    filename = f'{output_dir}/reward_vs_JD.pdf'
    plt.savefig(filename)
    plt.close()


    
    

    
