import numpy as np
import matplotlib.pyplot as plt
import pickle

settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'},
            {'dataset': 'images', 'title': 'Image summarization'}]
# settings = [{'dataset': 'images', 'title': 'Image summarization'}]
# settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'}]

cmap = plt.get_cmap("viridis")  # colorblind palette
marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']  # Unique markers for each line

for setting in settings:
    dataset = setting['dataset']
    title = setting['title']
    with open(f'./results/{dataset}.pkl', 'rb') as file:
        data = pickle.load(file)

    # Retrieve the rewards array from the dictionary
    frac_rewards_GA = data["frac_rewards_GA"]
    int_rewards_GA = data["int_rewards_GA"]
    frac_rewards_FTRL_l2 = data["frac_rewards_FTRL_l2"]
    int_rewards_FTRL_l2 = data["int_rewards_FTRL_l2"]
    frac_rewards_FTRL_entropy = data["frac_rewards_FTRL_entropy"]
    int_rewards_FTRL_entropy = data["int_rewards_FTRL_entropy"]

    frac_rewards_one_stage_GA = data['one_stage_frac_rewards_GA']
    int_rewards_one_stage_GA = data['one_stage_int_rewards_GA']

    rewards_random = data['random']

    frac_opt = data['frac_opt']
    int_opt = data['int_opt']

    balkanskiVal = data[f'balkanski_val']

    repGreedyVal = data['repGreedy_val']

    # Compute cumulative sum of rewards
    cum_frac_GA = np.cumsum(frac_rewards_GA)
    cum_int_GA = np.cumsum(int_rewards_GA)
    cum_frac_FTRL_l2 = np.cumsum(frac_rewards_FTRL_l2)
    cum_int_FTRL_l2 = np.cumsum(int_rewards_FTRL_l2)
    cum_frac_FTRL_entropy = np.cumsum(frac_rewards_FTRL_entropy)
    cum_int_FTRL_entropy = np.cumsum(int_rewards_FTRL_entropy)

    cum_frac_one_stage_GA = np.cumsum(frac_rewards_one_stage_GA)
    cum_int_one_stage_GA = np.cumsum(int_rewards_one_stage_GA)

    cum_random = np.cumsum(rewards_random)

    T = len(cum_frac_GA)
    timesteps = np.arange(1, T+1)
    # Compute the average cumulative rewards at each time step
    avg_cum_frac_GA = cum_frac_GA / timesteps
    avg_cum_int_GA = cum_int_GA / timesteps

    avg_cum_frac_FTRL_l2 = cum_frac_FTRL_l2/ timesteps
    avg_cum_int_FTRL_l2 = cum_int_FTRL_l2 / timesteps
    avg_cum_frac_FTRL_entropy = cum_frac_FTRL_entropy/ timesteps
    avg_cum_int_FTRL_entropy = cum_int_FTRL_entropy / timesteps

    avg_cum_frac_one_stage_GA = cum_frac_one_stage_GA/ timesteps
    avg_cum_int_one_stage_GA = cum_int_one_stage_GA/ timesteps

    avg_cum_random = cum_random / timesteps

    # Plot the rewards
    # plt.plot(avg_cum_int_GA, label='GA')
    # plt.plot(avg_cum_int_one_stage_GA, label='one-stage GA')
    # plt.plot(avg_cum_int_FTRL_l2, label='FTRL - L2')
    # plt.plot(avg_cum_int_FTRL_entropy, label='FTRL - entropy')
    # plt.plot([int_opt / T] * T, linestyle='--', linewidth=1.5, label='OPT')
    # plt.plot([balkanskiVal / T] * T, linestyle='-.', linewidth=1.5, label='CONTINUOUS-OPT')
    # plt.plot([repGreedyVal / T] * T, linestyle='-.', linewidth=1.5, label='REP-GREEDY')


    # Prepare data and labels
    lines = [
        (avg_cum_int_GA, 'RAOCO-GA'),
        (avg_cum_int_FTRL_l2, 'RAOCO-FTRL-L2'),
        (avg_cum_int_FTRL_entropy, 'RAOCO-FTRL-H'),
        (avg_cum_int_one_stage_GA, '1S-GA'),
        (avg_cum_random, 'Random')
    ]
    fixed_lines = [
        ([int_opt / T] * T, '--', 'OPT'),
        ([balkanskiVal / T] * T, '-.', 'CNT-OPT'),
        ([repGreedyVal / T] * T, '-.', 'REP-GRD'),
    ]

    num_lines = len(lines) + len(fixed_lines)
    colors = [cmap(i / (num_lines - 1)) for i in range(num_lines)]

    # Plot lines with markers
    for i, (data, label) in enumerate(lines):
        plt.plot(
            data,
            label=label,
            color=colors[i],
            marker=marker_styles[i],
            markevery=10,  # Place markers every 10 points
        )

    # Plot fixed lines (optional)
    for i, (data, style, label) in enumerate(fixed_lines):
        plt.plot(
            data,
            linestyle=style,
            linewidth=1.5,
            label=label,
            color=colors[len(lines) + i],
            marker=marker_styles[len(lines) + i],
            markevery=10,  
        )

    # plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Rewards")
    plt.legend()
    plt.grid(True)

    plt.savefig(f'./plots/{dataset}.pdf', bbox_inches='tight', )
    plt.close()  # Close the plot to avoid display
