import numpy as np
import matplotlib.pyplot as plt
import pickle

settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'},
            {'dataset': 'images', 'title': 'Image summarization'},
            {'dataset': 'teamformation', 'title': 'Team formation'},
            {'dataset': 'influence', 'title': 'Influence maximization'}]

# settings = [{'dataset': 'images', 'title': 'Image summarization'}]
# settings = [{'dataset': 'wikipedia', 'title': 'Wikipedia articles representatives'}]
settings = [{'dataset': 'teamformation', 'title': 'Team formation'}]
# settings = [{'dataset': 'movies', 'title': 'Movie recommendation'}]
# settings = [{'dataset': 'influence', 'title': 'Influence maximization'}]
settings = [{'dataset': 'coverage', 'title': 'Coverage'}]



cmap = plt.get_cmap("viridis")  # colorblind palette
marker_styles = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']  # Unique markers for each line

for setting in settings:
    dataset = setting['dataset']
    title = setting['title']
    with open(f'./results/{dataset}.pkl', 'rb') as file:
        data = pickle.load(file)

    # Retrieve the rewards array from the dictionary
    if 'frac_rewards_GA' in data:
        frac_rewards_GA = data["frac_rewards_GA"]
        int_rewards_GA = data["int_rewards_GA"]
        eta_GA = data['eta_GA']
        print(f'eta GA = {eta_GA}')
        print(f"GA sol = {np.round(data['frac_sol_GA'], 1)}")
        T = len(frac_rewards_GA)
    if 'frac_rewards_FTRL_l2' in data:
        frac_rewards_FTRL_l2 = data["frac_rewards_FTRL_l2"]
        int_rewards_FTRL_l2 = data["int_rewards_FTRL_l2"]
        eta_FTRL_l2 = data['eta_FTRL_l2']
        print(f'eta FTRL l2 = {eta_FTRL_l2}')
        T = len(frac_rewards_FTRL_l2)
    if 'frac_rewards_FTRL_entropy' in data:   
        frac_rewards_FTRL_entropy = data["frac_rewards_FTRL_entropy"]
        int_rewards_FTRL_entropy = data["int_rewards_FTRL_entropy"]
        eta_FTRL_entropy = data['eta_FTRL_entropy']
        print(f'eta FTRL entropy = {eta_FTRL_entropy}')
        T = len(frac_rewards_FTRL_entropy)
    if 'one_stage_frac_rewards_GA' in data:
        frac_rewards_one_stage_GA = data['one_stage_frac_rewards_GA']
        int_rewards_one_stage_GA = data['one_stage_int_rewards_GA']
        eta_one_stage_GA = data['eta_one_stage_GA']
        print(f"one stage GA sol = {np.round(data['frac_sol_one_stage_GA'], 1)}")

        print(f'eta one-stage GA = {eta_one_stage_GA}')

    if 'random' in data:
        rewards_random = data['random']

    if 'frac_opt' in data:
        frac_opt = data['frac_opt']
        int_opt = data['int_opt']
        print(f"int_sol = {data['int_sol']}")

    if 'balkanski_val' in data:
        balkanskiVal = data[f'balkanski_val']

    if 'repGreedy_val' in data:
        repGreedyVal = data['repGreedy_val']

    # print(data['int_sol_GA'])
    # print(data['int_sol'])

    
    timesteps = np.arange(1, T+1)

    # Compute cumulative sum of rewards and running average
    if 'frac_rewards_GA' in data:
        cum_frac_GA = np.cumsum(frac_rewards_GA)
        cum_int_GA = np.cumsum(int_rewards_GA)
        avg_cum_frac_GA = cum_frac_GA / timesteps
        avg_cum_int_GA = cum_int_GA / timesteps
    if 'frac_rewards_FTRL_l2' in data:
        cum_frac_FTRL_l2 = np.cumsum(frac_rewards_FTRL_l2)
        cum_int_FTRL_l2 = np.cumsum(int_rewards_FTRL_l2)
        avg_cum_frac_FTRL_l2 = cum_frac_FTRL_l2 / timesteps
        avg_cum_int_FTRL_l2 = cum_int_FTRL_l2 / timesteps
    if 'frac_rewards_FTRL_entropy' in data:
        cum_frac_FTRL_entropy = np.cumsum(frac_rewards_FTRL_entropy)
        cum_int_FTRL_entropy = np.cumsum(int_rewards_FTRL_entropy)
        avg_cum_frac_FTRL_entropy = cum_frac_FTRL_entropy / timesteps
        avg_cum_int_FTRL_entropy = cum_int_FTRL_entropy / timesteps
    if 'one_stage_frac_rewards_GA' in data:
        cum_frac_one_stage_GA = np.cumsum(frac_rewards_one_stage_GA)
        cum_int_one_stage_GA = np.cumsum(int_rewards_one_stage_GA)
        avg_cum_frac_one_stage_GA = cum_frac_one_stage_GA / timesteps
        avg_cum_int_one_stage_GA = cum_int_one_stage_GA / timesteps
    if 'random' in data:
        cum_random = np.cumsum(rewards_random)
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
    lines = []
    if 'frac_rewards_GA' in data:
        lines.append((avg_cum_int_GA, 'RAOCO-OGA'))
    if 'frac_rewards_FTRL_l2' in data:
        lines.append((avg_cum_int_FTRL_l2, 'RAOCO-FTRL-$L_2$'))
    if 'frac_rewards_FTRL_entropy' in data:
        lines.append((avg_cum_int_FTRL_entropy, 'RAOCO-FTRL-H'))
    if 'one_stage_frac_rewards_GA' in data:
        lines.append((avg_cum_int_one_stage_GA, '1S-OGA'))
    if 'random' in data:
        lines.append((avg_cum_random, 'Random'))

    # lines = [
    #     (avg_cum_int_GA, 'RAOCO-OGA'),
    #     (avg_cum_int_FTRL_l2, 'RAOCO-FTRL-$L_2$'),
    #     (avg_cum_int_FTRL_entropy, 'RAOCO-FTRL-H'),
    #     (avg_cum_int_one_stage_GA, '1S-OGA'),
    #     (avg_cum_random, 'Random')
    # ]
    fixed_lines = []
    if 'frac_opt' in data:
        fixed_lines.append(([int_opt / T] * T, '--', 'OFLN - OPT'))

    if 'balkanski_val' in data:
        fixed_lines.append(([balkanskiVal / T] * T, '-.', 'OFLN - CO'),)

    if 'repGreedy_val' in data:
        fixed_lines.append(([repGreedyVal / T] * T, '-.', 'OFLN - RGR'))

    # fixed_lines = [
    #     ([int_opt / T] * T, '--', 'OFLN - OPT'),
    #     ([balkanskiVal / T] * T, '-.', 'OFLN - CO'),
    #     ([repGreedyVal / T] * T, '-.', 'OFLN - RGR'),
    # ]

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
    plt.xlabel("Time Steps", fontsize=16)
    plt.ylabel("$C_t$", fontsize=16)
    plt.legend()
    plt.grid(True)

    plt.savefig(f'./plots/{dataset}.pdf', bbox_inches='tight', )
    plt.close()  # Close the plot to avoid display
