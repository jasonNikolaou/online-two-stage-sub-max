import numpy as np
import matplotlib.pyplot as plt
import pickle

with open('./results/wikipedia.pkl', 'rb') as file:
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

T = len(cum_frac_GA)
# Compute the average cumulative rewards at each time step
avg_cum_frac_GA = cum_frac_GA / (np.arange(1, T + 1))
avg_cum_int_GA = cum_int_GA / (np.arange(1, T + 1))

avg_cum_frac_FTRL_l2 = cum_frac_FTRL_l2/ (np.arange(1, T + 1))
avg_cum_int_FTRL_l2 = cum_int_FTRL_l2 / (np.arange(1, T + 1))
avg_cum_frac_FTRL_entropy = cum_frac_FTRL_entropy/ (np.arange(1, T + 1))
avg_cum_int_FTRL_entropy = cum_int_FTRL_entropy / (np.arange(1, T + 1))

avg_cum_frac_one_stage_GA = cum_frac_one_stage_GA/ (np.arange(1, T + 1))
avg_cum_int_one_stage_GA = cum_int_one_stage_GA/ (np.arange(1, T + 1))

# Plot the rewards
# plt.plot(avg_cum_frac_GA, label='frac GA')
plt.plot(avg_cum_int_GA, label='GA')
# plt.plot(avg_cum_frac_one_stage_GA, label='frac one-stage GA')
plt.plot(avg_cum_int_one_stage_GA, label='one-stage GA')
# plt.plot(avg_cum_frac_FTRL, label='frac FTRL')
plt.plot(avg_cum_int_FTRL_l2, label='FTRL - L_2')
plt.plot(avg_cum_int_FTRL_entropy, label='FTRL - entropy')
# plt.plot([frac_opt / T] * T, linestyle=':', linewidth=1.5, label='frac opt')
plt.plot([int_opt / T] * T, linestyle='--', linewidth=1.5, label='OPT')
plt.plot([balkanskiVal / T] * T, linestyle='-.', linewidth=1.5, label='CONTINUOUS-OPT')
plt.plot([repGreedyVal / T] * T, linestyle='-.', linewidth=1.5, label='REP-GREEDY')

plt.title("Wikipedia article representatives")
plt.xlabel("Time Steps")
plt.ylabel("Rewards")
plt.legend()
plt.grid(True)

plt.savefig('./plots/wikipedia.pdf', bbox_inches='tight', )
plt.close()  # Close the plot to avoid display


print(frac_opt / T)
print(int_opt / T)