from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats

import ray

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy.random as npr
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression

from utilities import *

#%matplotlib inline

ray.init()


@ray.remote
def run_constrained_bandits(T, reward_gaussian_means, cost_gaussian_means, known_arms_indicator, threshold, do_UCB = False):
	#reward_gaussian_means = [.1, .2, .4, .7]
	reward_gaussian_arms = [SimpleGaussianArm(mean, .1, truncated = True) for mean in reward_gaussian_means]
	#cost_gaussian_means = [0, .4, .5, .2]
	#cost_gaussian_means = [0,0,0,0]
	#known_arms_indicator = [1, 1, 1 , 1]
	cost_gaussian_arms = [SimpleGaussianArm(mean, .1, truncated = True) for mean in cost_gaussian_means]
	banditenv = DoubleMultiArm(reward_gaussian_arms, cost_gaussian_arms)
	#threshold = .4
	alpha_r = 1/threshold + 1
	#alpha_r = 1
	alpha_c = 1
	num_arms = len(reward_gaussian_means)
	initial_rewards_means = [0 for _ in range(num_arms)]
	initial_cost_means = [1 for _ in range(num_arms)]
	for i in range(len(known_arms_indicator)):
	  if known_arms_indicator[i]:
	    initial_cost_means[i] = cost_gaussian_means[i]

	opt_policy = banditenv.get_optimal_policy(threshold)
	opt_policy_means = banditenv.evaluate_policy(opt_policy)

	rewards = []
	costs = []
	#logging_frequency = 100

	eps = 0.000001
	banditalg = ConstrainedBandit( initial_rewards_means, initial_cost_means, threshold, T, known_arms_indicator = known_arms_indicator, alpha_r = alpha_r, alpha_c = alpha_c, do_UCB = do_UCB)
	for t in range(T):
	  policy = banditalg.get_policy()
	  if np.abs(sum(policy) -1) > eps:
	    raise ValueError("Policy didn't sum to 1. It summed to {} instead.".format(sum(policy)))

	  index = banditalg.get_arm_index()
	  our_policy_means = banditenv.evaluate_policy(policy)
	  rewards.append(our_policy_means[0])
	  costs.append(our_policy_means[1])
	  if our_policy_means[1] > threshold:
	  	raise ValueError("Our policy means {}, threshold {} our policy {} upper cost mean {} upper reward mean {}".format(our_policy_means[1], threshold, policy, banditalg.upper_cost_means, banditalg.upper_rewards_means))


	  (r, f) = banditenv.get_rewards(index) 
	  banditalg.update(index, r, f)

	return costs, rewards


def get_summary(rewards_costs, num_repetitions, opt_reward, T):
	rewards = []
	costs = []
	regrets = []
	for i in range(len(rewards_costs)):
		costs.append(rewards_costs[i][0])
		rewards.append(rewards_costs[i][1])
		regrets.append(np.cumsum(opt_reward - rewards_costs[i][1]))

	reg_summary = np.zeros((num_repetitions, T))
	cost_summary = np.zeros((num_repetitions, T))
	regret_summary = np.zeros((num_repetitions, T))
	for i in range(num_repetitions):
		cost_summary[i, :] = costs[i]
		reg_summary[i, :] = rewards[i]
		regret_summary[i, :] = regrets[i]


	mean_cost = np.mean(cost_summary, axis = 0)
	std_cost = np.std(cost_summary, axis = 0)

	mean_reward = np.mean(reg_summary, axis = 0)
	std_reward = np.std(reg_summary, axis = 0)

	mean_regret = np.mean(regret_summary, axis = 0)
	std_regret = np.mean(regret_summary, axis = 0)


	return mean_cost, std_cost, mean_reward, std_reward, mean_regret, std_regret



T = 1000000
#T = 100
num_repetitions = 3


##################################################
reward_gaussian_means = [.1, .2, .4, .7]
cost_gaussian_means = [0, .4, .5, .2]
known_arms_indicator = [1, 0, 0, 0]

reward_gaussian_arms = [SimpleGaussianArm(mean, .1, truncated = True) for mean in reward_gaussian_means]
cost_gaussian_arms = [SimpleGaussianArm(mean, .1, truncated = True) for mean in cost_gaussian_means]
banditenv = DoubleMultiArm(reward_gaussian_arms, cost_gaussian_arms)

threshold = .5
num_arms = len(reward_gaussian_means)

opt_policy = banditenv.get_optimal_policy(threshold)
opt_policy_means = banditenv.evaluate_policy(opt_policy)
###################################################


opt_rewards = [opt_policy_means[0]]*T
opt_cost =[opt_policy_means[1]]*T
threshold_cost = [threshold]*T

print("Opt policy ", opt_policy)
print("opt policy means ", opt_policy_means)
print("true reward means ", reward_gaussian_means)
print("true costs means ", cost_gaussian_means)



rewards_costs = [run_constrained_bandits.remote(T, reward_gaussian_means, cost_gaussian_means, known_arms_indicator, threshold) for _ in range(num_repetitions)]
rewards_costs = ray.get(rewards_costs)

#rewards_costs = [run_constrained_bandits(T, reward_gaussian_means, cost_gaussian_means, known_arms_indicator, threshold) for _ in range(num_repetitions)]


print(rewards_costs)

mean_cost, std_cost, mean_reward, std_reward, mean_regret, std_regret = get_summary(rewards_costs, num_repetitions, opt_rewards[0], T)


print("Inst regret ", mean_regret)
print("Reward ", mean_reward)

timesteps = np.arange(T) + 1

print("opt rewards ", opt_rewards[0])


font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)

plt.figure(figsize=(5,5))
# plt.subplot(1, 2, 1)
plt.title("Rewards Evolution")
plt.plot(timesteps, mean_reward, label = "Rewards", linewidth = 3.5, color = "blue")
plt.fill_between(timesteps, mean_reward - .2*std_reward, mean_reward  + .2*std_reward, color = "blue", alpha = .1)
plt.plot(timesteps, opt_rewards, label = "Opt rewards", linewidth = 3.5, color = "red")
plt.legend(loc = "lower right", fontsize = 15)
# plt.subplot(1, 2, 2)
plt.savefig("./rewards_evolution_tau_{}.png".format(threshold))

plt.figure(figsize=(5,5))
plt.title("Cost Evolution")
plt.plot(timesteps, mean_cost,  label = "Costs", linewidth = 3.5, color = "blue")
plt.fill_between(timesteps, mean_cost - .2*std_cost, mean_cost + .2*std_cost, color = "blue", alpha = .1)
plt.plot(timesteps, opt_cost,  label = "Optimal cost", linewidth = 3.5, color = "red")
plt.plot(timesteps, threshold_cost,  label = "Threshold cost", linewidth = 3.5, color = "black")
plt.legend(loc = "lower right", fontsize = 15)
plt.savefig("./cost_evolution_tau_{}.png".format(threshold))


plt.figure(figsize=(5,5))
plt.title("Regret")
plt.plot(timesteps, mean_regret, linewidth = 3.5, color = "black")
plt.fill_between(timesteps, mean_regret - .2*std_regret, mean_regret + .2*std_regret, color = "blue", alpha = .1)
plt.legend(loc="upper left")
#plt.title("Constrained Bandits")
plt.savefig("./constrained_regrets_tau_{}.png".format(threshold))



