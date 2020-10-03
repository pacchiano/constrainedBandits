from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib 
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

import scipy.stats
import os

import ray

import requests
import pandas as pd
import tempfile
import matplotlib.pyplot as plt
#from sklearn.model_selection import train_test_split
#from sklearn import metrics
import numpy.random as npr
#from scipy.stats import wasserstein_distance, ks_2samp
#from sklearn.linear_model import LogisticRegression

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
	local_rewards = []
	local_costs = []

	for t in range(T):
		policy = banditalg.get_policy()
		if np.abs(sum(policy) -1) > eps:
			raise ValueError("Policy didn't sum to 1. It summed to {} instead.".format(sum(policy)))
		index = banditalg.get_arm_index()
		our_policy_means = banditenv.evaluate_policy(policy)

		local_rewards.append(our_policy_means[0])
		local_costs.append(our_policy_means[1])

		if t%logging_frequency == 0:
		  print("iteration {}".format(t))
		  rewards.append(np.mean(local_rewards))
		  costs.append(np.mean(local_costs))
		  local_rewards = []
		  local_costs = []

		if our_policy_means[1] > threshold:
			raise ValueError("Our policy means {}, threshold {} our policy {} upper cost mean {} upper reward mean {}".format(our_policy_means[1], threshold, policy, banditalg.upper_cost_means, banditalg.upper_rewards_means))


		(r, f) = banditenv.get_rewards(index) 
		banditalg.update(index, r, f)

	return costs, rewards




T = 1000000
#T = 100
num_repetitions = 10

thresholds = [.2, .5, .6, .8]
logging_frequency = 10


path = os.getcwd()
if not os.path.isdir("{}/mab/data/T{}".format(path,T)):
	try:
		os.mkdir("{}/mab/data/T{}".format(path,T))
		os.mkdir("{}/mab/plots/T{}".format(path,T))
	except OSError:
		print ("Creation of the directories failed")
	else:
		print ("Successfully created the directory ")



##################################################
reward_gaussian_means = [.1, .2, .4, .7]
cost_gaussian_means = [0, .4, .5, .2]
known_arms_indicator = [1, 0, 0, 0]

reward_gaussian_arms = [SimpleGaussianArm(mean, .1, truncated = True) for mean in reward_gaussian_means]
cost_gaussian_arms = [SimpleGaussianArm(mean, .1, truncated = True) for mean in cost_gaussian_means]
banditenv = DoubleMultiArm(reward_gaussian_arms, cost_gaussian_arms)

num_arms = len(reward_gaussian_means)



for threshold in thresholds:

	opt_policy = banditenv.get_optimal_policy(threshold)
	opt_policy_means = banditenv.evaluate_policy(opt_policy)
	###################################################


	opt_reward = opt_policy_means[0]
	opt_cost = opt_policy_means[1]

	opt_rewards = [opt_policy_means[0]]*int(T/logging_frequency)
	opt_costs =[opt_policy_means[1]]*int(T/logging_frequency)
	threshold_cost = [threshold]*int(T/logging_frequency)

	# print("Opt policy ", opt_policy)
	# print("opt policy means ", opt_policy_means)
	# print("true reward means ", reward_gaussian_means)
	# print("true costs means ", cost_gaussian_means)



	rewards_costs = [run_constrained_bandits.remote(T, reward_gaussian_means, cost_gaussian_means, known_arms_indicator, threshold) for _ in range(num_repetitions)]
	rewards_costs = ray.get(rewards_costs)

	#rewards_costs = [run_constrained_bandits(T, reward_gaussian_means, cost_gaussian_means, known_arms_indicator, threshold) for _ in range(num_repetitions)]


	# print(rewards_costs)

	mean_cost, std_cost, mean_reward, std_reward, mean_regret, std_regret = get_summary(rewards_costs, num_repetitions, opt_rewards[0], T, logging_frequency)


	# print("Inst regret ", mean_regret)
	# print("Reward ", mean_reward)


	timesteps = np.arange(int(T/logging_frequency))*logging_frequency + 1




	# print("opt rewards ", opt_rewards[0])


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
	plt.savefig("./mab/plots/T{}/rewards_evolution_tau_{}_{}.png".format(T,threshold, T))

	plt.figure(figsize=(5,5))
	plt.title("Cost Evolution")
	plt.plot(timesteps, mean_cost,  label = "Costs", linewidth = 3.5, color = "blue")
	plt.fill_between(timesteps, mean_cost - .2*std_cost, mean_cost + .2*std_cost, color = "blue", alpha = .1)
	plt.plot(timesteps, opt_costs,  label = "Optimal cost", linewidth = 3.5, color = "red")
	plt.plot(timesteps, threshold_cost,  label = "Threshold cost", linewidth = 3.5, color = "black")
	plt.legend(loc = "lower right", fontsize = 15)
	plt.savefig("./mab/plots/T{}/cost_evolution_tau_{}_{}.png".format(T,threshold, T))


	plt.figure(figsize=(5,5))
	plt.title("Regret")
	plt.plot(timesteps, mean_regret, linewidth = 3.5, color = "black")
	plt.fill_between(timesteps, mean_regret - .2*std_regret, mean_regret + .2*std_regret, color = "blue", alpha = .1)
	plt.legend(loc="upper left")
	#plt.title("Constrained Bandits")
	plt.savefig("./mab/plots/T{}/constrained_regrets_tau_{}_{}.png".format(T,threshold,T))

	plt.close("all")

	import pickle

	pickle.dump((timesteps, mean_regret, std_regret, mean_cost, std_cost, mean_reward, 
		std_reward, threshold, opt_cost, opt_reward, T ), open("./mab/data/T{}/data_mab_{}_{}.p".format(T, threshold, T), "wb"))