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



ray.init()
@ray.remote
def run_random_mab_sweep_experiment(T, num_arms, lower_threshold , do_UCB = False, logging_frequency = 1):



	reward_gaussian_means = np.random.random(num_arms)
	cost_gaussian_means = np.random.random(num_arms)
	threshold = max(np.min(cost_gaussian_means) + (np.max(cost_gaussian_means) - np.min(cost_gaussian_means))*np.random.random(), lower_threshold)


	known_arms_indicator =  np.zeros(num_arms)
	known_arms_indicator[np.argmin(cost_gaussian_means)] = 1
	


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
	local_rewards = []
	local_costs = []

	eps = 0.000001
	banditalg = ConstrainedBandit( initial_rewards_means, initial_cost_means, threshold, T, known_arms_indicator = known_arms_indicator, alpha_r = alpha_r, alpha_c = alpha_c, do_UCB = do_UCB)
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

	return costs, rewards, opt_policy_means[0]








T = 1000000
num_arms = 10
num_repetitions = 10
logging_frequency = 100

lower_thresholds = [0, .2, .5]
colors = ["black", "blue", "red", "green", "violet", "purple", "orange", "yellow"]


if int(T/logging_frequency)*logging_frequency < T:
	raise ValueError("The logging frequency does not divide T.")


path = os.getcwd()
if not os.path.isdir("{}/mab/data/RandomT{}".format(path,T)):
	try:
		os.mkdir("{}/mab/data/RandomT{}".format(path,T))
		os.mkdir("{}/mab/plots/RandomT{}".format(path,T))
	except OSError:
		print ("Creation of the directories failed")
	else:
		print ("Successfully created the directory ")



font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}

matplotlib.rc('font', **font)
matplotlib.rcParams['text.usetex'] = True
plt.figure(figsize=(5,5))

timesteps = np.arange(int(T/logging_frequency))*logging_frequency + 1

for j in range(len(lower_thresholds)):

	lower_threshold = lower_thresholds[j]
	rewards_costs = [run_random_mab_sweep_experiment.remote(T, num_arms, lower_threshold, do_UCB = False, 
		logging_frequency = logging_frequency) for _ in range(num_repetitions)]
	rewards_costs = ray.get(rewards_costs)
	#opt_reward = rewards_costs[-1]
	#rewards_costs = (rewards_costs[0], rewards_costs[1])

	#opt_rewards = [rewards_costs[-1]]*int(T/logging_frequency)
	opt_rewards = [ rewards_costs[i][-1] for i in range(len(rewards_costs))  ]
	opt_rewards_average = np.average(opt_rewards)

	mean_cost, std_cost, mean_reward, std_reward, mean_regret, std_regret = get_summary(rewards_costs, num_repetitions, 
		opt_rewards_average, T, logging_frequency)

	
	plt.plot(timesteps, mean_regret, linewidth = 3.5, color = colors[j], label = 'Min threshold {}'+str(lower_threshold))
	plt.fill_between(timesteps, mean_regret - .2*std_regret, mean_regret + .2*std_regret, color = colors[j], alpha = .1)
	#plt.title("Constrained Bandits")
	


	import pickle

	pickle.dump((timesteps, mean_regret, std_regret, mean_cost, std_cost, mean_reward, 
		std_reward, opt_rewards_average, T , lower_threshold), open("./mab/data/RandomT{}/data_mab_{}_lower_threshold_{}.p".format(T, T, lower_threshold), "wb"))

plt.title("Regret")
plt.legend(loc="upper left")
plt.savefig("./mab/plots/RandomT{}/constrained_regrets_{}.png".format(T,T))


