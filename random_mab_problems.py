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
def run_random_mab_sweep_experiment(T, num_arms, do_UCB = False, logging_frequency = 1):



	reward_gaussian_means = np.random.random(num_arms)
	cost_gaussian_means = np.random.random(num_arms)
	threshold = np.min(cost_gaussian_means) + (np.max(cost_gaussian_means) - np.min(cost_gaussian_means))*np.random.random()


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

	eps = 0.000001
	banditalg = ConstrainedBandit( initial_rewards_means, initial_cost_means, threshold, T, known_arms_indicator = known_arms_indicator, alpha_r = alpha_r, alpha_c = alpha_c, do_UCB = do_UCB)
	for t in range(T):
	  policy = banditalg.get_policy()
	  if np.abs(sum(policy) -1) > eps:
	    raise ValueError("Policy didn't sum to 1. It summed to {} instead.".format(sum(policy)))

	  index = banditalg.get_arm_index()
	  our_policy_means = banditenv.evaluate_policy(policy)

	  if t%logging_frequency == 0:
		  rewards.append(our_policy_means[0])
		  costs.append(our_policy_means[1])
	  
	  if our_policy_means[1] > threshold:
	  	raise ValueError("Our policy means {}, threshold {} our policy {} upper cost mean {} upper reward mean {}".format(our_policy_means[1], threshold, policy, banditalg.upper_cost_means, banditalg.upper_rewards_means))


	  (r, f) = banditenv.get_rewards(index) 
	  banditalg.update(index, r, f)

	return costs, rewards





T = 1000000
num_arms = 10
num_experiments = 10
logging_frequency = 100

if int(T/logging_frequency)*logging_frequency < T:
	raise ValueError("The logging frequency does not divide T.")


path = os.getcwd()
if not os.path.isdir("{}/linear_experiments/data/RandomT{}".format(path,T)):
	try:
		os.mkdir("{}/linear_experiments/data/RandomT{}".format(path,T))
		os.mkdir("{}/linear_experiments/plots/RandomT{}".format(path,T))
	except OSError:
		print ("Creation of the directories failed")
	else:
		print ("Successfully created the directory ")








	

