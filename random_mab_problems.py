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
def run_random_mab_sweep_experiment(T, reward_gaussian_means, cost_gaussian_means, known_arms_indicator, threshold, do_UCB = False):






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





	




ray.init()
@ray.remote
def run_random_linucb_sweep_experiment(T, d, err_var, lam, nm_ini, 
	TS = False, logging_frequency  = 1, num_arms = 100):
	
	theta = np.random.normal(0, 1, d)
	theta = theta/np.linalg.norm(theta)
	mu = np.random.normal(0, 1, d)
	mu = mu/np.linalg.norm(mu)
	tau =  np.random.random()

	A_0 = np.random.normal(0, 1, (d, num_arms))#np.eye(10)
	for i in range(num_arms):
		A_0[:,i] = A_0[:,i]/np.linalg.norm(A_0[:,i]) 

	bandit = LinearBandit( theta, mu, tau, err_var, A_0 )

	if TS:
		solv = LinTS(bandit, lam, nm_ini, tau, 
			logging_frequency = logging_frequency)
	else:
		solv = LinUCB(bandit, lam, nm_ini, tau, 
			logging_frequency = logging_frequency)

	#opt_reward = bandit.best_proba
	#opt_cost = bandit.best_cost
	costs   = []
	rewards = []

	for t in range(T):
		_, i, scaling, r, c = solv.run_one_step()
		solv.update_regret(i, scaling)
		if t&100 ==0 :
			print("Iteration {}".format(t))
		if t%logging_frequency == 0:
			costs.append(c)
			rewards.append(r)

	return solv.regrets, costs, rewards#, opt_reward, opt_cost 