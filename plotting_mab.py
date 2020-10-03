import time
import ray
import IPython
import numpy as np
from math import log, exp
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os



T = 1000
thresholds = [.5]

for threshold in thresholds:

	(timesteps, mean_regret, std_regret, mean_cost, std_cost, mean_reward, 
		std_reward, threshold, opt_cost, opt_reward, T ) = pickle.load(open("./mab/data/T{}/data_mab_{}_{}.p".format(T, threshold, T), "rb"))


	opt_rewards = [opt_reward]*T
	opt_costs =[opt_cost]*T
	threshold_cost = [threshold]*T

	logging_frequency = int(T/len(timesteps))



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
