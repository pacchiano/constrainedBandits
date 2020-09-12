import time
import ray
import IPython
import numpy as np
from math import log, exp
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os


taus = [1, .2, .5, .8]
TSvals =[True,  False]
T = 10000



for tau in taus:
	for TS in TSvals:

		if TS:
			algo_label = "Safe-LTS"
		else:
			algo_label = "HPLC-LUCB"


		(timesteps, mean_regret, std_regret, mean_cost, std_cost, mean_reward, std_reward, tau, opt_cost, opt_reward, T) = pickle.load(open("./linear_experiments/data/T{}/data_linear_{}_{}_{}.p".format(T, tau, algo_label, T), "rb"))

		#IPython.embed()

		font = {#'family' : 'normal',
		        #'weight' : 'bold',
		        'size'   : 16}

		matplotlib.rc('font', **font)

		#IPython.embed()
		timesteps = np.arange(T) + 1
		#IPython.embed()
		plt.figure(figsize=(5,5))

		plt.title("Regret")
		plt.plot(timesteps, mean_regret, label = algo_label, color = "red")
		plt.fill_between(timesteps, mean_regret - .5*std_regret, mean_regret + .5*std_regret, color = "red", alpha = .1 )

		#IPython.embed()
		#plt.figure(figsize=(40,20))
		plt.legend(loc="lower right", fontsize = 15)
		plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
		plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
		plt.savefig("./linear_experiments/plots/T{}/Linear_Regret_{}_{}_{}.png".format(T, tau, algo_label, T))
		# ####################



		plt.figure(figsize=(5,5))

		plt.title("Cost")
		plt.plot(timesteps, [tau]*T, label = "Threshold", color = "black")
		plt.plot(timesteps, [opt_cost]*T, label = "Opt Cost", color = "blue")
		plt.plot(timesteps, mean_cost, label = algo_label, color = "red")
		plt.fill_between(timesteps, mean_cost - .5*std_cost, mean_cost + .5*std_cost, color = "red", alpha = .1 )

		#IPython.embed()
		#plt.figure(figsize=(40,20))
		plt.legend(loc="lower right", fontsize = 15)
		plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

		plt.savefig("./linear_experiments/plots/T{}/Linear_Cost_{}_{}_{}.png".format(T, tau, algo_label, T))
		# ####################
		plt.figure(figsize=(5,5))

		plt.title("Reward")
		plt.plot(timesteps, [opt_reward]*T, label = "Opt Reward", color = "blue")
		plt.plot(timesteps, mean_reward, label = algo_label, color = "red")
		plt.fill_between(timesteps, mean_reward - .5*std_reward, mean_reward + .5*std_reward, color = "red", alpha = .1 )
		plt.legend(loc="lower right", fontsize = 15)
		plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

		plt.savefig("./linear_experiments/plots/T{}/Linear_Reward_{}_{}_{}.png".format(T,tau, algo_label,  T))
		# ####################
		plt.close('all')
