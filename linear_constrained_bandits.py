import time
import ray
import IPython
import numpy as np
from math import log, exp
import matplotlib.pyplot as plt
from utilities_linear import LinearBandit, LinUCB




# ray.init()

# @ray.remote
def run_linucb_sweep_experiment(T, theta, mu, err_var, A_0, lam, nm_ini):
	bandit = LinearBandit( theta, mu, err_var, A_0 )

	solv = LinUCB(bandit, lam, nm_ini)


	for i in range(T):
		_, i, r, c = solv.run_one_step()
		solv.update_regret(i)

	return solv.regrets


num_repetitions = 10
T = 100
d = 10
theta = np.arange(10)
mu = np.arange(10)
err_var = 1
A_0 = np.eye(10)
lam = .1
nm_ini = 5



#linucb_regrets = [run_linucb_sweep_experiment.remote(T, theta, mu, err_var, A_0, lam, nm_ini) for _ in range(num_repetitions) ]
#linucb_regrets = ray.get(linucb_regrets)
linucb_regrets = [run_linucb_sweep_experiment(T, theta, mu, err_var, A_0, lam, nm_ini) for _ in range(num_repetitions) ]




reg_summary = np.zeros((num_repetitions, T))
for j in range(num_repetitions):
		reg_summary[j, :] = linucb_regrets[j]


#print([np.sum(reg) for reg in ray.get(cor_regrets) ])
mean_regret = np.mean(reg_summary, axis = 0)
std_regret = np.std(reg_summary, axis = 0)


#IPython.embed()
timesteps = np.arange(T) + 1
#IPython.embed()


plt.plot(timesteps, mean_regret, label = "LinUCB ", color = "red")
plt.fill_between(timesteps, mean_regret - .5*std_regret, mean_regret + .5*std_regret, color = "red", alpha = .1 )

#IPython.embed()
#plt.figure(figsize=(40,20))
plt.legend(loc="lower right")
plt.savefig("./LinUCB_Contextual.png")
# ####################
