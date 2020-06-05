import time
import ray
import IPython
import numpy as np
from math import log, exp
import matplotlib.pyplot as plt
from utilities_linear import LinearBandit, LinUCB




# ray.init()

# @ray.remote
def run_linucb_sweep_experiment(T, theta, mu, tau, err_var, A_0, lam, nm_ini):
	bandit = LinearBandit( theta, mu, tau, err_var, A_0 )

	solv = LinUCB(bandit, lam, nm_ini, tau)

	opt_reward = bandit.best_proba
	opt_cost = bandit.best_cost
	costs   = []
	rewards = []

	for i in range(T):
		_, i, scaling, r, c = solv.run_one_step()
		solv.update_regret(i, scaling)
		costs.append(c)
		rewards.append(r)

	return solv.regrets, costs, rewards, opt_reward, opt_cost 


def strided_method(ar):
    a = np.concatenate(( ar, ar[:-1] ))
    L = len(ar)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a[L-1:], (L,L), (-n,n))



num_repetitions = 10
T = 1000
d = 10
theta = np.arange(d)
theta = theta/np.max(theta)
mu = np.flip(theta)
err_var = .1
A_0 = strided_method(np.arange(d))#np.eye(10)

lam = .1
nm_ini = 0

tau = .5




#linucb_regrets = [run_linucb_sweep_experiment.remote(T, theta, mu, err_var, A_0, lam, nm_ini) for _ in range(num_repetitions) ]
#linucb_regrets = ray.get(linucb_regrets)
linucb_regrets = [run_linucb_sweep_experiment(T, theta, mu, tau, err_var, A_0, lam, nm_ini) for _ in range(num_repetitions) ]

opt_reward = linucb_regrets[0][3]
opt_cost = linucb_regrets[0][4]



### REGRET
reg_summary = np.zeros((num_repetitions, T))
for j in range(num_repetitions):
		reg_summary[j, :] = linucb_regrets[j][0]


#print([np.sum(reg) for reg in ray.get(cor_regrets) ])
mean_regret = np.mean(reg_summary, axis = 0)
std_regret = np.std(reg_summary, axis = 0)




### COST
cost_summary = np.zeros((num_repetitions, T))
for j in range(num_repetitions):
		cost_summary[j, :] = linucb_regrets[j][1]

#print([np.sum(reg) for reg in ray.get(cor_regrets) ])
mean_cost = np.mean(cost_summary, axis = 0)
std_cost = np.std(cost_summary, axis = 0)


### REWARD
reward_summary = np.zeros((num_repetitions, T))
for j in range(num_repetitions):
		reward_summary[j, :] = linucb_regrets[j][2]

mean_reward = np.mean(reward_summary, axis = 0)
std_reward = np.std(reward_summary, axis = 0)





font = {#'family' : 'normal',
        #'weight' : 'bold',
        'size'   : 16}


#IPython.embed()
timesteps = np.arange(T) + 1
#IPython.embed()
plt.figure(figsize=(5,5))

plt.title("Regret")
plt.plot(timesteps, mean_regret, label = "LinUCB ", color = "red")
plt.fill_between(timesteps, mean_regret - .5*std_regret, mean_regret + .5*std_regret, color = "red", alpha = .1 )

#IPython.embed()
#plt.figure(figsize=(40,20))
plt.legend(loc="lower right")
plt.savefig("./Linear_Regret_{}.png".format(tau))
# ####################



plt.figure(figsize=(5,5))

plt.title("Cost")
plt.plot(timesteps, [tau]*T, label = "Threshold", color = "black")
plt.plot(timesteps, [opt_cost]*T, label = "Opt Cost", color = "blue")
plt.plot(timesteps, mean_cost, label = "LinUCB ", color = "red")
plt.fill_between(timesteps, mean_cost - .5*std_cost, mean_cost + .5*std_cost, color = "red", alpha = .1 )

#IPython.embed()
#plt.figure(figsize=(40,20))
plt.legend(loc="lower right")
plt.savefig("./Linear_Cost_{}.png".format(tau))
# ####################
plt.figure(figsize=(5,5))

plt.title("Reward")
plt.plot(timesteps, [opt_reward]*T, label = "Opt Reward", color = "blue")
plt.plot(timesteps, mean_reward, label = "LinUCB ", color = "red")
plt.fill_between(timesteps, mean_reward - .5*std_reward, mean_reward + .5*std_reward, color = "red", alpha = .1 )
plt.legend(loc="lower right")
plt.savefig("./Linear_Reward_{}.png".format(tau))
# ####################



