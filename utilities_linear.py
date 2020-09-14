import numpy as np
import copy
import scipy.stats


class ContextualBandit(object):

    def generate_context(self):
        raise NotImplementedError

    def generate_reward(self,i):
        raise NotImplementedError

EPS = .000000001


def helper(x):
    if x < 0:
        return 1
    else:
        return x



class LinearBandit(ContextualBandit):
    def __init__(self, theta, mu, tau, err_var, A_0):
        self.theta = theta
        self.mu = mu
        self.tau = tau

        self.err_var = err_var
        self.A_0 = A_0
        self.context_shape = A_0.shape
        self.d = self.context_shape[0]
        self.K = self.context_shape[1]
        self.n = self.K

        self.probas =  [np.dot(self.A_0[:,i], self.theta) for i in range(self.K)]
        self.costs = [np.dot(self.A_0[:,i], self.mu) for i in range(self.K)]
        self.opt_scalings = [helper(min(1, self.tau/(self.costs[k]+EPS))) for k in range(self.K)]

        self.scaled_probas = [self.probas[k]*self.opt_scalings[k] for k in range(self.K)]
        self.scaled_costs = [self.costs[k]*self.opt_scalings[k] for k in range(self.K)]
        self.best_action = np.argmax(self.scaled_probas)
        self.best_proba = self.scaled_probas[self.best_action]

        self.best_cost = self.scaled_costs[self.best_action]
        

    def generate_context(self):
        ### If we change this to have a changing context, uncomment the following lines:
        # self.A_0 = define this
        # self.probas = [np.dot(self.A_0[:,i], self.theta) for i in range(self.K)]
        # self.best_proba = max( self.probas )
        return self.A_0


    def generate_reward(self, i, scaling = 1, real = False):
        rwd = scaling*self.probas[i] + (1-real)*np.random.normal(0, self.err_var, 1)

        return rwd

    def generate_cost(self, i, scaling = 1, real = False):
        cost =  scaling*self.costs[i] + (1-real)*np.random.normal(0, self.err_var, 1)
        return cost






class Solver(object):
    def __init__(self, bandit, logging_frequency = 1):
        """
        bandit (Bandit): the target bandit to solve.
        """
        #assert isinstance(bandit, BernoulliBandit) or isinstance(bandit, LinearBandit) or isinstance(bandit, DuplicitousLinearBandit) or isinstance(bandit, StochasticContextualBandit) or isinstance(bandit , GaussianBandit)
#        np.random.seed(int(time.time()))

        self.bandit = bandit

        self.counts = [0] * self.bandit.n
        self.actions = []  # A list of machine ids, 0 to bandit.n-1.
        self.regret = 0.  # Cumulative regret.
        self.regrets = []  # History of cumulative regret.
        self.logging_frequency = logging_frequency
        self.counter = 0

    def update_regret(self, i, scaling):
        # i (int): index of the selected machine.
        instant_regret = self.bandit.best_proba - scaling*self.bandit.probas[i]
        self.regret += instant_regret
        if self.counter%self.logging_frequency == 0:
            self.regrets.append(self.regret)
        self.counter += 1
        return instant_regret

    @property
    def estimated_probas(self):
        raise NotImplementedError

    def run_one_step(self):
        """Return the machine index to take action on."""
        raise NotImplementedError




class LinUCB(Solver):
    def __init__(self, bandit, lam, nm_ini, tau = 1, 
        compressed = True, logging_frequency = 1):
        """

        init_proba (float): default to be 0.0: pesimistic initialization.
        """
        super(LinUCB, self).__init__(bandit, logging_frequency = logging_frequency)
        ##### Check the bandit instance is a LinearBandit
        self.cost_upper_bound = 2*np.max(bandit.costs)
        self.d, self.K = bandit.context_shape
        self.tau = tau
        self.pulls = [0]*self.K
        self.lam = lam
        self.nm_ini = nm_ini
        self.X = []
        self.Y = []
        self.C = []
        self.estimates_reward = np.zeros(self.K)
        self.estimates_cost = np.zeros(self.K)
        self.theta_hat = np.zeros(self.d)
        self.mu_hat = np.zeros(self.d)

        self.compressed = compressed
        self.X_Y = np.zeros(self.d)
        self.X_C = np.zeros(self.d)


        self.V = self.lam *  np.identity(self.d)
        if self.lam > 0:
            self.V_inverse = np.linalg.inv(self.V)
        else:
            self.V_inverse = np.zeros((d, d))

    def scale_context_to_safety(self, UCB_cost):
        scalings = np.zeros(self.K)
        for k in range(self.K):
            scalings[k] = helper(min(1, self.tau/(UCB_cost[k]+EPS)))
        return scalings

    def get_arm(self, context=None):
        if len(self.regrets) <= self.nm_ini:
            i = np.random.randint(0, self.K)
            scaling = self.tau/self.cost_upper_bound

        elif len(self.regrets) >= self.nm_ini +1:

            delta = 1/(1+len(self.regrets))
            #IPython.embed()
            #delta = 1.0/100


            

            if not self.compressed:
                self.theta_hat = self.V_inverse @ self.X_numpy.transpose() @ self.Y_numpy
                self.mu_hat = self.V_inverse @ self.X_numpy.transpose() @ self.C_numpy

            else:
                self.theta_hat = self.V_inverse @ self.X_Y
                self.mu_hat = self.V_inverse @ self.X_C



            ft_cost = 2.0 * (2 * np.log((np.linalg.det(self.V) ** 0.5) * (np.linalg.det(self.lam * np.identity(self.d)) ** -0.5) /delta)) ** 0.5 + self.lam ** 0.5 * np.linalg.norm(self.bandit.mu)

            UCB_cost = np.zeros(self.K)
            self.estimates_cost = np.matmul(np.squeeze(self.mu_hat), context )
            for k in range(self.K):
                UCB_cost[k] = self.estimates_cost[k] + ft_cost*((context[:, k].transpose() @ np.linalg.inv(self.V) @ context[:, k]) ** 0.5)

            scalings = self.scale_context_to_safety(UCB_cost)


            
            #ft = self.bandit.err_var * (2 * np.log((np.linalg.det(self.V) ** 0.5) * (np.linalg.det(self.lam * np.identity(self.d)) ** -0.5) /delta)) ** 0.5 + self.lam ** 0.5 * np.linalg.norm(self.bandit.theta)
            ft_reward = 1.0/(min(1, self.bandit.tau))*2.0 * (2 * np.log((np.linalg.det(self.V) ** 0.5) * (np.linalg.det(self.lam * np.identity(self.d)) ** -0.5) /delta)) ** 0.5 + self.lam ** 0.5 * np.linalg.norm(self.bandit.theta)

            UCB_reward = np.zeros(self.K)
            self.estimates_reward = np.matmul(np.squeeze(self.theta_hat), context )
            for k in range(self.K):
                UCB_reward[k] = self.estimates_reward[k] + ft_reward*((context[:, k].transpose() @ np.linalg.inv(self.V) @ context[:, k]) ** 0.5)

            scaled_UCB_rewards = [UCB_reward[k]*scalings[k] for k in range(self.K)]
            i = np.argmax(scaled_UCB_rewards)
            scaling = scalings[i]
            #print(scalings)
        else:
            raise ValueError("Something strange happened. This condition should never activate.")
        #   pass

        # self.X.append(context[:, i])
        # self.X_numpy = np.array( self.X )
        # self.V += np.outer(context[:, i], context[:,i])

        # if len(self.regrets) <= self.nm_ini:
        #   self.V_inverse = np.linalg.inv(self.V)
        # else:
        #   self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ context[:,i ], context[:,i ]) @ self.V_inverse) / (1 + context[:,i ] @ self.V_inverse @ context[:,i].transpose())
        return i, scaling


    def update_X_and_V(self, context, i, scaling):

        if not self.compressed:
            self.X.append(scaling*context[:, i])
            #print(scaling)
            self.X_numpy = np.array( self.X )
        
        self.V += np.outer(scaling*context[:, i], scaling*context[:,i])

        if len(self.regrets) <= self.nm_ini:
            self.V_inverse = np.linalg.inv(self.V)
        else:
            x = scaling*context[:,i]
            #self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ scaling*context[:,i ], scaling*context[:,i ]) @ self.V_inverse) / (1 + scaling*context[:,i ] @ self.V_inverse @ scaling*context[:,i].transpose())
            #self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ context[:,i ], context[:,i ]) @ self.V_inverse) / (1 + context[:,i ] @ self.V_inverse @ context[:,i].transpose())

            self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ x, x) @ self.V_inverse) / (1 + x @ self.V_inverse @ x.transpose())

    def update_with_reward(self,i,r):
        ## Update the estimates
        pass



    def run_one_step(self):
        context = self.bandit.generate_context()
        i, scaling = self.get_arm(context = context)
        


        self.update_X_and_V(context, i, scaling)
        r = self.bandit.generate_reward(i, scaling)
        r_real = self.bandit.generate_reward(i, scaling, real = True)
        c = self.bandit.generate_cost(i, scaling)
        c_real = self.bandit.generate_cost(i, scaling, real = True)

        if not self.compressed:
            self.Y.append(r)
            self.C.append(c)
            self.Y_numpy = np.array(self.Y)
            self.C_numpy = np.array(self.C)
        #self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        #self.estimates[i] = (self.counts[i]*self.estimates[i] + r)/(self.counts[i] + 1.)

        self.X_Y += scaling*context[:, i]*r
        self.X_C += scaling*context[:, i]*c
        self.counts[i] += 1

        return context,i, scaling, r_real, c_real


class LinTS(Solver):
    def __init__(self, bandit, lam, nm_ini, tau = 1, 
        compressed = True, logging_frequency = 1):
        """

        init_proba (float): default to be 0.0: pesimistic initialization.
        """
        super(LinTS, self).__init__(bandit, logging_frequency = logging_frequency)
        ##### Check the bandit instance is a LinearBandit
        self.cost_upper_bound = 2*np.max(bandit.costs)
        self.d, self.K = bandit.context_shape
        self.tau = tau
        self.pulls = [0]*self.K
        self.lam = lam
        self.nm_ini = nm_ini
        self.X = []
        self.Y = []
        self.C = []
        self.estimates_reward = np.zeros(self.K)
        self.estimates_cost = np.zeros(self.K)
        self.theta_hat = np.zeros(self.d)
        self.mu_hat = np.zeros(self.d)
        self.X_Y = np.zeros(self.d)
        self.X_C = np.zeros(self.d)

        self.compressed = compressed


        self.V = self.lam *  np.identity(self.d)
        if self.lam > 0:
            self.V_inverse = np.linalg.inv(self.V)
        else:
            self.V_inverse = np.zeros((d, d))

    def scale_context_to_safety(self, UCB_cost):
        scalings = np.zeros(self.K)
        for k in range(self.K):
            scalings[k] = helper(min(1, self.tau/(UCB_cost[k]+EPS)))
        return scalings

    def get_arm(self, context=None):
        if len(self.regrets) <= self.nm_ini:
            i = np.random.randint(0, self.K)
            scaling = self.tau/self.cost_upper_bound

        elif len(self.regrets) >= self.nm_ini +1:

            delta = 1/(1+len(self.regrets))


            #delta = 1.0/100

            #IPython.embed()

            #if not self.sgd:
            if not self.compressed:
                self.theta_hat = self.V_inverse @ self.X_numpy.transpose() @ self.Y_numpy
                self.mu_hat = self.V_inverse @ self.X_numpy.transpose() @ self.C_numpy
            else:
                self.theta_hat = self.V_inverse @ self.X_Y
                self.mu_hat = self.V_inverse @ self.X_C

            ft_cost = 2.0 * (2 * np.log((np.linalg.det(self.V) ** 0.5) * (np.linalg.det(self.lam * np.identity(self.d)) ** -0.5) /delta)) ** 0.5 + self.lam ** 0.5 * np.linalg.norm(self.bandit.mu)

            UCB_cost = np.zeros(self.K)
            self.estimates_cost = np.matmul(np.squeeze(self.mu_hat), context )
            for k in range(self.K):
                UCB_cost[k] = self.estimates_cost[k] + ft_cost*((context[:, k].transpose() @ np.linalg.inv(self.V) @ context[:, k]) ** 0.5)

            scalings = self.scale_context_to_safety(UCB_cost)


            
            #ft = self.bandit.err_var * (2 * np.log((np.linalg.det(self.V) ** 0.5) * (np.linalg.det(self.lam * np.identity(self.d)) ** -0.5) /delta)) ** 0.5 + self.lam ** 0.5 * np.linalg.norm(self.bandit.theta)
            ft_reward = 1.0/(min(1, self.bandit.tau))*2.0 * (2 * np.log((np.linalg.det(self.V) ** 0.5) * (np.linalg.det(self.lam * np.identity(self.d)) ** -0.5) /delta)) ** 0.5 + self.lam ** 0.5 * np.linalg.norm(self.bandit.theta)

            UCB_reward = np.zeros(self.K)
            #print("MU hat shape ", self.mu_hat.shape)


            theta_tilde = np.random.multivariate_normal(self.theta_hat.reshape((self.d, )),self.d*(ft_reward**2)*self.V_inverse, 1).reshape((self.d,))
            #theta_tilde = self.theta_hat
            #theta_tilde = np.random.multivariate_normal(self.theta_hat.reshape((self.d, )),.1*self.V_inverse, 1).reshape((self.d,))

            self.estimates_reward = np.matmul(np.squeeze(theta_tilde), context )
            for k in range(self.K):
                UCB_reward[k] = self.estimates_reward[k] #+ ft_reward*((context[:, k].transpose() @ np.linalg.inv(self.V) @ context[:, k]) ** 0.5)

            scaled_UCB_rewards = [UCB_reward[k]*scalings[k] for k in range(self.K)]
            i = np.argmax(scaled_UCB_rewards)
            scaling = scalings[i]
            #print(scalings)
        else:
            raise ValueError("Something strange happened. This condition should never activate.")
        #   pass

        # self.X.append(context[:, i])
        # self.X_numpy = np.array( self.X )
        # self.V += np.outer(context[:, i], context[:,i])

        # if len(self.regrets) <= self.nm_ini:
        #   self.V_inverse = np.linalg.inv(self.V)
        # else:
        #   self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ context[:,i ], context[:,i ]) @ self.V_inverse) / (1 + context[:,i ] @ self.V_inverse @ context[:,i].transpose())
        return i, scaling


    def update_X_and_V(self, context, i, scaling):
        
        #if not self.sgd:
        if not self.compressed:
            self.X.append(scaling*context[:, i])
            self.X_numpy = np.array( self.X )


        self.V += np.outer(scaling*context[:, i], scaling*context[:,i])

        if len(self.regrets) <= self.nm_ini:
            self.V_inverse = np.linalg.inv(self.V)
        else:
            x = scaling*context[:,i]
            #self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ scaling*context[:,i ], scaling*context[:,i ]) @ self.V_inverse) / (1 + scaling*context[:,i ] @ self.V_inverse @ scaling*context[:,i].transpose())
            #self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ context[:,i ], context[:,i ]) @ self.V_inverse) / (1 + context[:,i ] @ self.V_inverse @ context[:,i].transpose())

            self.V_inverse = self.V_inverse - (np.outer(self.V_inverse @ x, x) @ self.V_inverse) / (1 + x @ self.V_inverse @ x.transpose())

    def update_with_reward(self,i,r):
        ## Update the estimates
        pass



    def run_one_step(self):
        context = self.bandit.generate_context()
        i, scaling = self.get_arm(context = context)
        
        self.update_X_and_V(context, i, scaling)
        r = self.bandit.generate_reward(i, scaling)
        r_real = self.bandit.generate_reward(i, scaling, real = True)
        c = self.bandit.generate_cost(i, scaling)
        c_real = self.bandit.generate_cost(i, scaling, real = True)
        
        #if not self.sgd:

        if not self.compressed:
            self.Y.append(r)
            self.C.append(c)
            self.Y_numpy = np.array(self.Y)
            self.C_numpy = np.array(self.C)

        self.X_Y += scaling*context[:, i]*r
        self.X_C += scaling*context[:, i]*c

        #self.estimates[i] += 1. / (self.counts[i] + 1) * (r - self.estimates[i])
        #self.estimates[i] = (self.counts[i]*self.estimates[i] + r)/(self.counts[i] + 1.)
        self.counts[i] += 1


        return context,i, scaling, r_real, c_real




