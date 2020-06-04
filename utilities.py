import numpy as np
import copy
import scipy.stats


class Arm:
  def get_sample(self):
    pass
  def get_mean(self):
    return self.mean

class SimpleDiscreteArm(Arm):
  def __init__(self, reward_probabilities, reward_values ):
    self.reward_probabilities = reward_probabilities
    self.reward_values = reward_values
    self.cummulative_probabilities = np.zeros(len(reward_probabilities))
    cum_prob = 0
    for i,prob in enumerate(self.reward_probabilities):
      cum_prob += prob
      self.cummulative_probabilities[i] = cum_prob
    self.mean = sum(r*p for (r,p) in zip(self.reward_probabilities, self.reward_values))
  def get_sample(self):
      val = np.random.random()
      index = 0  
      while index <= len(self.cummulative_probabilities)-1:
        if val < self.cummulative_probabilities[index]:
          break
        index += 1
      return self.reward_values[index]

class SimpleGaussianArm(Arm):
  def __init__(self, mean, std, truncated = False):
    self.mean = mean
    self.std = std
    self.truncated = truncated
  def get_sample(self):
    sample = np.random.normal(self.mean, self.std)
    if self.truncated:
      if sample > 1: 
        return 1
      if sample < 0:
        return 0
    return sample


class DoubleMultiArm:
  def __init__(self, reward_arms, cost_arms):
    self.reward_arms = reward_arms
    self.cost_arms = cost_arms
    self.num_arms = len(self.reward_arms)
    
  def get_rewards(self, arm_index):
    return (self.reward_arms[arm_index].get_sample(), self.cost_arms[arm_index].get_sample())
  def get_means(self):
    return ([arm.get_mean() for arm in self.reward_arms], [arm.get_mean() for arm in self.cost_arms])
  
  def evaluate_policy(self, policy):
    reward_mean = 0
    cost_mean = 0 
    [mean_rewards, mean_cost] = self.get_means()
    for i in range(len(policy)):
      reward_mean += policy[i]*mean_rewards[i]
      cost_mean += policy[i]*mean_cost[i]
    return reward_mean, cost_mean

  def get_optimal_policy(self, threshold):
    [mean_rewards, mean_cost] = self.get_means()
    c = -np.array(mean_rewards)
    A_ub = np.array([mean_cost])
    b_ub = np.array([threshold])
    A_eq = np.array([np.ones(self.num_arms)])
    b_eq = np.array([1])
    bounds = [(0, 1) for _ in range(self.num_arms)]
    opt = scipy.optimize.linprog(c = c, A_ub = A_ub , b_ub = b_ub , A_eq =A_eq, b_eq = b_eq  )
    return opt.x


class ConstrainedBandit:
  def __init__(self, initial_rewards_means, initial_cost_means, threshold, T, known_arms_indicator, alpha_r, alpha_c, do_UCB = False):
    self.rewards_means = initial_rewards_means
    self.num_arms = len(self.rewards_means)
    self.do_UCB = do_UCB
    self.initial_cost_means = copy.copy(initial_cost_means)

    self.threshold = threshold
    self.alpha_r = alpha_r
    self.alpha_c = alpha_c

    self.cost_means = initial_cost_means
    self.num_arm_pulls = [0 for _ in range(len(self.rewards_means))]


    self.known_arms_indices = []
    for i in range(self.num_arms):
      if known_arms_indicator[i]:
        self.known_arms_indices.append(i)

    self.T = T
    self.upper_rewards_means = [min(r + self.alpha_r*self.confidence_interval(0,1.0/T**2), 1)  for r in self.rewards_means]
    self.upper_cost_means = [min(c + self.alpha_c*self.confidence_interval(0, 1.0/T**2), 1) for c in self.cost_means]

    self.fix_known_arms()

  def fix_known_arms(self):
    for index in self.known_arms_indices:
      self.upper_cost_means[index] = self.initial_cost_means[index]#self.cost_means[index]
      #self.upper_rewards_means[index] = self.rewards_means[index]
 

  def confidence_interval(self, t, delta):
    return np.sqrt(2*np.log(1/delta)/max(t, 1))

  def update(self, index, reward, cost):
    old_pulls = self.num_arm_pulls[index]
    self.rewards_means[index] = (self.rewards_means[index]*old_pulls + reward)/(old_pulls + 1)
    self.cost_means[index] = (self.cost_means[index]*old_pulls + cost)/(old_pulls + 1)
    self.num_arm_pulls[index] = old_pulls + 1

    #print("Confidence interval sizes ", [self.confidence_interval(self.num_arm_pulls[index],1.0/self.T**2) for index in range(self.num_arms)])

    self.upper_rewards_means = [min(self.rewards_means[index] +self.alpha_r*self.confidence_interval(self.num_arm_pulls[index],1.0/self.T**2), 1)  for index in range(self.num_arms)]
    self.upper_cost_means =  [min(self.cost_means[index] + self.alpha_c*self.confidence_interval(self.num_arm_pulls[index],1.0/self.T**2), 1) for index in range(self.num_arms)]

    #self.upper_rewards_means = [min(self.rewards_means[index] +self.alpha_r*self.confidence_interval(self.num_arm_pulls[index],1.0/10), 1)  for index in range(self.num_arms)]
    #self.upper_cost_means =  [min(self.cost_means[index] + self.alpha_c*self.confidence_interval(self.num_arm_pulls[index],1.0/10), 1) for index in range(self.num_arms)]

    self.fix_known_arms()

  def get_policy(self):
    if self.do_UCB:
      ucb_index = np.argmax(self.upper_rewards_means)
      policy = np.zeros(len(self.upper_rewards_means))
      policy[ucb_index] = 1
      #return policy
    else:
      
      c = -np.array(self.upper_rewards_means)
      A_ub = np.array([self.upper_cost_means])
      b_ub = np.array([self.threshold])
      A_eq = np.array([np.ones(self.num_arms)])
      b_eq = np.array([1])

      bounds = [(-0.000000001, 2) for _ in range(self.num_arms)]
      opt = scipy.optimize.linprog(c = c, A_ub = A_ub , b_ub = b_ub , A_eq =A_eq, b_eq = b_eq  )
      policy = opt.x
    # print("###################################")
    # print("policy ", policy, " upper rewards means ", self.upper_rewards_means)
    # print("cost means ", self.cost_means, " reward means ", self.rewards_means)
    # print("upper cost means ", self.upper_cost_means, " initial cost means ", self.initial_cost_means)
    return policy

  def get_arm_index(self):
    policy = self.get_policy()
    cumulative_probabilities = np.zeros(self.num_arms)
    cum_prob = 0
    for i,prob in enumerate(policy):
      cum_prob += prob
      cumulative_probabilities[i] = cum_prob

    val = np.random.random()
    index = 0  
    while index <= len(cumulative_probabilities)-1:
      if val < cumulative_probabilities[index]:
          break
      index += 1
    return index