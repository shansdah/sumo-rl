import numpy as np

class EpsilonGreedy:
    
    def __init__(self, action_list, epsilon_decay):
        
        self.decay = epsilon_decay
        self.action_list = action_list
        self.epsilon = 1.0
        
    def decay_epsilon(self):
        if self.epsilon >= 0.02:
            self.epsilon *= self.decay
        
    def select_action(self, action, agent_index):
        total = self.action_list[agent_index].n
        probabilities = np.ones(total, dtype = float) * self.epsilon / float(total)
        probabilities[action] += (1.0 - self.epsilon)
        r = np.random.choice(np.arange(total), p = probabilities)
        return r
