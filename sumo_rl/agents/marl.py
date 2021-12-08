import numpy as np
from sumo_rl.memory import Memory, PerMemory
from sumo_rl.networks import Network
from sumo_rl.exploration import EpsilonGreedy
import tensorflow as tf
tf.compat.v1.enable_eager_execution()



# Definition of MARL agents
class Marl:

    def __init__(self, agent_total, agent_state_space_list, agent_action_space_list, agent_adjacency_list, marl_gamma, marl_learning_freq, marl_target_update, marl_learning_starts, marl_history_length, marl_batch_size, rb_memory, explore_decay, dqn_comm, dqn_double, dqn_duel, per):
        tf.device('/gpu:1')
        
        # Initialize MARL agents
        self.marl_num_agents = agent_total                                                      # Number of agents
        self.marl_state_space = agent_state_space_list                                          # State space
        self.marl_action_space = agent_action_space_list                                        # Action space
        self.marl_adjacency = agent_adjacency_list                                              # Agent neighborhood
        self.marl_gamma = marl_gamma                                                            # Discount factor
        self.marl_learning_freq = marl_learning_freq                                            # Learning frequency
        self.marl_target_update = marl_target_update                                            # Target update frequency
        self.marl_learning_starts = marl_learning_starts                                        # Learning start
        self.marl_history_length = marl_history_length                                          # State history length
        self.marl_batch_size = marl_batch_size                                                  # Batch size
        
        self.dqn_comm = dqn_comm                                                                # Enable communication
        self.dqn_double = dqn_double                                                            # Enable DDQN target
        self.dqn_duel = dqn_duel                                                                # Enable dueling network
        self.per = per                                                                          # Enable prioritized replay
        
        # Initialize Replay Buffer
        # Prioritized replay buffer
        if self.per:
            self.buffer = PerMemory(rb_agents = self.marl_num_agents,                           # Number of agents 
                                rb_memory = rb_memory,                                          # Capacity
                                rb_batch_size = self.marl_batch_size)                           # Batch size
        # Sequential buffer
        else:                        
            self.buffer = Memory(rb_agents = self.marl_num_agents, 
                                rb_memory = rb_memory, 
                                rb_batch_size = self.marl_batch_size)
        # Initialize Exploration
        self.explore = EpsilonGreedy(action_list = self.marl_action_space,                      # Action space
                                    epsilon_decay = explore_decay)                              # Exploration decay rate

        # Initialize Deep Q Networks
        # Communication DQN network
        if self.dqn_comm:
            self.q_net = Network.build_dqn_model_communication2(self.marl_state_space, 
                                                            self.marl_action_space, 
                                                            self.marl_adjacency,
                                                            self.marl_history_length,
                                                            self.dqn_duel,
                                                            0)
                                                        
            self.target_q_net = Network.build_dqn_model_communication2(self.marl_state_space, 
                                                                    self.marl_action_space, 
                                                                    self.marl_adjacency,
                                                                    self.marl_history_length,
                                                                    self.dqn_duel,
                                                                    1)
        # Vanilla DQN network
        else:
            self.q_net = Network.build_dqn_model_vanilla(self.marl_state_space, 
                                                        self.marl_action_space, 
                                                        self.marl_adjacency,
                                                        self.marl_history_length,
                                                        self.dqn_duel,
                                                        0)
                                                        
            self.target_q_net = Network.build_dqn_model_vanilla(self.marl_state_space, 
                                                                self.marl_action_space, 
                                                                self.marl_adjacency,
                                                                self.marl_history_length,
                                                                self.dqn_duel,
                                                                1)

        # Initialize local variables
        self.marl_state = []                                                                    # MARL state history
        self.marl_action = None                                                                 # MARL current action
        self.marl_epsilon = 1.0                                                                 # Initial epsilon
        self.marl_experience_num = 0                                                            # Number of experiences
        self.marl_experience_history = 0                                                        # Experience sequence size

    # MARL set initial states
    def set_initial(self, state):
        self.marl_state = []
        for i in range(self.marl_num_agents):
            self.marl_state.append([])
        self.add_state(state)
        self.marl_experience_num += 1
        self.marl_experience_history = 1
        
    # Add new experience
    def add_state(self, state):
        for i in range(self.marl_num_agents):
            self.marl_state[i].append(state[i])
            if len(self.marl_state[i]) > self.marl_history_length + 1:
                del self.marl_state[i][0]
            
    # Get previous state
    def get_state(self):
        state_list = []
        for i in range(self.marl_num_agents):
            state_list.append(self.marl_state[i][-self.marl_history_length-1:-1])
            state_list[i] = tf.expand_dims(tf.convert_to_tensor(state_list[i], dtype=tf.float32), axis=0)
        return state_list        
    
    # Get latest state
    def get_next_state(self):
        next_state_list = []
        for i in range(self.marl_num_agents):
            next_state_list.append(self.marl_state[i][-self.marl_history_length:])
            next_state_list[i] = tf.expand_dims(tf.convert_to_tensor(next_state_list[i], dtype=tf.float32), axis=0)
        return next_state_list
    
    # Save experience to replay buffer
    def save_experience(self, next_state_list, reward_list):
        
        # Target update condition
        if self.marl_experience_num % self.marl_target_update == 0:
            self.update_target()

        # Network learning condition
        if self.marl_experience_num % self.marl_learning_freq == 0 and self.marl_experience_num > self.marl_learning_starts:
            self.learn()      
            self.explore.decay_epsilon()  
            
        self.add_state(next_state_list)
        self.marl_experience_num += 1
        self.marl_experience_history += 1

        # Historical experience store condition
        if self.marl_experience_history >= self.marl_history_length+1:
            # Store experience
            self.buffer.store((self.get_state(), 
                                self.get_next_state(), 
                                self.marl_action, 
                                reward_list))
    
    # Execution of Q-network
    def act(self):
        
        # Select random action condition
        if (self.marl_experience_history < self.marl_history_length):
            self.marl_action = [0]*self.marl_num_agents
            return self.marl_action
            
        # Get latest state
        net_input = self.get_next_state()

        # Evaluate network to get action values
        net_output = self.q_net(net_input)

        self.marl_action = []
        
        # Apply epsilon greedy exploration
        agent_index = 0
        for i in net_output:
            self.marl_action.append(self.explore.select_action(np.argmax(i.numpy()[0], axis=0), agent_index))
            agent_index += 1
        return self.marl_action


    # Training of Q-network
    def learn(self):
        
        # Sample experiences
        idx, samples = self.buffer.sample()

        net_inputs = samples[0]
        net_inputs_next = samples[1]
        net_actions = samples[2]
        net_rewards = samples[3]

        # Define targets for training
        q_current = self.q_net(net_inputs)
        net_targets = []
        for agent_index in range(self.marl_num_agents):
            net_targets.append(np.copy(q_current[agent_index]))
        q_next = self.q_net(net_inputs_next)
        target_q_next = self.target_q_net(net_inputs_next)
        max_q_next = []
        for agent_index in range(self.marl_num_agents):
            max_q_next.append(np.argmax(q_next[agent_index], axis=1))

        absolute_errors = []
        for agent_index in range(self.marl_num_agents):
            absolute_errors.append([])
            for i in range(self.marl_batch_size):
                # Double DQN targets
                if self.dqn_double:
                    net_targets[agent_index][i][net_actions[agent_index][i]] = net_rewards[agent_index][i] + self.marl_gamma * target_q_next[agent_index][i][max_q_next[agent_index][i]]
                # DQN targets
                else:
                    net_targets[agent_index][i][net_actions[agent_index][i]] = net_rewards[agent_index][i] + self.marl_gamma * np.amax(q_next[agent_index][i])
                #.Prioritized replay priorities
                absolute_errors[agent_index].append(np.abs(q_current[agent_index][i][net_actions[agent_index][i]] - net_targets[agent_index][i][net_actions[agent_index][i]]))
                
        # Set priority value of experience to the maximum among agents
        absolute_errors = np.max(absolute_errors, 0)
        
        # Update priorities of sampled experiences
        if self.per:
            self.buffer.batch_update(idx, absolute_errors)        
        
        # Q-network training
        result = self.q_net.fit(x=net_inputs, y=net_targets, batch_size = self.marl_batch_size, epochs=1, shuffle = True)
        
        

    # Save DQN model
    def save_model(self, filename):
        self.q_net.save(filename)
                
    # Update DQN target
    def update_target(self):
        self.target_q_net.set_weights(self.q_net.get_weights())