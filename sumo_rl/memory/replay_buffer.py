import numpy as np
import tensorflow as tf
import random



class Memory:

    def __init__(self, rb_agents, rb_memory, rb_batch_size):
        self.exp_num = 0
        self.total_agents = rb_agents
        self.capacity = rb_memory
        self.batch_size = rb_batch_size
        self.buffer = []



    def store(self, arg):
        if self.exp_num == self.capacity:
            self.buffer[self.exp_num%self.capacity] = arg
        else:
            self.buffer.append(arg)
        self.exp_num += 1
        return

    
    
    def sample(self):

        
        idx_list = random.sample(range(min(self.exp_num,self.capacity)), min(self.batch_size,min(self.exp_num,self.capacity)))
        
        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        for agent_index in range(self.total_agents):
            state_batch.append([])
            next_state_batch.append([])
            action_batch.append([])
            reward_batch.append([])            
            for i in idx_list:
                state_batch[agent_index].append(self.buffer[i][0][agent_index])
                next_state_batch[agent_index].append(self.buffer[i][1][agent_index])
                action_batch[agent_index].append(self.buffer[i][2][agent_index])
                reward_batch[agent_index].append(self.buffer[i][3][agent_index])
            state_batch[agent_index] = tf.squeeze(tf.convert_to_tensor(state_batch[agent_index],dtype=tf.float32),[1])
            next_state_batch[agent_index] = tf.squeeze(tf.convert_to_tensor(next_state_batch[agent_index],dtype=tf.float32),[1])
        return 0, (state_batch, next_state_batch, action_batch, reward_batch)
        


class SumTree(object):
    data_pointer = 0
    

    def __init__(self, capacity):

        self.capacity = capacity
        

        self.tree = np.zeros(2 * capacity - 1)
        

        self.data = np.zeros(capacity, dtype=object)
    
    

    def add(self, priority, data):

        tree_index = self.data_pointer + self.capacity - 1


        self.data[self.data_pointer] = data


        self.update (tree_index, priority)


        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  
            self.data_pointer = 0
            

    def update(self, tree_index, priority):

        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority



        while tree_index != 0:
            tree_index = (tree_index - 1) // 2
            self.tree[tree_index] += change
        

    def get_leaf(self, v):
        parent_index = 0


        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1


            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        data_index = leaf_index - self.capacity + 1

        return leaf_index, self.tree[leaf_index], self.data[data_index]
    
    @property
    def total_priority(self):
        return self.tree[0] 


class PerMemory(object):  
    PER_e = 0.01  
    PER_a = 0.6  
    PER_b = 0.4  
    
    PER_b_increment_per_sampling = 0.001
    
    absolute_error_upper = 1.  

    def __init__(self, rb_agents, rb_memory, rb_batch_size):

        self.tree = SumTree(rb_memory)
        self.total_agents = rb_agents
        self.batch_size = rb_batch_size
        


    def store(self, experience):

        max_priority = np.max(self.tree.tree[-self.tree.capacity:])


        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  
        

    def sample(self):

        minibatch = []
        n = self.batch_size
        b_idx = np.empty((n,), dtype=np.int32)


        priority_segment = self.tree.total_priority / n       

        for i in range(n):

            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)


            index, priority, data = self.tree.get_leaf(value)

            b_idx[i]= index

            minibatch.append(data)

        state_batch = []
        next_state_batch = []
        action_batch = []
        reward_batch = []
        for agent_index in range(self.total_agents):
            state_batch.append([])
            next_state_batch.append([])
            action_batch.append([])
            reward_batch.append([])            
            for i in minibatch:
                state_batch[agent_index].append(i[0][agent_index])
                next_state_batch[agent_index].append(i[1][agent_index])
                action_batch[agent_index].append(i[2][agent_index])
                reward_batch[agent_index].append(i[3][agent_index])
            state_batch[agent_index] = tf.squeeze(tf.convert_to_tensor(state_batch[agent_index],dtype=tf.float32),[1])
            next_state_batch[agent_index] = tf.squeeze(tf.convert_to_tensor(next_state_batch[agent_index],dtype=tf.float32),[1])
        
        
        return b_idx, (state_batch, next_state_batch, action_batch, reward_batch)
        

    

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e 
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
