import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, BatchNormalization, Lambda, Add, LayerNormalization, Input, LSTM, Concatenate
from tensorflow.keras.losses import Huber
from tensorflow.keras import backend as K
import numpy as np

tf.compat.v1.enable_eager_execution()

# Predefined shared dimensions, global variables not extracted dynamically
SHARED_MESSAGE_LENGTH = 8
SHARED_OUTPUT_SIZE = 4
ATTENTION_DIMENSION = 64

class Network:

    # Build vanilla DQN network
    def build_dqn_model_vanilla(input_list, output_list, communication_list, window_size, duel, target_net):
        agent_index = 0
        inputs = []
        inputs1 = []
        
        # Configure inputs for each agent
        for i in input_list:
            inputs.append(Input(shape = (window_size, len(i.high),)))                           #(batches,time,dimension)
            input_temp1 = tf.slice(inputs[agent_index], [0,window_size-1,0],[-1,1,-1])          
            input_temp2 = tf.transpose(input_temp1, [1,0,2])
            inputs1.append(tf.squeeze(input_temp2,[0]))                                         #(batches,dimension)
            
            agent_index +=1
            
        agent_index = 0
        outs = []
        outs1 = []
        outs2 = []
        state_value = []
        action_advantage = []
        
        # Configure outputs for each agent
        for i in output_list:
            outs1.append(layer1_t(inputs1[agent_index]) if target_net else layer1_v(inputs1[agent_index]))
            outs2.append(layer2_t(outs1[agent_index]) if target_net else layer2_v(outs1[agent_index]))

            # Dueling network
            if duel:
                
                # State values
                state_value.append(layer_value_t(outs2[agent_index]) if target_net else layer_value_v(outs2[agent_index]))
                state_value[agent_index] = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(i.n,))(state_value[agent_index])

                # Action advantages
                action_advantage.append(layer_advantage_t(outs2[agent_index]) if target_net else layer_advantage_v(outs2[agent_index]))
                action_advantage[agent_index] = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(i.n,))(action_advantage[agent_index])
                
                outs.append(Add()([state_value[agent_index], action_advantage[agent_index]]))
            
            # DQN network
            else:
                outs.append(layer0_t(outs2[agent_index]) if target_net else layer0_v(outs2[agent_index]))
                
            agent_index += 1        
    
        q_net = Model(inputs=inputs, outputs=outs)
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,clipvalue=40),  loss=Huber(delta=1.0))
        q_net.summary()
        return q_net
        
        
        
        
    def build_dqn_model_communication(input_list, output_list, communication_list, window_size, duel, target_net):
        # Configure inputs for each agent
        agent_index = 0
        inputs = []
        for i in input_list:
            inputs.append(Input(shape = (window_size,len(i.high),)))
            agent_index += 1
        
        # Configure RNN for each agent
        agent_index = 0
        hidden1 = []
        for i in input_list:
            whole_seq_output = shared_lstm_t(inputs[agent_index]) if target_net else shared_lstm(inputs[agent_index])
            hidden1.append(whole_seq_output)
            agent_index += 1
  
            
        # Configure messages to send for each agent
        agent_index = 0
        hidden2 = []    
        for i in communication_list:
            hidden2_temp = []
            
            for j in i:
                hidden2_temp2 = tf.slice(inputs[j], [0,window_size-1,0],[-1,1,-1])      #(batch,time,dimension)
                hidden2_temp.append(hidden2_temp2)
                
            hidden2_temp1 = tf.convert_to_tensor(hidden2_temp, dtype=tf.float32)        #(neighbor,batch,time,dimension)
            hidden2_temp2 = tf.transpose(hidden2_temp1,[2,1,0,3])                       #(time,batch,neighbor,dimension)

            
            hidden2_temp3 = tf.squeeze(hidden2_temp2,[0])                               #(batch,neighbor,dimension)
                                           
            positional_encoding_handler = PositionalEncoding(window_size, len(input_list[agent_index].high))
            positional_encoding_handler2 = PositionalEncoding(len(i), len(input_list[agent_index].high))
            positional_encoding = positional_encoding_handler.get_positional_encoding()
            positional_encoding2 = positional_encoding_handler2.get_positional_encoding()
            
            context = AttentionLayer(ATTENTION_DIMENSION)(hidden2_temp3+positional_encoding2, inputs[agent_index]+positional_encoding, hidden1[agent_index])
            hidden2.append(context)
            agent_index += 1
            
        # Configure messages to receive for each agent
        agent_index = 0
        hidden3_temp = []
        hidden3 = []
        for i in communication_list:
            hidden3_temp = []
            a = tf.squeeze(tf.slice(inputs[agent_index], [0,window_size-1,0],[-1,1,-1]),[1])
            hidden3_temp7 = []
            for j in i:
                index = 0
                for k in communication_list[j]:

                    if agent_index == k:
                        
                        hidden3_temp1 = tf.transpose(hidden2[j],[1,0,2])                #(neighbor,batch,dimension)
                        hidden3_temp6 = tf.slice(hidden3_temp1, [index,0,0],[1,-1,-1])
                        hidden3_temp2 = tf.squeeze(hidden3_temp6,[0])                   #(batch,dimension)
                        hidden3_temp7.append(hidden3_temp2)
                        continue
                    index += 1
            
            hidden3_temp8 = Add()(hidden3_temp7)
            hidden3_temp9 = Concatenate()([a,hidden3_temp8])

            hidden3.append(hidden3_temp9)
            agent_index += 1    
                    
        # Configure outputs for each agent
        agent_index = 0
        outs = []
        outs1 = []
        outs2 = []
        outs3 = []
        outs4 = []
        outs5 = []
        state_value = []
        action_advantage = []
        for i in output_list:
            outs1.append(layer1_t_comm(hidden3[agent_index]) if target_net else layer1_v_comm(hidden3[agent_index]))
            outs2.append(layer2_t_comm(outs1[agent_index]) if target_net else layer2_v_comm(outs1[agent_index]))
            
            # Dueling network
            if duel:
                # State Values
                state_value.append(layer_value_t(outs2[agent_index]) if target_net else layer_value_v(outs2[agent_index]))
                state_value[agent_index] = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(i.n,))(state_value[agent_index])
                # Action Advantages
                action_advantage.append(layer_advantage_t(outs2[agent_index]) if target_net else layer_advantage_v(outs2[agent_index]))
                action_advantage[agent_index] = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(i.n,))(action_advantage[agent_index])
                
                outs.append(Add()([state_value[agent_index], action_advantage[agent_index]]))
            
            # DQN network
            else:
                outs.append(layer0_t(outs2[agent_index]) if target_net else layer0_v(outs2[agent_index]))
                
            agent_index += 1    
        q_net = Model(inputs=inputs, outputs=outs)
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,clipvalue=40),  loss=Huber(delta=1.0))
        q_net.summary()
        return q_net

        

        
        
    def build_dqn_model_communication2(input_list, output_list, communication_list, window_size, duel, target_net):
        # Input layer 0
        agent_index = 0
        inputs = []
        for i in input_list:
            inputs.append(Input(shape = (window_size,len(i.high),)))
            agent_index += 1
        
        hidden4 = []
        agent_index = 0
        for i in input_list:
            whole_seq_output = layer1_t(inputs[agent_index]) if target_net else layer1_v(inputs[agent_index])
            hidden4.append(whole_seq_output)
            agent_index += 1
            
        # LSTM layer 1
        agent_index = 0
        hidden1 = []
        for i in input_list:
            whole_seq_output = shared_lstm_t(hidden4[agent_index]) if target_net else shared_lstm(hidden4[agent_index])
            hidden1.append(whole_seq_output)
            agent_index += 1
  
            
        # Communication source layer 2                      
        
        agent_index = 0
        hidden2 = []    
        for i in communication_list:
            hidden2_temp = []    
            for j in i:
                hidden2_temp2 = tf.slice(hidden1[j], [0,window_size-1,0],[-1,1,-1])     #(batch,time,dimension)
                hidden2_temp.append(hidden2_temp2)
                
            hidden2_temp1 = tf.convert_to_tensor(hidden2_temp, dtype=tf.float32)        #(neighbor,batch,time,dimension)
            hidden2_temp2 = tf.transpose(hidden2_temp1,[2,1,0,3])                       #(time,batch,neighbor,dimension)

            
            hidden2_temp3 = tf.squeeze(hidden2_temp2,[0])                               #(batch,neighbor,dimension)
                                           
            positional_encoding_handler = PositionalEncoding(window_size, len(input_list[agent_index].high))
            positional_encoding_handler2 = PositionalEncoding(len(i), len(input_list[agent_index].high))
            positional_encoding = positional_encoding_handler.get_positional_encoding()
            positional_encoding2 = positional_encoding_handler2.get_positional_encoding()
            
            context = AttentionLayer(ATTENTION_DIMENSION)(hidden2_temp3+positional_encoding2, hidden1[agent_index]+positional_encoding, hidden1[agent_index])
            hidden2.append(context)
            agent_index += 1
            
        # Communication destination layer 3    
        agent_index = 0
        hidden3_temp = []
        hidden3 = []
        for i in communication_list:
            hidden3_temp = []
            a = tf.squeeze(tf.slice(hidden1[agent_index], [0,window_size-1,0],[-1,1,-1]),[1])
            hidden3_temp7 = []
            index1 = 0
            for j in i:
                index = 0
                for k in communication_list[j]:

                    if agent_index == k:
                        
                        hidden3_temp1 = tf.transpose(hidden2[j],[1,0,2])                #(neighbor,batch,dimension)
                        hidden3_temp6 = tf.slice(hidden3_temp1, [index,0,0],[1,-1,-1])
                        hidden3_temp2 = tf.squeeze(hidden3_temp6,[0])                   #(batch,dimension)
                        hidden3_temp7.append(hidden3_temp2)
                        continue
                    index += 1
                index1 += 1
            hidden3_temp8 = Add()(hidden3_temp7)
            hidden3_temp9 = Add()([a,hidden3_temp8])

            hidden3.append(hidden3_temp9)
            agent_index += 1    
                    
        # Final DQN layer
        agent_index = 0
        outs = []
        outs1 = []
        outs2 = []
        outs3 = []
        outs4 = []
        outs5 = []
        state_value = []
        action_advantage = []
        for i in output_list:
            outs1.append(layer1_t_comm(hidden3[agent_index]) if target_net else layer1_v_comm(hidden3[agent_index]))
            outs2.append(layer2_t_comm(outs1[agent_index]) if target_net else layer2_v_comm(outs1[agent_index]))

            if duel:
                state_value.append(layer_value_t(outs2[agent_index]) if target_net else layer_value_v(outs2[agent_index]))
                state_value[agent_index] = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(i.n,))(state_value[agent_index])

                action_advantage.append(layer_advantage_t(outs2[agent_index]) if target_net else layer_advantage_v(outs2[agent_index]))
                action_advantage[agent_index] = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(i.n,))(action_advantage[agent_index])
                
                outs.append(Add()([state_value[agent_index], action_advantage[agent_index]]))
            else:
                outs.append(layer0_t(outs2[agent_index]) if target_net else layer0_v(outs2[agent_index]))
                
            agent_index += 1    
        q_net = Model(inputs=inputs, outputs=outs)
        q_net.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001,clipvalue=40),  loss=Huber(delta=1.0))
        q_net.summary()
        return q_net
        
        
        
        
######################################################################################################################################################

# Global shared hidden layers for all agents
layer1_t = tf.keras.Sequential(
   [
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None
        ),
        Dense(32, activation=tf.nn.relu),
    ],
)
layer1_v = tf.keras.Sequential(
   [
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None
        ),
        Dense(32, activation=tf.nn.relu),
    ],
)

layer2_t = tf.keras.Sequential(
   [
        
        Dense(16, activation=tf.nn.relu),

    ],
)

layer2_v = tf.keras.Sequential(
   [
        
        Dense(16, activation=tf.nn.relu),

    ],
)
layer1_t_comm = tf.keras.Sequential(
   [
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None
        ),
        Dense(64, activation=tf.nn.relu),
    ],
)
layer1_v_comm = tf.keras.Sequential(
   [
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None
        ),
        Dense(64, activation=tf.nn.relu),
    ],
)

layer2_t_comm = tf.keras.Sequential(
   [
        
        Dense(32, activation=tf.nn.relu),

    ],
)

layer2_v_comm = tf.keras.Sequential(
   [
        
        Dense(32, activation=tf.nn.relu),

    ],
)

# Global shared output layers for all agents
layer0_t = tf.keras.Sequential(
   [
       
        Dense(SHARED_OUTPUT_SIZE, activation='linear'),

    ],
)

layer0_v = tf.keras.Sequential(
   [
       
        Dense(SHARED_OUTPUT_SIZE, activation='linear'),
       
    ],
)

layer_value_t = tf.keras.Sequential(
   [
       
        Dense(1, kernel_initializer='he_uniform'),
       
    ],
)
layer_value_v = tf.keras.Sequential(
   [
       
        Dense(1, kernel_initializer='he_uniform'),
       
    ],
)
layer_advantage_t = tf.keras.Sequential(
   [
       
        Dense(SHARED_OUTPUT_SIZE, kernel_initializer='he_uniform'),
       
    ],
)
layer_advantage_v = tf.keras.Sequential(
   [
       
        Dense(SHARED_OUTPUT_SIZE, kernel_initializer='he_uniform'),
       
    ],
)



#######################################################################################################################################

# Positional encoding for vector
class PositionalEncoding(object):
    # positon: number of distinct positions for each vector
    # d: dimensionality of vectors 
    def __init__(self, position, d):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        self._encoding = np.concatenate([sines, cosines], axis=1)
        self._encoding = self._encoding[np.newaxis,]
    
    def _get_angles(self, position, i, d):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d))
        return position * angle_rates
    
    def get_positional_encoding(self):
        return tf.cast(self._encoding, dtype=tf.float32)
        
# Definition of attention layer
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units):
        super( AttentionLayer, self).__init__()
        self.W1 = Dense(units)  
        self.W2 = Dense(units)  
        self.V = Dense(1)

        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'value_weights': self.W1,
            'query_weights': self.W2,
            'score_weights': self.V,
        })
        return config


    # Query:(neighbors,dimension)
    # Key: (time,dimension)
    # Value: (time,dimension)
    def call(self, query, key, values):                                         
        #calculate the Attention score      
        a=tf.transpose(key, [0,1,2])                                            #(neighbors,dimension)
        b=tf.transpose(query, [0,1,2])                                          #(time,dimension)
        c = self.W2(a)                                                          #(neighbors,attention_dimension)
        d = self.W1(b)                                                          #(time,attention_dimension)

        x = tf.matmul(tf.transpose(d, [0,1,2]) , tf.transpose(c, [0,2,1]))      #(neighbors,time)

        
        score = (tf.nn.tanh(tf.transpose(x, [0,1,2])))                      
        
        attention_weights = tf.nn.softmax(score, axis=1)
        
        context_vector= tf.matmul(attention_weights , values)                   #(neighbors,dimension)
       
        return context_vector

shared_attention = AttentionLayer(ATTENTIN_DIMENSION)

# Global shared LSTM layers
shared_lstm = tf.keras.Sequential(
   [
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None
        ),

        LSTM(SHARED_MESSAGE_LENGTH*SHARED_OUTPUT_SIZE,return_sequences=True),
    ],
)
shared_lstm_t = tf.keras.Sequential(
   [
        tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=0.001, center=True, scale=True,
            beta_initializer='zeros', gamma_initializer='ones',
            beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
            gamma_constraint=None
        ),

        LSTM(SHARED_MESSAGE_LENGTH*SHARED_OUTPUT_SIZE,return_sequences=True),
    ],
)
