import argparse
import os
import sys
import pandas as pd
from pathlib import Path


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

import traci

from sumo_rl.agents import Marl
from sumo_rl.util import SumoNetwork
from sumo_rl.util import PlotMetrics

parser = argparse.ArgumentParser(description='Process simulation and dqn parameters')

# Environment simulation parameters
# Network name
parser.add_argument("-net", dest="sumo_net_name", default="3x3grid", type=str, required=False)

# Traffic signal phase duration in seconds
parser.add_argument("-delta", dest="sumo_delta_time", default=10, type=int, required=False)

# Number of seconds per episode
parser.add_argument("-iterations", dest="sumo_simulation_length", default=3600, type=int, required=False)

# Number of simulation episodes
parser.add_argument("-episodes", dest="sumo_simulation_total", default=150, type=int, required=False)

# Period of vehicles spawning on the network in seconds
parser.add_argument("-v_freq", dest="sumo_vehicle_period", default=3.0, type=float, required=False)

# Time interval for accumulation of waiting time in seconds
parser.add_argument("-wt_mem", dest="sumo_wait_memory", default=10000, type=int, required=False)

# Reward for simulation - 0: Queue length, 1: Waiting time, 2: Pressure
parser.add_argument("-reward", dest="sumo_reward", default=1, type=int, required=False)

# Vehicle routing period in seconds - 0 to disable
parser.add_argument("-routing", dest="sumo_routing", default=0, type=int, required=False)

# Rewards accumulated across neighborhood
parser.add_argument("-neighbor", dest="sumo_neighbor_reward", action='store_true')
parser.set_defaults(dqn_communication=False)


# Q-Learning simulation parameters
# DQN discount factor
parser.add_argument("-gamma", dest="dqn_gamma", default=0.99, type=float, required=False)

# DQN learning frequency in iterations
parser.add_argument("-l_freq", dest="dqn_learning_freq", default=1, type=int, required=False)

# DQN target update frequency in iterations
parser.add_argument("-t_freq", dest="dqn_target_update_freq", default=100, type=int, required=False)

# DQN learning start iteration number
parser.add_argument("-l_start", dest="dqn_learning_starts", default=400, type=int, required=False)

# DQN state history sequence length
parser.add_argument("-history_length", dest="dqn_history_length", default=4, type=int, required=False)

# DQN learning batch size
parser.add_argument("-batch_size", dest="dqn_batch_size", default=32, type=int, required=False)

# Enable DQN communication
parser.add_argument("-comm", dest="dqn_communication", action='store_true')
parser.set_defaults(dqn_communication=False)

# Enable double DQN
parser.add_argument("-double", dest="dqn_double", action='store_true')
parser.set_defaults(dqn_double=True)

# Enable dueling network
parser.add_argument("-duel", dest="dqn_dueling", action='store_true')
parser.set_defaults(dqn_duelling=False)

# Load DQN network
parser.add_argument("-load", dest="dqn_model_load", action='store_true')
parser.set_defaults(dqn_model_load=False)



# Replay buffer parameters
# Replaly buffer capacity size
parser.add_argument("-mem_size", dest="rb_memory", default=10000, type=int, required=False)

# Enable prioritized experience replay buffer
parser.add_argument("-pr_buff", dest="rb_prioritized_replay", action='store_true')
parser.set_defaults(rb_prioritized_replay=True)

# Exploration
# Epsilon greedy decay rate
parser.add_argument("-decay", dest="explore_epsilon_decay", default=0.999, type=float, required=False)

#Training
# Execution mode
parser.add_argument("-execute_only", "--list", nargs="*", default = [], required=False)
    
args = parser.parse_args()

# Creation of new simulation directory
directory_contents = os.listdir("results/")
sim_num = 0
for item in directory_contents:

    if os.path.isdir("results/"+item):

        if item.split('_')[0] == "simulation" and sim_num <= int(item.split('_')[1]):
            sim_num = int(item.split('_')[1])

sim_num += 1

# Simulation result folder
result_folder = "simulation_"+str(sim_num)
result_folder1 = "results/"+result_folder+"/"+args.sumo_net_name+"/"
p = Path(result_folder1)
p.mkdir(parents=True, exist_ok=True)

# Simulation plots folder
plots_folder = "results/"+result_folder+"/plots/"
p = Path(plots_folder)
p.mkdir(parents=True, exist_ok=True)

plt = PlotMetrics(None, result_folder1+"tripinfo", plots_folder)

# Save simulation hyperparameters
hyperparams = vars(args)
f = open("results/"+result_folder+"/"+"hyperparams.txt", "a")
f.write('-'.join(key + ":" + str(val) for key, val in hyperparams.items()))
f.close()



# Initialize sumo structures
sumo_net = SumoNetwork(network = args.sumo_net_name,                                    # Network name
                    delta = args.sumo_delta_time,                                       # Phase duration
                    iterations = args.sumo_simulation_length,                           # Simulation length
                    episodes = args.sumo_simulation_total,                              # Number of episodes
                    vehicle_period = args.sumo_vehicle_period,                          # Vehicle spawn period
                    wait_time_memory = args.sumo_wait_memory,                           # Waiting time accumulation period
                    reward = args.sumo_reward,                                          # Reward mode
                    routing = args.sumo_routing,                                        # Rerouting period
                    neighbor = args.sumo_neighbor_reward,                               # Neighbor reward sum
                    results = result_folder1)                                           # Result folder


# Main simulation MARL training loop
for i in range(args.sumo_simulation_total):

    # Generate random trips
    sumo_net.generate_routes(i)
    
    
    if i == 0:
        # Extract network information for all junction:
        # Number of agents
        # State space dimensionality
        # Number of actions
        # Neighborhood of junctions
        agents, agent_state_space_list, agent_action_space_list, agent_adjacency_list = sumo_net.extract_topology()

        # Initialize MARL agent
        marl_agent = Marl(agent_total = agents,                                         # Number of agents 
                        agent_state_space_list = agent_state_space_list,                # State space
                        agent_action_space_list = agent_action_space_list,              # Action space
                        agent_adjacency_list = agent_adjacency_list,                    # Neighborhood of agents
                        marl_gamma = args.dqn_gamma,                                    # Discount factor
                        marl_learning_freq = args.dqn_learning_freq,                    # Learning frequency
                        marl_target_update = args.dqn_target_update_freq,               # Target update frequency
                        marl_learning_starts = args.dqn_learning_starts,                # Learning start
                        marl_history_length = args.dqn_history_length,                  # History length
                        marl_batch_size = args.dqn_batch_size,                          # Batch size
                        rb_memory = args.rb_memory,                                     # Memory capacity
                        explore_decay = args.explore_epsilon_decay,                     # Epsilon decay
                        dqn_comm = args.dqn_communication,                              # Enable communication
                        dqn_double = args.dqn_double,                                   # Enable double DQN
                        dqn_duel = args.dqn_dueling,                                    # Enable dueling DQN
                        per = args.rb_prioritized_replay)                               # Enable prioritized replay

    
    # Run simulation episode
    initial_states = sumo_net.environment_reset()                                       # Environment Get initial states
    infos = []
    done = {'__all__': False}
    marl_agent.set_initial(initial_states)                                              # MARL Set initial states
    iterate = 0
    
    # Simulation loop running all iterations per episode
    while not done['__all__']:
        actions = marl_agent.act()                                                      # MARL Get actions
        state, reward, done, info = sumo_net.environment_step(actions=actions)          # Environment Set actions
        print("iteration num: "+str(i)+"-"+str(iterate))
        marl_agent.save_experience(next_state_list=state, reward_list=reward)           # Save iteration experience
        iterate += 1      
    sumo_net.environment_close(i)                                                       # Save episode results
    marl_agent.save_model("results/"+result_folder+"/"+result_folder)                   # Save episode MARL network

plt.plot_episodes(args.sumo_simulation_total,["tripinfo_waitingTime","tripinfo_duration","tripinfo_rerouteNo","tripinfo_waitingCount"],None)


# Simulation MARL execution
sumo_net.generate_routes(-1)
agents, agent_state_space_list, agent_action_space_list, agent_adjacency_list = sumo_net.extract_topology()

marl_agent = Marl(agent_total = agents, 
                agent_state_space_list = agent_state_space_list, 
                agent_action_space_list = agent_action_space_list, 
                agent_adjacency_list = agent_adjacency_list, 
                marl_gamma = args.dqn_gamma, 
                marl_learning_freq = args.dqn_learning_freq, 
                marl_target_update = args.dqn_target_update_freq, 
                marl_learning_starts = args.dqn_learning_starts, 
                marl_history_length = args.dqn_history_length, 
                marl_batch_size = args.dqn_batch_size, 
                rb_memory = args.rb_memory,
                explore_decay = args.explore_epsilon_decay,
                dqn_comm = args.dqn_communication,
                dqn_double = args.dqn_double,
                dqn_duel = args.dqn_dueling,
                per = args.rb_prioritized_replay)


# Run simulation episode
initial_states = sumo_net.environment_reset()
infos = []
done = {'__all__': False}
marl_agent.load_model("results/"+result_folder+"/"+result_folder)                       # Load saved MARL netowrk
marl_agent.set_initial(initial_states)
iterate = 0
while not done['__all__']:
    actions = marl_agent.act()
    
    state, reward, done, info = sumo_net.environment_step(actions=actions)
    print("iteration num: "+str(iterate))  
    iterate += 1      
sumo_net.environment_close(-1)
