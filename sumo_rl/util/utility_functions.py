import os
import sys
import pandas as pd
from pathlib import Path
from sumo_rl import SumoEnvironment

class SumoNetwork:

    def __init__(self, network, delta, iterations, episodes, vehicle_period, wait_time_memory, reward, routing, neighbor, results):
        
        # Initialize sumo files
        self.sumo_net_file = "nets/"+network+"/"+network+".net.xml"
        self.sumo_route_file = "nets/"+network+"/"+network+".rou.xml"
        self.sumo_result_folder = results
        self.sumo_result_file = results+"dqn_"
        
        # Initialize sumo parameters
        self.sumo_delta = delta
        self.sumo_iterations = iterations
        self.sumo_episodes = episodes
        self.sumo_vehicle_period = vehicle_period
        self.sumo_wait_time_memory = wait_time_memory
        self.sumo_reward = reward
        self.sumo_routing = routing
        self.sumo_neighbor_reward = neighbor

        # sumo environment
        self.env = None
        self.adjacency_list = None

    # Initialize environment and generate routes
    def generate_routes(self, episode):

        # Generate random trips
        os.system("python3 sumo_rl/util/randomTrips.py"+" -n "+self.sumo_net_file+" -e "+str(self.sumo_iterations)+" --period "+str(self.sumo_vehicle_period)+" --fringe-factor 100 "+" --route-file "+self.sumo_route_file)
        
        
        self.env = SumoEnvironment(net_file = self.sumo_net_file,
                    route_file = self.sumo_route_file,
                    out_csv_name = self.sumo_result_file,
                    delta_time = self.sumo_delta,
                    min_green = self.sumo_delta,
                    num_seconds = self.sumo_iterations,
                    sumo_wait_time_memory = self.sumo_wait_time_memory,
                    sumo_reward = self.sumo_reward,
                    sumo_routing = self.sumo_routing,
                    sumo_results = self.sumo_result_folder,
                    sumo_episode = episode)
                 


    # Extract junctiion topology
    def extract_topology(self):
        
        sumo_junctions = [int(i) for i in self.env.ts_ids]
        sumo_phase_junctions = self.env.action_space
        sumo_feature_junctions = self.env.observation_space
        
        os.system("netconvert -s "+ self.sumo_net_file +" --plain-output-prefix temp/plain")
        os.system("python3 sumo_rl/util/xml2csv.py temp/plain.edg.xml")
        df = pd.read_csv("temp/plain.edg.csv", usecols=["edge_from", "edge_to"], sep=';')

        
        agent_adjacency_list = []
        for i in sumo_junctions:
            agent_adjacency_list.append([])
        
        for i in range(df.shape[0]):

            if df.edge_from[i] <= max(sumo_junctions) and df.edge_to[i] <= max(sumo_junctions):
                agent_adjacency_list[df.edge_to[i]].append(df.edge_from[i])
        
        self.adjacency_list = agent_adjacency_list
        agent_state_space_list = self.dict_to_list(sumo_feature_junctions)
        agent_action_space_list = self.dict_to_list(sumo_phase_junctions)
    
        return len(sumo_junctions), agent_state_space_list, agent_action_space_list, agent_adjacency_list
        
    # Reset environment to return initial states
    def environment_reset(self):
        initial_state = self.env.reset()
        return self.dict_to_list(initial_state)
    # Single step of the environment
    def environment_step(self, actions):
        state, reward, done, info = self.env.step(action=self.list_to_dict(actions))
        reward_list = self.dict_to_list(reward)
        reward_list_copy = reward_list.copy()
        if self.sumo_neighbor_reward:
            junction_index = 0
            for i in self.adjacency_list:
                for j in i:
                    reward_list_copy[junction_index] += reward_list[j]
                junction_index += 1
        
        return self.dict_to_list(state), reward_list_copy, done, info
    # Close environment and save results
    def environment_close(self,i):
        self.env.save_csv(self.sumo_result_file, i)
        self.env.close()
        os.system("python3 sumo_rl/util/xml2csv.py "+self.sumo_result_folder+'tripinfo_'+str(i)+" --output "+self.sumo_result_folder+'tripinfo_'+str(i))
    # Utility functions
    def dict_to_list(self, arg):
        ret = [None]*len(arg)
        for idx in arg.keys():
            ret[int(idx)] = arg[idx]
        return ret
        
    def list_to_dict(self, arg):
        return {str(idx): arg[idx] for idx in range(len(arg))}


