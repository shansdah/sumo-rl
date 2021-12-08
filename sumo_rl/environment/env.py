import os
import sys
from pathlib import Path
from typing import Optional, Union
import sumo_rl
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import gym
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd

from .traffic_signal import TrafficSignal

from gym.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


def env(**kwargs):
    env = SumoEnvironmentPZ(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

parallel_env = parallel_wrapper_fn(env)


class SumoEnvironment(gym.Env):
    """
    SUMO Environment for Traffic Signal Control

    :param net_file: (str) SUMO .net.xml file
    :param route_file: (str) SUMO .rou.xml file
    :param out_csv_name: (str) name of the .csv output with simulation results. If None no output is generated
    :param use_gui: (bool) Wheter to run SUMO simulation with GUI visualisation
    :param begin_time: (int) The time step (in seconds) the simulation starts
    :param num_seconds: (int) Number of simulated seconds on SUMO. The time in seconds the simulation must end.
    :param max_depart_delay: (int) Vehicles are discarded if they could not be inserted after max_depart_delay seconds
    :param delta_time: (int) Simulation seconds between actions
    :param min_green: (int) Minimum green time in a phase
    :param max_green: (int) Max green time in a phase
    :single_agent: (bool) If true, it behaves like a regular gym.Env. Else, it behaves like a MultiagentEnv (https://github.com/ray-project/ray/blob/master/python/ray/rllib/env/multi_agent_env.py)
    :sumo_seed: (int/string) Random seed for sumo. If 'random' it uses a randomly chosen seed.
    :fixed_ts: (bool) If true, it will follow the phase configuration in the route_file and ignore the actions.
    :sumo_warnings: (bool) If False, remove SUMO warnings in the terminal
    """
    CONNECTION_LABEL = 0  # For traci multi-client support

    def __init__(
        self, 
        net_file: str, 
        route_file: str, 
        out_csv_name: Optional[str] = None, 
        use_gui: bool = False, 
        begin_time: int = 0, 
        num_seconds: int = 20000, 
        max_depart_delay: int = 100000,
        time_to_teleport: int = -1, 
        delta_time: int = 5, 
        yellow_time: int = 0, 
        min_green: int = 10, 
        max_green: int = 60, 
        single_agent: bool = False, 
        sumo_seed: Union[str,int] = 'random', 
        fixed_ts: bool = False,
        sumo_warnings: bool = True,
        sumo_wait_time_memory: int = 10000,
        sumo_reward: int = 0,
        sumo_routing: int = 0,
        sumo_results: str = None,
        sumo_episode: int = 0,
    ):

        self._net = net_file
        self._route = route_file
        self.use_gui = use_gui
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')

        assert delta_time > yellow_time, "Time between actions must be at least greater than yellow time."

        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time  # seconds on sumo at each step
        self.max_depart_delay = max_depart_delay  # Max wait time to insert a vehicle
        self.time_to_teleport = time_to_teleport
        self.min_green = min_green
        self.max_green = max_green
        self.yellow_time = yellow_time
        self.single_agent = single_agent
        self.sumo_seed = sumo_seed
        self.fixed_ts = fixed_ts
        self.sumo_warnings = sumo_warnings
        self.label = str(SumoEnvironment.CONNECTION_LABEL)
        SumoEnvironment.CONNECTION_LABEL += 1
        self.sumo = None

        if LIBSUMO:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net])  # Start only to retrieve traffic light information
            conn = traci
        else:
            traci.start([sumolib.checkBinary('sumo'), '-n', self._net], label='init_connection'+self.label)
            conn = traci.getConnection('init_connection'+self.label)
        self.ts_ids = list(conn.trafficlight.getIDList())
        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time, 
                                                  self.yellow_time, 
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  conn) for ts in self.ts_ids}
        conn.close()

        self.vehicles = dict()
        self.reward_range = (-float('inf'), float('inf'))
        self.metadata = {}
        self.spec = EnvSpec('SUMORL-v0')
        self.run = 0
        self.metrics = []
        self.out_csv_name = out_csv_name
        self.observations = {ts: None for ts in self.ts_ids}
        self.rewards = {ts: None for ts in self.ts_ids}
        self.sumo_wait_time_memory = sumo_wait_time_memory 
        self.sumo_reward = sumo_reward
        self.sumo_routing = sumo_routing
        self.sumo_results = sumo_results
        self.sumo_episode = sumo_episode
        if not(self.sumo_routing == 0):
            self.sumo_route_probability = str(1.0)
        else:
            self.sumo_route_probability = str(0.0)
    
    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                     '-n', self._net,
                     '-r', self._route,
                     '--tripinfo-output', self.sumo_results+'tripinfo_'+str(self.sumo_episode),
                     '--max-depart-delay', str(self.max_depart_delay), 
                     '--waiting-time-memory', str(self.sumo_wait_time_memory),
                     '--time-to-teleport', str(self.time_to_teleport),
                     '--device.rerouting.probability', self.sumo_route_probability,
                     '--device.rerouting.period',str(self.sumo_routing)]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
        
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

    def reset(self):
        if self.run != 0:
            self.close()
            self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        self._start_simulation()

        self.traffic_signals = {ts: TrafficSignal(self, 
                                                  ts, 
                                                  self.delta_time, 
                                                  self.yellow_time, 
                                                  self.min_green, 
                                                  self.max_green, 
                                                  self.begin_time,
                                                  self.sumo) for ts in self.ts_ids}
        self.vehicles = dict()

        if self.single_agent:
            return self._compute_observations()[self.ts_ids[0]]
        else:
            return self._compute_observations()

    @property
    def sim_step(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()

    def step(self, action):
        
        # No action, follow fixed TL defined in self.phases
        """
        if not(self.sumo_routing == 0) and (self.sim_step()%self.sumo_routing == 0):
            v_list=traci.vehicle.getIDList()
            for veh_id in v_list:
                 traci.vehicle.rerouteTraveltime(veh_id,True)
        """         
        if action is None or action == {}:
            for _ in range(self.delta_time):
                self._sumo_step()
        else:
            self._apply_actions(action)
            self._run_steps()

        observations = self._compute_observations()
        rewards = self._compute_rewards(self.sumo_reward)
        dones = self._compute_dones()
        self._compute_info()

        if self.single_agent:
            return observations[self.ts_ids[0]], rewards[self.ts_ids[0]], dones['__all__'], {}
        else:
            return observations, rewards, dones, {}

    def _run_steps(self):
        time_to_act = False
        while not time_to_act:
            self._sumo_step()
            for ts in self.ts_ids:
                self.traffic_signals[ts].update()
                if self.traffic_signals[ts].time_to_act:
                    time_to_act = True

    def _apply_actions(self, actions):
        """
        Set the next green phase for the traffic signals
        :param actions: If single-agent, actions is an int between 0 and self.num_green_phases (next green phase)
                        If multiagent, actions is a dict {ts_id : greenPhase}
        """   
        if self.single_agent:
            if self.traffic_signals[self.ts_ids[0]].time_to_act:
                self.traffic_signals[self.ts_ids[0]].set_next_phase(actions)
        else:
            for ts, action in actions.items():
                if self.traffic_signals[ts].time_to_act:
                    self.traffic_signals[ts].set_next_phase(action)

    def _compute_dones(self):
        dones = {ts_id: False for ts_id in self.ts_ids}
        dones['__all__'] = self.sim_step > self.sim_max_time
        return dones
    
    def _compute_info(self):
        info = self._compute_step_info()
        self.metrics.append(info)

    def _compute_observations(self):
        self.observations.update({ts: self.traffic_signals[ts].compute_observation() for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.observations[ts].copy() for ts in self.observations.keys() if self.traffic_signals[ts].time_to_act}

    def _compute_rewards(self, option):
        self.rewards.update({ts: self.traffic_signals[ts].compute_reward(option) for ts in self.ts_ids if self.traffic_signals[ts].time_to_act})
        return {ts: self.rewards[ts] for ts in self.rewards.keys() if self.traffic_signals[ts].time_to_act}

    @property
    def observation_space(self):
        return {ts: self.traffic_signals[ts].observation_space for ts in self.ts_ids}
    
    @property
    def action_space(self):
        return {ts: self.traffic_signals[ts].action_space for ts in self.ts_ids}
    
    def observation_spaces(self, ts_id):
        return self.traffic_signals[ts_id].observation_space
    
    def action_spaces(self, ts_id):
        return self.traffic_signals[ts_id].action_space

    def _sumo_step(self):
        self.sumo.simulationStep()

    def _compute_step_info(self):
        rew_dict = {'reward_'+str(ts): self.traffic_signals[ts].last_reward for ts in self.ts_ids}
        met_dict = {
            'step_time': self.sim_step,
            'total_stopped': sum(self.traffic_signals[ts].get_total_queued() for ts in self.ts_ids),
            'total_wait_time': sum(sum(self.traffic_signals[ts].get_waiting_time_per_lane()) for ts in self.ts_ids)
        }
        met_dict.update(rew_dict)
        return met_dict

    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        self.sumo = None
    
    def __del__(self):
        self.close()
    
    def render(self, mode=None):
        pass
    
    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_'+str(run) + '.csv', index=False)

    # Below functions are for discrete state space

    def encode(self, state, ts_id):
        phase = int(np.where(state[:self.traffic_signals[ts_id].num_green_phases] == 1)[0])
        min_green = state[self.traffic_signals[ts_id].num_green_phases]
        density_queue = [self._discretize_density(d) for d in state[self.traffic_signals[ts_id].num_green_phases + 1:]]
        # tuples are hashable and can be used as key in python dictionary
        return tuple([phase, min_green] + density_queue)

    def _discretize_density(self, density):
        return min(int(density*10), 9)


class SumoEnvironmentPZ(AECEnv, EzPickle):
    metadata = {'render.modes': [], 'name': "sumo_rl_v0"}

    def __init__(self, **kwargs):
        EzPickle.__init__(self, **kwargs)
        self._kwargs = kwargs

        self.seed()
        self.env = SumoEnvironment(**self._kwargs)

        self.agents = self.env.ts_ids
        self.possible_agents = self.env.ts_ids
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        # spaces
        self.action_spaces = {a: self.env.action_spaces(a) for a in self.agents}
        self.observation_spaces = {a: self.env.observation_spaces(a) for a in self.agents}

        # dicts
        self.rewards = {a: 0 for a in self.agents}
        self.dones = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

    def seed(self, seed=None):
        self.randomizer, seed = seeding.np_random(seed)

    def reset(self):
        self.env.reset()
        self.agents = self.possible_agents[:]
        self.agent_selection = self._agent_selector.reset()
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def observe(self, agent):
        obs = self.env.observations[agent].copy()
        return obs

    def state(self):
        raise NotImplementedError('Method state() currently not implemented.')

    def close(self):
        self.env.close()

    def render(self, mode='human'):
        return self.env.render(mode)
    
    def save_csv(self, out_csv_name, run):
        self.env.save_csv(out_csv_name, run)

    def step(self, action):
        if self.dones[self.agent_selection]:
            return self._was_done_step(action)
        agent = self.agent_selection
        if not self.action_spaces[agent].contains(action):
            raise Exception('Action for agent {} must be in Discrete({}).'
                            'It is currently {}'.format(agent, self.action_spaces[agent].n, action))

        self.env._apply_actions({agent: action})

        if self._agent_selector.is_last():
            self.env._run_steps()
            self.env._compute_observations()
            self.rewards = self.env._compute_rewards()
            self.env._compute_info()
        else:
            self._clear_rewards()
        
        done = self.env._compute_dones()['__all__']
        self.dones = {a : done for a in self.agents}

        self.agent_selection = self._agent_selector.next()
        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
