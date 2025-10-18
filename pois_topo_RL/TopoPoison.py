#!/usr/bin/env python

import gymnasium as gym
from gymnasium import spaces
from gymnasium import Env
#import gym
#from gym import Env
from gymnasium.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete

#import helper
import numpy as np
import random
import os
import networkx as nx 
import matplotlib.pyplot as plt

#import stable baselines stuff
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
import os
import utilities as util
import sim_util as sim_util


class TopoPoisoningEnv(Env):
    def __init__(self, vul_node, node_num, action_space_size, max_degree, sh_paths, path_num, adj_matrix, ori_adj_matrix, st_list, neighbours, baseline, expectation, best_topo, expected_sim, step_len,proj_name):
        self.vul_node = vul_node
        #initiate random paths
        self.st_list = st_list
        self.ori_adj_matrix = ori_adj_matrix
        self.baseline = baseline
        self.neighbours = neighbours
        self.sh_path_list = sh_paths
        self.step_length = step_len
        self.expectation = expectation
        self.init_step_len = step_len
        self.init_best_topo = best_topo
        self.init_baseline = baseline
        self.init_sh_paths = sh_paths
        self.expected_sim = expected_sim
        self.action_space = Discrete(action_space_size)
        self.observation_space = Box(low=0,high=max_degree,shape=(node_num,node_num), dtype=int)
        self.state = adj_matrix #(recent topo, gained number of eavesdropped flows)
        self.p_name = proj_name
        self.best = {"adj_matrix":best_topo, "paths":sh_paths, "similarity": -1, "score": baseline}
    
    def step(self, action):
        self.step_length -= 1
        terminated = False
        truncated = False
        reward = 0
        score = 0
        info = {}
        sim = 0
        is_connected = False
        score, self.sh_path_list = util.cal_coverage(self.st_list, self.state, self.neighbours, self.vul_node)
        valid_action_space = util.create_action_space(self.state)
        if action < len(valid_action_space):
            real_action = valid_action_space[action]
            flag, msg = util.can_switch(self.state, real_action)
            if flag and score < self.expectation:
                info = {"msg": msg}
                if msg == "two switching":
                    self.state, is_connected = util.two_switching(self.state, real_action)
                    if is_connected: # the graph after two switching is still connected
                        score, self.sh_path_list = util.cal_coverage(self.st_list, self.state, self.neighbours, self.vul_node)
                else:
                    self.state = util.port_switching(self.state, real_action)
                    score, self.sh_path_list = util.cal_coverage(self.st_list, self.state, self.neighbours, self.vul_node)
            
            sim = sim_util.EOverlap(self.ori_adj_matrix, self.state)
        #when eavedropping
        if sim >= self.expected_sim and score >= self.expectation:
            reward = 1.0
            self.best = util.update_best(self.best, self.state, self.sh_path_list, sim, score)
            terminated = True                     # task success
        else:
            reward = -1.0

        if self.step_length <= 0 and not terminated:
            truncated = True                      # time limit hit

        obs = np.array(self.state, dtype=np.float32)
        return obs, float(reward), bool(terminated), bool(truncated), info
        
    def render(self):
        pass
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ori_adj_clone = util.clone(self.ori_adj_matrix)
        self.state = np.array(ori_adj_clone, dtype=np.float32)
        self.step_length = self.init_step_len
        info = {}
        return np.array(self.state, dtype=np.float32), info




