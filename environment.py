import gymnasium as gym
from gymnasium import spaces
import numpy as np

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()

        self.grid_size = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)

    def reset(self, seed=None):
        self.agent_pos = np.array([0, 0])
        return self.agent_pos, {}
    
    def step(self, action):
        # 0: DOWN, 1: UP, 2: LEFT, 3: RIGHT
        if action == 0: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        else: raise f"Invalid Action: {action}"

        # Define the Goal
        goal = np.array([4, 4])
        done = np.array_equal(self.agent_pos, goal)
        reward = 1 if done else -0.1 # Small penalty for each step to encourage speed ( Reward to find the shortest path )

        return self.agent_pos, reward, done, False, {}