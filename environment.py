import gymnasium as gym
from gymnasium import spaces
import numpy as np
from planner import get_llm_hint

class GridWorldEnv(gym.Env):
    def __init__(self):
        super(GridWorldEnv, self).__init__()

        self.grid_size = 5
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=self.grid_size - 1, shape=(2,), dtype=np.int32)
        self.agent_pos = np.array([0, 0])

    def reset(self, seed=None):
        self.agent_pos = np.array([0, 0])
        return self.agent_pos, {}
    
    def step(self, action):
        # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT
        if action == 0: self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == 1: self.agent_pos[0] = min(self.grid_size - 1, self.agent_pos[0] + 1)
        elif action == 2: self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == 3: self.agent_pos[1] = min(self.grid_size - 1, self.agent_pos[1] + 1)
        else: raise f"Invalid Action: {action}"

        # Get LLM Hint (The "Grounding" Mechanism)
        hint_action = get_llm_hint(self.agent_pos)

        # Define the Goal
        goal = np.array([4, 4])
        done = np.array_equal(self.agent_pos, goal)
        reward = 10 if done else -0.1 # Small penalty for each step to encourage speed ( Reward to find the shortest path )

        # Apply Bonus: If agent matches LLM hint, add +0.5 reward
        if hint_action == action:
            reward += 0.5

        return self.agent_pos, reward, done, False, {}