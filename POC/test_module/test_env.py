from environment import GridWorldEnv
import numpy as np

def test_environment():
    env = GridWorldEnv()

    # Reset the environment to get the starting state
    state, info = env.reset()
    print(f"Starting Position: {state}")

    # Run a loop of 10 random actions
    for i in range(10):
        # Sample a random action (0, 1, 2, or 3)
        action = env.action_space.sample()

        # Execute the action
        next_state, reward, done, truncated, info = env.step(action)

        print(f"Step {i+1}: Action={action} | New Pos={next_state} | Reward={reward}")

        if done:
            print("Goal Reached!")
            break
    
    print("Test Complete.")
