from agent import train_agent
import time

if __name__ == "__main__":
    # Train Agent
    start_time = time.perf_counter()
    train_agent()
    end_time = time.perf_counter()

    # Calculate the Duratio of the Agent Training
    duration = end_time - start_time
    hours, reminder = divmod(duration, 3600)
    minutes, seconds = divmod(reminder, 60)

    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.6f}s")