"""
evaluate.py

Deterministic PPO evaluation.
"""

import time

from config.config import config


def evaluate_ppo(
    agent,
    env,
    run_name,
    logger,
    evaluation_episodes=None,
):

    evaluation_episodes = (
        evaluation_episodes
        if evaluation_episodes is not None
        else config.training.EVALUATION_EPISODES
    )

    was_training = agent.training

    agent.eval()

    results = []

    episode_ids = []

    start_time = time.perf_counter()

    logger.info(
        "PPO evaluation started | episodes=%s",
        evaluation_episodes,
    )

    for episode_index in range(
        1,
        evaluation_episodes + 1,
    ):

        episode_seed = config.environment.RANDOM_SEED + 100_000 + episode_index

        state = env.reset(
            seed=episode_seed,
            options={
                "phase": "EVALUATION",
                "experiment_name": run_name,
                "agent_type": "RL",
                "algorithm": "PPO",
                "guidance_mode": "NONE",
                "llm_model": None,
                "llm_plan": [],
            },
        )

        episode_ids.append(env.episode_id)

        done = False

        episode_reward = 0.0

        episode_steps = 0

        final_info = {}

        while not done:

            (
                action,
                _,
                _,
            ) = agent.select_action(state)

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            if reward is not None:

                episode_reward += float(reward)

            episode_steps += 1

            state = next_state

            final_info = info

        results.append(
            {
                "episode": episode_index,
                "reward": episode_reward,
                "steps": episode_steps,
                "completed": bool(env.state["task_completed"]),
                "terminatedReason": final_info.get("terminated_reason"),
                "environmentError": bool(
                    final_info.get(
                        "environment_error",
                        False,
                    )
                ),
            }
        )

    evaluation_time = time.perf_counter() - start_time

    success_count = sum(1 for row in results if row["completed"])

    logger.info(
        (
            "PPO evaluation complete | "
            "success=%s/%s | "
            "success_rate=%.2f%% | "
            "time=%.2fs"
        ),
        success_count,
        evaluation_episodes,
        (success_count / evaluation_episodes * 100.0),
        evaluation_time,
    )

    if was_training:

        agent.train()

    return {
        "evaluation_time": evaluation_time,
        "episode_ids": episode_ids,
        "local_episodes": results,
    }
