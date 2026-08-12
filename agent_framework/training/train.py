"""
train.py

PPO training loop for the Finance RL Environment.
"""

import time

from pathlib import Path

from config.config import config


def train_ppo(
    agent,
    env,
    run_name,
    model_directory,
    logger,
    total_episodes=None,
):

    total_episodes = (
        total_episodes if total_episodes is not None else config.training.TOTAL_EPISODES
    )

    save_every = config.training.SAVE_EVERY

    log_every = config.training.LOG_EVERY

    agent.train()

    episode_ids = []

    local_episodes = []

    update_records = []

    training_start = time.perf_counter()

    logger.info(
        "PPO training started | episodes=%s",
        total_episodes,
    )

    # ==========================================================
    # Episodes
    # ==========================================================

    for episode_index in range(
        1,
        total_episodes + 1,
    ):

        episode_seed = config.environment.RANDOM_SEED + episode_index

        state = env.reset(
            seed=episode_seed,
            options={
                "phase": "TRAIN",
                "experiment_name": run_name,
                "agent_type": "RL",
                "algorithm": "PPO",
                "guidance_mode": "NONE",
                "llm_model": None,
                "llm_plan": [],
            },
        )

        agent.start_episode()

        backend_episode_id = env.episode_id

        backend_episode_number = env.episode_number

        episode_ids.append(backend_episode_id)

        episode_reward = 0.0

        episode_steps = 0

        done = False

        last_info = {}

        # ======================================================
        # Episode Steps
        # ======================================================

        while not done:

            (
                action,
                log_prob,
                value,
            ) = agent.select_action(state)

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            environment_error = bool(
                info.get(
                    "environment_error",
                    False,
                )
            )

            # --------------------------------------------------
            # Store only valid PPO experiences
            # --------------------------------------------------

            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
                environment_error=environment_error,
            )

            # --------------------------------------------------
            # PPO update when rollout is ready
            # --------------------------------------------------

            update_metrics = agent.learn()

            if update_metrics is not None:

                update_records.append(
                    {
                        "episode": episode_index,
                        "total_steps": agent.total_steps,
                        **update_metrics,
                    }
                )

            if reward is not None:

                episode_reward += float(reward)

            episode_steps += 1

            state = next_state

            last_info = info

        # ======================================================
        # Local Training Record
        # ======================================================

        local_episodes.append(
            {
                "episode": episode_index,
                "backendEpisodeId": backend_episode_id,
                "backendEpisodeNumber": backend_episode_number,
                "reward": episode_reward,
                "steps": episode_steps,
                "completed": bool(env.state["task_completed"]),
                "terminatedReason": last_info.get("terminated_reason"),
                "environmentError": bool(
                    last_info.get(
                        "environment_error",
                        False,
                    )
                ),
            }
        )

        # ======================================================
        # Console Progress
        # ======================================================

        if (
            episode_index == 1
            or episode_index % log_every == 0
            or episode_index == total_episodes
        ):

            recent = local_episodes[
                -min(
                    log_every,
                    len(local_episodes),
                ) :
            ]

            recent_reward = sum(row["reward"] for row in recent) / len(recent)

            recent_success = sum(1 for row in recent if row["completed"]) / len(recent)

            logger.info(
                (
                    "Episode %s/%s | "
                    "reward=%.2f | "
                    "steps=%s | "
                    "completed=%s | "
                    "recent_avg_reward=%.2f | "
                    "recent_success=%.2f%% | "
                    "buffer=%s"
                ),
                episode_index,
                total_episodes,
                episode_reward,
                episode_steps,
                env.state["task_completed"],
                recent_reward,
                recent_success * 100.0,
                len(agent.buffer),
            )

        # ======================================================
        # Checkpoint
        # ======================================================

        if save_every > 0 and episode_index % save_every == 0:

            checkpoint = Path(model_directory) / (f"ppo_episode_" f"{episode_index}.pt")

            agent.save(checkpoint)

            logger.info(
                "Checkpoint saved: %s",
                checkpoint,
            )

    # ==========================================================
    # Use Remaining Rollout
    # ==========================================================

    final_update = agent.learn(force=True)

    if final_update is not None:

        update_records.append(
            {
                "episode": total_episodes,
                "total_steps": agent.total_steps,
                **final_update,
            }
        )

    # ==========================================================
    # Final Model
    # ==========================================================

    final_checkpoint = Path(model_directory) / "ppo_final.pt"

    agent.save(final_checkpoint)

    training_time = time.perf_counter() - training_start

    logger.info(
        ("PPO training complete | " "time=%.2fs | " "steps=%s | " "updates=%s"),
        training_time,
        agent.total_steps,
        agent.update_count,
    )

    logger.info(
        "Final model: %s",
        final_checkpoint,
    )

    return {
        "run_name": run_name,
        "training_time": training_time,
        "episode_ids": episode_ids,
        "local_episodes": local_episodes,
        "ppo_updates": update_records,
        "final_checkpoint": str(final_checkpoint),
    }
