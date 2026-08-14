"""
train.py

Shared PPO training pipeline for baseline PPO and LLM + PPO.

Corrections
-----------
- environment-error transitions are never stored;
- the explicit ``trainable`` flag from FinanceEnvironment is respected;
- environment-error episodes are recorded separately from valid agent
  performance statistics;
- backend/setup failures do not terminate the entire experiment.
"""

import time
from pathlib import Path

from config.config import config

# ==========================================================
# Agent Type
# ==========================================================


def is_llm_agent(agent):
    return (
        hasattr(agent, "prepare_episode")
        and hasattr(agent, "step_environment")
        and hasattr(agent, "get_llm_metrics")
    )


# ==========================================================
# Prepare Episode
# ==========================================================


def prepare_episode(
    agent,
    env,
    episode_seed,
    run_name,
    phase,
):
    if is_llm_agent(agent):
        return agent.prepare_episode(
            env=env,
            seed=episode_seed,
            phase=phase,
            experiment_name=run_name,
            goal=config.agent.TASK,
        )

    state = env.reset(
        seed=episode_seed,
        options={
            "phase": phase,
            "experiment_name": run_name,
            "agent_type": "RL",
            "algorithm": "PPO",
            "guidance_mode": "NONE",
            "llm_model": None,
            "llm_plan": [],
        },
    )

    agent.start_episode()
    return state


# ==========================================================
# Environment Step
# ==========================================================


def execute_agent_step(agent, env, action):
    if is_llm_agent(agent):
        return agent.step_environment(
            env=env,
            action=action,
        )

    return env.step(action)


# ==========================================================
# Training
# ==========================================================


def train_agent(
    agent,
    env,
    run_name,
    model_directory,
    logger,
    total_episodes=None,
    agent_label="ppo",
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
        "%s training started | episodes=%s",
        agent_label.upper(),
        total_episodes,
    )

    for episode_index in range(1, total_episodes + 1):
        episode_seed = config.environment.RANDOM_SEED + episode_index

        # ======================================================
        # Prepare Episode
        # ======================================================

        try:
            state = prepare_episode(
                agent=agent,
                env=env,
                episode_seed=episode_seed,
                run_name=run_name,
                phase="TRAIN",
            )
        except Exception as exc:
            logger.error(
                "Episode %s setup failed: %s",
                episode_index,
                exc,
            )

            local_episodes.append(
                {
                    "episode": episode_index,
                    "backendEpisodeId": getattr(
                        env,
                        "episode_id",
                        None,
                    ),
                    "backendEpisodeNumber": getattr(
                        env,
                        "episode_number",
                        None,
                    ),
                    "reward": 0.0,
                    "baseReward": 0.0,
                    "guidanceBonus": 0.0,
                    "completionBonus": 0.0,
                    "steps": 0,
                    "completed": False,
                    "terminatedReason": "SETUP_ERROR",
                    "environmentError": True,
                    "validForMetrics": False,
                    "setupError": str(exc),
                    "procedureAttempts": 0,
                    "procedureFollowed": 0,
                    "procedureAdherence": None,
                }
            )
            continue

        backend_episode_id = env.episode_id
        backend_episode_number = env.episode_number
        episode_ids.append(backend_episode_id)

        episode_reward = 0.0
        episode_base_reward = 0.0
        episode_guidance_bonus = 0.0
        episode_completion_bonus = 0.0
        episode_steps = 0

        procedure_attempts = 0
        procedure_followed_count = 0

        done = False
        last_info = {}
        episode_environment_error = False

        # ======================================================
        # Episode Steps
        # ======================================================

        while not done:
            action, log_prob, value = agent.select_action(state)

            next_state, reward, done, info = execute_agent_step(
                agent=agent,
                env=env,
                action=action,
            )

            environment_error = bool(info.get("environment_error", False))
            trainable = bool(
                info.get(
                    "trainable",
                    not environment_error,
                )
            )

            if environment_error:
                episode_environment_error = True

            agent.store_transition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                log_prob=log_prob,
                value=value,
                environment_error=environment_error,
                trainable=trainable,
            )

            update_metrics = agent.learn()

            if update_metrics is not None:
                update_records.append(
                    {
                        "episode": episode_index,
                        "total_steps": agent.total_steps,
                        **update_metrics,
                    }
                )

            # Valid rewards are accumulated for logging. An
            # infrastructure transition should already be reward=0.
            episode_reward += float(reward or 0.0)
            episode_base_reward += float(info.get("base_reward", 0.0) or 0.0)
            episode_guidance_bonus += float(info.get("guidance_bonus", 0.0) or 0.0)
            episode_completion_bonus += float(info.get("completion_bonus", 0.0) or 0.0)

            procedure_followed = info.get("procedure_followed")

            if procedure_followed is not None:
                procedure_attempts += 1
                if procedure_followed:
                    procedure_followed_count += 1

            episode_steps += 1
            state = next_state
            last_info = info

        procedure_adherence = None
        if procedure_attempts > 0:
            procedure_adherence = procedure_followed_count / procedure_attempts

        episode_record = {
            "episode": episode_index,
            "backendEpisodeId": backend_episode_id,
            "backendEpisodeNumber": backend_episode_number,
            "reward": episode_reward,
            "baseReward": episode_base_reward,
            "guidanceBonus": episode_guidance_bonus,
            "completionBonus": episode_completion_bonus,
            "steps": episode_steps,
            "completed": bool(env.state["task_completed"]),
            "terminatedReason": last_info.get("terminated_reason"),
            "environmentError": episode_environment_error,
            "validForMetrics": not episode_environment_error,
            "procedureAttempts": procedure_attempts,
            "procedureFollowed": procedure_followed_count,
            "procedureAdherence": procedure_adherence,
        }

        if is_llm_agent(agent):
            episode_record.update(
                {
                    "llmPlan": agent.get_current_plan(),
                    "llmPrerequisites": (agent.get_current_prerequisites()),
                    "llmPlanCached": agent.current_plan_cached,
                    "llmPlanningTimeMs": (agent.current_plan_latency_ms),
                }
            )

        local_episodes.append(episode_record)

        # ======================================================
        # Console Progress
        # ======================================================

        if (
            episode_index == 1
            or (log_every > 0 and episode_index % log_every == 0)
            or episode_index == total_episodes
        ):
            window_size = min(
                log_every if log_every > 0 else 1,
                len(local_episodes),
            )
            recent = local_episodes[-window_size:]
            valid_recent = [row for row in recent if row.get("validForMetrics", True)]

            if valid_recent:
                recent_reward = sum(row["reward"] for row in valid_recent) / len(
                    valid_recent
                )

                recent_success = sum(
                    1 for row in valid_recent if row["completed"]
                ) / len(valid_recent)
            else:
                recent_reward = 0.0
                recent_success = 0.0

            recent_env_errors = sum(
                1 for row in recent if row.get("environmentError", False)
            )

            logger.info(
                (
                    "Episode %s/%s | reward=%.2f | steps=%s | "
                    "completed=%s | recent_reward=%.2f | "
                    "recent_success=%.2f%% | env_errors=%s | "
                    "buffer=%s"
                ),
                episode_index,
                total_episodes,
                episode_reward,
                episode_steps,
                env.state["task_completed"],
                recent_reward,
                recent_success * 100.0,
                recent_env_errors,
                len(agent.buffer),
            )

            if is_llm_agent(agent):
                logger.info(
                    (
                        "LLM | cached=%s | plan=%s | "
                        "guidance_bonus=%.2f | "
                        "procedure_adherence=%s"
                    ),
                    agent.current_plan_cached,
                    agent.current_plan,
                    episode_guidance_bonus,
                    (
                        f"{procedure_adherence:.2%}"
                        if procedure_adherence is not None
                        else "N/A"
                    ),
                )

        # ======================================================
        # Checkpoint
        # ======================================================

        if save_every > 0 and episode_index % save_every == 0:
            checkpoint = Path(model_directory) / (
                f"{agent_label}_episode_{episode_index}.pt"
            )
            agent.save(checkpoint)
            logger.info("Checkpoint saved: %s", checkpoint)

    # ==========================================================
    # Final PPO Update / Save
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

    final_checkpoint = Path(model_directory) / f"{agent_label}_final.pt"
    agent.save(final_checkpoint)

    training_time = time.perf_counter() - training_start

    llm_metrics = None
    if is_llm_agent(agent):
        llm_metrics = agent.get_llm_metrics()

    valid_episodes = [row for row in local_episodes if row.get("validForMetrics", True)]
    environment_error_episodes = sum(
        1 for row in local_episodes if row.get("environmentError", False)
    )

    logger.info(
        (
            "%s training complete | time=%.2fs | steps=%s | "
            "updates=%s | valid_episodes=%s | env_errors=%s"
        ),
        agent_label.upper(),
        training_time,
        agent.total_steps,
        agent.update_count,
        len(valid_episodes),
        environment_error_episodes,
    )

    return {
        "run_name": run_name,
        "agent_label": agent_label,
        "training_time": training_time,
        "episode_ids": episode_ids,
        "local_episodes": local_episodes,
        "valid_episode_count": len(valid_episodes),
        "environment_error_episodes": environment_error_episodes,
        "ppo_updates": update_records,
        "llm_metrics": llm_metrics,
        "final_checkpoint": str(final_checkpoint),
    }


# ==========================================================
# Backwards-Compatible PPO Name
# ==========================================================


def train_ppo(
    agent,
    env,
    run_name,
    model_directory,
    logger,
    total_episodes=None,
):
    return train_agent(
        agent=agent,
        env=env,
        run_name=run_name,
        model_directory=model_directory,
        logger=logger,
        total_episodes=total_episodes,
        agent_label="ppo",
    )
