"""
train.py

Shared PPO training pipeline.

Supports:

    - PPO baseline
    - LLM + PPO

The difference between the agents is isolated to episode
preparation and environment stepping.

PPO:
    env.reset()
    env.step()

LLM_RL:
    agent.prepare_episode()
    agent.step_environment()

All learning remains PPO.
"""

import time

from pathlib import Path

from config.config import config

# ==========================================================
# Agent Type
# ==========================================================


def is_llm_agent(agent):
    """
    Detect whether the agent exposes the LLM-RL interface.
    """

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
    """
    Prepare one episode for either PPO or LLM + PPO.
    """

    # ======================================================
    # LLM + PPO
    # ======================================================

    if is_llm_agent(agent):

        return agent.prepare_episode(
            env=env,
            seed=episode_seed,
            phase=phase,
            experiment_name=run_name,
            goal=config.agent.TASK,
        )

    # ======================================================
    # PPO Baseline
    # ======================================================

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


def execute_agent_step(
    agent,
    env,
    action,
):
    """
    Execute one step using the correct observation interface.

    LLM input modes may require a 21-element next observation,
    while FinanceEnvironment itself returns the base
    13-element state.
    """

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
        ("%s training started | " "episodes=%s"),
        agent_label.upper(),
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

        # ======================================================
        # Prepare Episode
        # ======================================================

        state = prepare_episode(
            agent=agent,
            env=env,
            episode_seed=episode_seed,
            run_name=run_name,
            phase="TRAIN",
        )

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
            ) = execute_agent_step(
                agent=agent,
                env=env,
                action=action,
            )

            environment_error = bool(
                info.get(
                    "environment_error",
                    False,
                )
            )

            # ==================================================
            # Store Valid PPO Transition
            # ==================================================

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

            # ==================================================
            # PPO Update
            # ==================================================

            update_metrics = agent.learn()

            if update_metrics is not None:

                update_records.append(
                    {
                        "episode": episode_index,
                        "total_steps": agent.total_steps,
                        **update_metrics,
                    }
                )

            # ==================================================
            # Episode Metrics
            # ==================================================

            episode_reward += float(reward or 0.0)

            episode_base_reward += float(
                info.get(
                    "base_reward",
                    0.0,
                )
                or 0.0
            )

            episode_guidance_bonus += float(
                info.get(
                    "guidance_bonus",
                    0.0,
                )
                or 0.0
            )

            episode_completion_bonus += float(
                info.get(
                    "completion_bonus",
                    0.0,
                )
                or 0.0
            )

            procedure_followed = info.get("procedure_followed")

            if procedure_followed is not None:

                procedure_attempts += 1

                if procedure_followed:

                    procedure_followed_count += 1

            episode_steps += 1

            state = next_state

            last_info = info

        # ======================================================
        # Procedure Adherence
        # ======================================================

        procedure_adherence = None

        if procedure_attempts > 0:

            procedure_adherence = procedure_followed_count / procedure_attempts

        # ======================================================
        # Local Episode Record
        # ======================================================

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
            "environmentError": bool(
                last_info.get(
                    "environment_error",
                    False,
                )
            ),
            "procedureAttempts": procedure_attempts,
            "procedureFollowed": procedure_followed_count,
            "procedureAdherence": procedure_adherence,
        }

        if is_llm_agent(agent):

            episode_record.update(
                {
                    "llmPlan": agent.get_current_plan(),
                    "llmPlanCached": agent.current_plan_cached,
                    "llmPlanningTimeMs": agent.current_plan_latency_ms,
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

            recent = local_episodes[
                -min(
                    log_every if log_every > 0 else 1,
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
                    "recent_reward=%.2f | "
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

            if is_llm_agent(agent):

                logger.info(
                    (
                        "LLM | cached=%s | "
                        "plan=%s | "
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
                f"{agent_label}_episode_" f"{episode_index}.pt"
            )

            agent.save(checkpoint)

            logger.info(
                "Checkpoint saved: %s",
                checkpoint,
            )

    # ==========================================================
    # Final Incomplete PPO Rollout
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

    final_checkpoint = Path(model_directory) / f"{agent_label}_final.pt"

    agent.save(final_checkpoint)

    training_time = time.perf_counter() - training_start

    # ==========================================================
    # LLM Metrics
    # ==========================================================

    llm_metrics = None

    if is_llm_agent(agent):

        llm_metrics = agent.get_llm_metrics()

    logger.info(
        ("%s training complete | " "time=%.2fs | " "steps=%s | " "updates=%s"),
        agent_label.upper(),
        training_time,
        agent.total_steps,
        agent.update_count,
    )

    return {
        "run_name": run_name,
        "agent_label": agent_label,
        "training_time": training_time,
        "episode_ids": episode_ids,
        "local_episodes": local_episodes,
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
