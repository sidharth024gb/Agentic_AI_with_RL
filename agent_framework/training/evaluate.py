"""
evaluate.py

Shared deterministic evaluation for PPO and LLM + PPO.

Corrections
-----------
- deterministic policy behaviour is preserved;
- environment-error episodes are excluded from agent success-rate
  calculations rather than counted as agent failures;
- setup/backend failures are recorded without aborting evaluation;
- valid episode count and environment-error count are returned.
"""

import time

from config.config import config
from training.train import (
    execute_agent_step,
    is_llm_agent,
    prepare_episode,
)

# ==========================================================
# Numeric LLM Metric Difference
# ==========================================================


def _llm_metric_delta(before, after):
    if before is None or after is None:
        return None

    result = {}

    for key, value in after.items():
        before_value = before.get(key)

        if (
            isinstance(value, (int, float))
            and isinstance(before_value, (int, float))
            and not isinstance(value, bool)
        ):
            result[key] = value - before_value
        else:
            result[key] = value

    return result


# ==========================================================
# Evaluation
# ==========================================================


def evaluate_agent(
    agent,
    env,
    run_name,
    logger,
    evaluation_episodes=None,
    agent_label="ppo",
):
    evaluation_episodes = (
        evaluation_episodes
        if evaluation_episodes is not None
        else config.training.EVALUATION_EPISODES
    )

    was_training = agent.training

    llm_before = None
    if is_llm_agent(agent):
        llm_before = agent.get_llm_metrics()

    agent.eval()

    results = []
    episode_ids = []
    start_time = time.perf_counter()

    logger.info(
        "%s deterministic evaluation started | episodes=%s",
        agent_label.upper(),
        evaluation_episodes,
    )

    try:
        for episode_index in range(1, evaluation_episodes + 1):
            episode_seed = config.environment.RANDOM_SEED + 100_000 + episode_index

            # ==================================================
            # Episode Setup
            # ==================================================

            try:
                state = prepare_episode(
                    agent=agent,
                    env=env,
                    episode_seed=episode_seed,
                    run_name=run_name,
                    phase="EVALUATION",
                )
            except Exception as exc:
                logger.error(
                    "Evaluation episode %s setup failed: %s",
                    episode_index,
                    exc,
                )

                results.append(
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
                        "guidanceBonus": 0.0,
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

            episode_ids.append(env.episode_id)

            done = False
            episode_reward = 0.0
            episode_guidance_bonus = 0.0
            episode_steps = 0
            procedure_attempts = 0
            procedure_followed_count = 0
            final_info = {}
            episode_environment_error = False

            # ==================================================
            # Deterministic Episode
            # ==================================================

            while not done:
                action, _, _ = agent.select_action(state)

                next_state, reward, done, info = execute_agent_step(
                    agent=agent,
                    env=env,
                    action=action,
                )

                environment_error = bool(info.get("environment_error", False))

                if environment_error:
                    episode_environment_error = True

                episode_reward += float(reward or 0.0)
                episode_guidance_bonus += float(info.get("guidance_bonus", 0.0) or 0.0)

                procedure_followed = info.get("procedure_followed")

                if procedure_followed is not None:
                    procedure_attempts += 1
                    if procedure_followed:
                        procedure_followed_count += 1

                episode_steps += 1
                state = next_state
                final_info = info

            adherence = None
            if procedure_attempts:
                adherence = procedure_followed_count / procedure_attempts

            episode_record = {
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
                "reward": episode_reward,
                "guidanceBonus": episode_guidance_bonus,
                "steps": episode_steps,
                "completed": bool(env.state["task_completed"]),
                "terminatedReason": final_info.get("terminated_reason"),
                "environmentError": episode_environment_error,
                "validForMetrics": (not episode_environment_error),
                "procedureAttempts": procedure_attempts,
                "procedureFollowed": procedure_followed_count,
                "procedureAdherence": adherence,
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

            results.append(episode_record)
    finally:
        if was_training:
            agent.train()

    evaluation_time = time.perf_counter() - start_time

    valid_results = [row for row in results if row.get("validForMetrics", True)]

    environment_error_count = sum(
        1 for row in results if row.get("environmentError", False)
    )

    success_count = sum(1 for row in valid_results if row["completed"])

    valid_count = len(valid_results)
    success_rate = success_count / valid_count if valid_count else 0.0

    logger.info(
        (
            "%s evaluation complete | success=%s/%s valid | "
            "success_rate=%.2f%% | environment_errors=%s | "
            "time=%.2fs"
        ),
        agent_label.upper(),
        success_count,
        valid_count,
        success_rate * 100.0,
        environment_error_count,
        evaluation_time,
    )

    evaluation_llm_metrics = None

    if is_llm_agent(agent):
        llm_after = agent.get_llm_metrics()
        evaluation_llm_metrics = _llm_metric_delta(
            llm_before,
            llm_after,
        )

    return {
        "evaluation_time": evaluation_time,
        "episode_ids": episode_ids,
        "local_episodes": results,
        "valid_episode_count": valid_count,
        "environment_error_episodes": environment_error_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "llm_metrics": evaluation_llm_metrics,
    }


# ==========================================================
# Backwards-Compatible PPO Name
# ==========================================================


def evaluate_ppo(
    agent,
    env,
    run_name,
    logger,
    evaluation_episodes=None,
):
    return evaluate_agent(
        agent=agent,
        env=env,
        run_name=run_name,
        logger=logger,
        evaluation_episodes=evaluation_episodes,
        agent_label="ppo",
    )
