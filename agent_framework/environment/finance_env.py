"""
finance_env.py

Finance reinforcement-learning environment.

Supports:
    - PPO baseline
    - LLM + PPO input guidance
    - LLM + PPO reward shaping
    - LLM + PPO input + reward guidance

Key corrections
---------------
- validation findings are treated as successful processing;
- partial batch success is preserved instead of becoming a full loss;
- a first valid no-op is neutral; consecutive no-progress repeats get a tiny penalty;
- LLM-recommended repeated no-ops are exempt from that artificial repeat penalty;
- duplicate/supplier/budget exclusions persist across refreshes;
- task completion means no valid/payable outstanding invoice remains;
- budget allocation is sequential and can skip an invoice then include
  a later smaller invoice;
- environment errors never create trainable transitions.
"""

from datetime import datetime, timedelta, UTC

import random
import time

import numpy as np
import pandas as pd

from config.config import config
from environment.action_space import ActionSpace, FinanceAction
from environment.state_encoder import StateEncoder
from environment.reward_processor import RewardProcessor
from environment.procedure_tracker import ProcedureTracker


class FinanceEnvironment:
    def __init__(self, api_client, llm_plan=None, llm_prerequisites=None):
        self.api_client = api_client

        self.max_steps = config.environment.MAX_STEPS_PER_EPISODE
        self.seed = config.environment.RANDOM_SEED
        self.observation_type = config.environment.OBSERVATION_TYPE

        self.agent_type = config.agent.AGENT_TYPE
        self.algorithm = config.agent.ALGORITHM
        self.goal = config.agent.TASK

        self.phase = config.experiment.PHASE
        self.experiment_name = config.experiment.EXPERIMENT_NAME
        self.guidance_mode = config.experiment.GUIDANCE_MODE
        self.guidance_bonus = config.experiment.GUIDANCE_BONUS
        self.llm_model = config.llm.MODEL if self.agent_type == "LLM_RL" else None

        self.prompt_version = None
        self.llm_plan_cached = False
        self.llm_planning_time_ms = 0.0

        self.action_space_handler = ActionSpace()
        self.state_encoder = StateEncoder()

        self.reward_processor = RewardProcessor(
            use_guidance=self._uses_reward_guidance(),
            guidance_bonus=self.guidance_bonus,
        )

        self.llm_plan = []
        self.llm_procedure = []
        self.llm_prerequisites = {}

        self.procedure_tracker = ProcedureTracker(
            procedure=[],
            action_dim=self.action_space_handler.action_count,
        )

        self.set_llm_plan(
            llm_plan or [],
            prerequisites=llm_prerequisites,
        )

        self.episode_id = None
        self.episode_number = None
        self.episode_active = False
        self.current_step = 0

        # Tracks only the immediately previous action in the episode.
        # This lets us discourage consecutive no-progress loops without
        # penalising a legitimate action that becomes useful again later.
        self.last_action_name = None

        self.all_invoices = pd.DataFrame()
        self.paid_invoices = pd.DataFrame()
        self.rejected_invoices = pd.DataFrame()
        self.pending_approval_invoices = pd.DataFrame()
        self.approved_invoices = pd.DataFrame()
        self.report_df = pd.DataFrame()

        # Episode-level exclusions. These are business findings, not
        # failed actions.
        self.duplicate_invoice_ids = set()
        self.invalid_supplier_invoice_ids = set()
        self.budget_excluded_invoice_ids = set()

        self.state = self._initial_state()

    # ==========================================================
    # Public dimensions
    # ==========================================================

    @property
    def observation_size(self):
        return self.state_encoder.get_state_size()

    @property
    def action_size(self):
        return self.action_space_handler.action_count

    # ==========================================================
    # Guidance modes / plan
    # ==========================================================

    def _uses_reward_guidance(self):
        return self.agent_type == "LLM_RL" and self.guidance_mode in {
            "REWARD_SHAPING",
            "INPUT_AND_REWARD",
        }

    def _uses_input_guidance(self):
        return self.agent_type == "LLM_RL" and self.guidance_mode in {
            "INPUT",
            "INPUT_AND_REWARD",
        }

    def set_llm_plan(
        self,
        plan,
        prerequisites=None,
    ):
        """
        Install the LLM procedure and its prerequisite graph.

        ``ProcedureTracker`` owns procedural completion state.
        Environment action flags are not used as proof that a
        procedure step was completed correctly.

        ``prerequisites`` can contain integer keys or JSON string keys.
        When omitted, ProcedureTracker derives the legacy cumulative
        prerequisite graph from the plan order.
        """

        names = []
        action_ids = []

        for item in plan or []:
            if isinstance(item, str):
                name = item.strip().upper().replace("-", "_").replace(" ", "_")

                try:
                    action = FinanceAction[name]
                except KeyError as exc:
                    raise ValueError(f"Unknown LLM action: {item}") from exc
            else:
                try:
                    action = FinanceAction(int(item))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid LLM action: {item}") from exc

            names.append(action.name)
            action_ids.append(int(action.value))

        self.procedure_tracker.set_procedure(
            action_ids,
            prerequisites=prerequisites,
        )

        self.llm_plan = names
        self.llm_procedure = action_ids
        self.llm_prerequisites = self.procedure_tracker.get_prerequisites()

    # ==========================================================
    # State / observation
    # ==========================================================

    def _initial_state(self):
        return {
            "get_invoices": False,
            "check_duplicate": False,
            "check_supplier": False,
            "approve_invoices": False,
            "check_budget": False,
            "pay_invoices": False,
            "generate_report": False,
            "check_payment_completed": False,
            "has_paid_invoices": False,
            "has_rejected_invoices": False,
            "has_pending_approval_invoices": False,
            "has_approved_invoices": False,
            "has_excluded_invoices": False,
            "has_duplicate_invoices": False,
            "has_invalid_supplier_invoices": False,
            "has_budget_excluded_invoices": False,
            "has_payable_invoices": False,
            "task_completed": False,
        }

    def _update_invoice_states(self):
        self.state["has_paid_invoices"] = not self.paid_invoices.empty
        self.state["has_rejected_invoices"] = not self.rejected_invoices.empty
        self.state["has_pending_approval_invoices"] = (
            not self.pending_approval_invoices.empty
        )
        self.state["has_approved_invoices"] = not self.approved_invoices.empty

        self.state["has_duplicate_invoices"] = bool(self.duplicate_invoice_ids)
        self.state["has_invalid_supplier_invoices"] = bool(
            self.invalid_supplier_invoice_ids
        )
        self.state["has_budget_excluded_invoices"] = bool(
            self.budget_excluded_invoice_ids
        )
        self.state["has_excluded_invoices"] = bool(self._all_excluded_ids())

        self.state["has_payable_invoices"] = bool(
            not self.pending_approval_invoices.empty or not self.approved_invoices.empty
        )

    def get_state(self):
        self._update_invoice_states()
        return self.state.copy()

    def get_observation(self):
        return self.state_encoder.encode(self.get_state())

    def get_guidance_vector(self):
        return np.asarray(
            self.procedure_tracker.get_guidance(),
            dtype=np.float32,
        )

    def get_guided_observation(self):
        base = self.get_observation()
        guidance = self.get_guidance_vector()

        return np.concatenate([base, guidance]).astype(np.float32)

    # ==========================================================
    # Result helpers
    # ==========================================================

    @staticmethod
    def _action_result(
        success,
        useful_action,
        message,
        error_type=None,
        environment_error=False,
        **extra,
    ):
        result = {
            "success": bool(success),
            "useful_action": bool(useful_action),
            "procedure_followed": None,
            "error_type": error_type,
            "environment_error": bool(environment_error),
            "message": message,
        }
        result.update(extra)
        return result

    def _environment_error_result(
        self,
        response,
        fallback_message,
    ):
        return self._action_result(
            success=False,
            useful_action=False,
            environment_error=True,
            error_type=None,
            message=self._response_message(
                response,
                fallback_message,
            ),
        )

    @staticmethod
    def _get_action_endpoint(action_name):
        endpoints = {
            "GET_INVOICES": "GET /invoice",
            "CHECK_DUPLICATE": "POST /invoice/duplicate-check",
            "CHECK_SUPPLIER": "POST /supplier/validate",
            "APPROVE_INVOICES": "PATCH /approval/approve",
            "PAY_INVOICES": "POST /payment/pay",
            "CHECK_BUDGET": "POST /account/budget/check",
            "GENERATE_REPORT": "POST /report/generate-report",
            "CHECK_PAYMENT_COMPLETED": (
                "GET /invoice + duplicate/supplier/budget validation"
            ),
        }
        return endpoints.get(action_name)

    # ==========================================================
    # Episode lifecycle
    # ==========================================================

    def _start_episode(self):
        payload = {
            "agentType": self.agent_type,
            "algorithm": self.algorithm,
            "goal": self.goal,
            "phase": self.phase,
            "experimentName": self.experiment_name,
            "seed": self.seed,
            "llmModel": self.llm_model,
            "promptVersion": self.prompt_version,
            "llmPlanCached": self.llm_plan_cached,
            "llmPlanningTimeMs": self.llm_planning_time_ms,
            "guidanceMode": self.guidance_mode,
            "llmPlan": self.llm_plan,
            "initialState": self.get_state().copy(),
        }

        response = self.api_client.start_episode(payload)

        if self._is_environment_error(response):
            raise RuntimeError("Environment error while starting episode.")

        if not self._response_succeeded(response):
            raise RuntimeError("Failed to start episode.")

        data = self._response_data(response)
        self.episode_id = data.get("episodeId")
        self.episode_number = data.get("episodeNumber")

        if not self.episode_id:
            raise RuntimeError("Episode started but no episodeId was returned.")

        self.episode_active = True

    def _record_episode_step(
        self,
        action_name,
        endpoint,
        base_reward,
        guidance_bonus,
        reward,
        success,
        useful_action,
        environment_error,
        procedure_followed,
        message,
        state_before,
        state_after,
        duration_ms,
    ):
        if not self.episode_active or not self.episode_id:
            return None

        payload = {
            "action": action_name,
            "endpoint": endpoint,
            "baseReward": float(base_reward),
            "guidanceBonus": float(guidance_bonus),
            "reward": float(reward),
            "success": bool(success),
            "usefulAction": bool(useful_action),
            "environmentError": bool(environment_error),
            "procedureFollowed": procedure_followed,
            "message": message,
            "stateBefore": state_before,
            "stateAfter": state_after,
            "durationMs": float(duration_ms),
        }

        return self.api_client.record_step(
            self.episode_id,
            payload,
        )

    def _end_episode(self, terminated_reason):
        if not self.episode_active or not self.episode_id:
            return None

        payload = {
            "finalState": self.get_state().copy(),
            "completed": bool(self.state["task_completed"]),
            "terminatedReason": terminated_reason,
        }

        response = self.api_client.end_episode(
            self.episode_id,
            payload,
        )

        self.episode_active = False
        return response

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self, seed=None, options=None):
        options = options or {}

        if self.episode_active:
            self._end_episode("RESET")

        if seed is not None:
            self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.agent_type = options.get(
            "agent_type",
            self.agent_type,
        )
        self.algorithm = options.get(
            "algorithm",
            self.algorithm,
        )
        self.goal = options.get("goal", self.goal)
        self.phase = options.get("phase", self.phase)
        self.experiment_name = options.get(
            "experiment_name",
            self.experiment_name,
        )
        self.guidance_mode = options.get(
            "guidance_mode",
            self.guidance_mode,
        )
        self.llm_model = options.get(
            "llm_model",
            (config.llm.MODEL if self.agent_type == "LLM_RL" else None),
        )

        self.prompt_version = options.get(
            "prompt_version",
            None,
        )
        self.llm_plan_cached = bool(options.get("llm_plan_cached", False))
        self.llm_planning_time_ms = float(
            options.get("llm_planning_time_ms", 0.0) or 0.0
        )

        self.reward_processor.configure_guidance(
            use_guidance=self._uses_reward_guidance(),
            guidance_bonus=self.guidance_bonus,
        )

        if "llm_plan" in options:
            self.set_llm_plan(
                options["llm_plan"] or [],
                prerequisites=options.get("llm_prerequisites"),
            )
        else:
            # Same plan, fresh tracker-owned completion state.
            self.procedure_tracker.reset()

        if self.agent_type != "LLM_RL":
            self.set_llm_plan(
                [],
                prerequisites={},
            )
            self.llm_model = None
            self.prompt_version = None
            self.llm_plan_cached = False
            self.llm_planning_time_ms = 0.0

        # Use the same episode-specific seed for both the Python
        # environment and the backend sandbox. This keeps each episode
        # varied while making repeated experiments reproducible.
        response = self.api_client.reset_environment(
            seed=self.seed,
        )

        if self._is_environment_error(response):
            raise RuntimeError("Backend environment reset failed.")

        if not self._response_succeeded(response):
            raise RuntimeError("Backend rejected environment reset.")

        self.current_step = 0
        self.last_action_name = None

        self.all_invoices = pd.DataFrame()
        self.paid_invoices = pd.DataFrame()
        self.rejected_invoices = pd.DataFrame()
        self.pending_approval_invoices = pd.DataFrame()
        self.approved_invoices = pd.DataFrame()
        self.report_df = pd.DataFrame()

        self.duplicate_invoice_ids.clear()
        self.invalid_supplier_invoice_ids.clear()
        self.budget_excluded_invoice_ids.clear()

        self.state = self._initial_state()

        self._start_episode()
        return self.get_observation()

    # ==========================================================
    # Step
    # ==========================================================

    def step(self, action):
        if self.current_step >= self.max_steps:
            if self.episode_active:
                self._end_episode("MAX_STEPS")

            return (
                self.get_observation(),
                0.0,
                True,
                {
                    "reason": "MAX_STEPS_REACHED",
                    "environment_error": False,
                    "trainable": True,
                    "episode_id": self.episode_id,
                    "episode_number": self.episode_number,
                },
            )

        state_before = self.get_state().copy()
        self.current_step += 1
        start_time = time.perf_counter()

        try:
            action_name = self.action_space_handler.get_action_name(action)
            endpoint = self._get_action_endpoint(action_name)
            result = self.action_space_handler.execute(self, action)
        except ValueError as exc:
            action_name = f"INVALID_ACTION_{action}"
            endpoint = None
            result = self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_ACTION",
                message=str(exc),
            )
        except Exception as exc:
            action_name = locals().get("action_name", str(action))
            endpoint = locals().get("endpoint")
            result = self._action_result(
                success=False,
                useful_action=False,
                environment_error=True,
                message=str(exc),
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        environment_error = bool(result.get("environment_error", False))
        action_success = bool(result.get("success", False))
        useful_action = bool(result.get("useful_action", False))
        error_type = result.get("error_type")

        # ======================================================
        # LLM Procedure Tracking
        #
        # ProcedureTracker owns its own completion state.
        #
        # A successful environment action becomes procedure-complete
        # only if that action's tracker prerequisites were already
        # complete. PPO is never blocked from taking another action.
        #
        # ``procedure_followed`` is stricter: it is True only when
        # PPO selected the action that was being recommended before
        # this step and the action succeeded validly.
        # ======================================================

        if self.agent_type == "LLM_RL" and self.procedure_tracker.has_procedure():
            procedure_followed = self.procedure_tracker.check_action(
                action=action,
                action_succeeded=(action_success and not environment_error),
            )
        else:
            procedure_followed = None

        result["procedure_followed"] = procedure_followed

        task_completed = bool(self.state["task_completed"])

        # ======================================================
        # Repeated No-Progress Detection
        #
        # A repeated action is still detected for diagnostics when:
        #
        #   - the action succeeded,
        #   - it was not an environment error,
        #   - it produced no useful progress, and
        #   - it is the same action selected on the previous step.
        #
        # PPO baseline / unguided repeat:
        #     repeated no-progress action -> -0.1
        #
        # LLM-guided repeat:
        #     if the repeated action is the CURRENT LLM recommendation
        #     and PPO successfully follows that recommendation, the
        #     artificial repeat penalty is suppressed.
        #
        # This avoids teaching the policy that following valid LLM
        # guidance itself produces a negative reward.
        #
        # The guided repeat does NOT receive an extra positive reward
        # merely for repeating. If it is still a no-op, its base reward
        # is neutral (0.0). RewardProcessor continues to award guidance
        # bonus only for useful progress or successful task completion.
        # ======================================================

        repeated_action = bool(
            action_success
            and not environment_error
            and not useful_action
            and self.last_action_name == action_name
        )

        guidance_enabled = bool(
            self.agent_type == "LLM_RL"
            and (
                self._uses_input_guidance()
                or self._uses_reward_guidance()
            )
        )

        guided_repeated_action = bool(
            repeated_action
            and guidance_enabled
            and procedure_followed is True
        )

        reward_result = self.reward_processor.process(
            action_name=action_name,
            action_success=action_success,
            useful_action=useful_action,
            environment_error=environment_error,
            error_type=error_type,
            task_completed=task_completed,
            procedure_followed=procedure_followed,
            repeated_action=repeated_action,
            guided_repeated_action=guided_repeated_action,
        )

        reward = reward_result["reward"]
        base_reward = reward_result["base_reward"]
        guidance_bonus = reward_result["guidance_bonus"]
        completion_bonus = reward_result["completion_bonus"]
        trainable = bool(reward_result["trainable"])

        state_after = self.get_state().copy()

        # Store the selected action for repeat detection on the next
        # step. Environment errors terminate the episode, so retaining
        # that action is unnecessary.
        if not environment_error:
            self.last_action_name = action_name

        logging_response = self._record_episode_step(
            action_name=action_name,
            endpoint=endpoint,
            base_reward=base_reward + completion_bonus,
            guidance_bonus=guidance_bonus,
            reward=reward,
            success=action_success,
            useful_action=useful_action,
            environment_error=environment_error,
            procedure_followed=procedure_followed,
            message=result.get("message"),
            state_before=state_before,
            state_after=state_after,
            duration_ms=duration_ms,
        )

        logging_error = logging_response is not None and (
            self._is_environment_error(logging_response)
            or not self._response_succeeded(logging_response)
        )

        if logging_error:
            environment_error = True
            reward = 0.0
            trainable = False

        goal_reached = bool(self.state["task_completed"])
        max_steps_reached = self.current_step >= self.max_steps

        done = goal_reached or max_steps_reached or environment_error

        terminated_reason = None
        if goal_reached:
            terminated_reason = "GOAL_REACHED"
        elif environment_error:
            terminated_reason = "ENVIRONMENT_ERROR"
        elif max_steps_reached:
            terminated_reason = "MAX_STEPS"

        if done and self.episode_active:
            self._end_episode(terminated_reason)

        info = {
            "action": action_name,
            "step": self.current_step,
            "success": action_success,
            "useful_action": useful_action,
            "repeated_action": repeated_action,
            "guided_repeated_action": guided_repeated_action,
            "repeat_penalty_applied": bool(
                repeated_action
                and not guided_repeated_action
            ),
            "error_type": error_type,
            "environment_error": environment_error,
            "trainable": trainable,
            "procedure_followed": procedure_followed,
            "base_reward": base_reward,
            "guidance_bonus": guidance_bonus,
            "completion_bonus": completion_bonus,
            "reward": reward,
            "message": result.get("message"),
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "terminated_reason": terminated_reason,
            "duration_ms": duration_ms,
            "guidance_vector": self.procedure_tracker.get_guidance(),
            "procedure_prerequisites": (self.procedure_tracker.get_prerequisites()),
            "procedure_status": self.procedure_tracker.get_status(),
            "processed_count": result.get("processed_count", 0),
            "skipped_count": result.get("skipped_count", 0),
            "excluded_count": len(self._all_excluded_ids()),
        }

        return (
            self.get_observation(),
            float(reward),
            done,
            info,
        )

    # ==========================================================
    # Action 0 - GET_INVOICES
    # ==========================================================

    def get_invoices(self):
        response = self.api_client.get_invoices()

        if self._is_environment_error(response):
            return self._environment_error_result(
                response,
                "Environment error while loading invoices.",
            )

        if not self._response_succeeded(response):
            return self._action_result(
                success=False,
                useful_action=False,
                error_type=self._response_error_type(
                    response,
                    "INVALID_ACTION",
                ),
                message=self._response_message(
                    response,
                    "Failed to load invoices.",
                ),
            )

        data = self._response_data(response)
        invoices = (
            data.get("invoices", data.get("data", []))
            if isinstance(data, dict)
            else data
        )

        was_loaded = self.state["get_invoices"]
        self.all_invoices = pd.DataFrame(invoices)

        split_result = self._split_invoice_frames(
            self.all_invoices,
            apply_exclusions=True,
        )

        if split_result is not None:
            return split_result

        self.state["get_invoices"] = True
        self._update_invoice_states()

        return self._action_result(
            success=True,
            useful_action=(not was_loaded),
            message="Invoices loaded.",
            processed_count=len(self.all_invoices),
        )

    # ==========================================================
    # Action 1 - CHECK_DUPLICATE
    # ==========================================================

    def check_duplicate(self):
        if not self.state["get_invoices"]:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded before " "duplicate checking."),
            )

        if self.state["check_duplicate"]:
            return self._action_result(
                success=True,
                useful_action=False,
                message="Duplicate checking already completed.",
            )

        candidates = self._candidate_invoices()

        if candidates.empty:
            self.state["check_duplicate"] = True
            return self._action_result(
                success=True,
                useful_action=False,
                message="No invoices available for duplicate checking.",
            )

        processed_count = 0
        duplicate_ids = set()

        for _, invoice in candidates.iterrows():
            invoice_id = self._invoice_id(invoice)
            supplier_id = self._get_value(
                invoice,
                ["supplierId", "supplier", "supplier_id"],
            )
            amount = self._get_value(
                invoice,
                ["amount", "totalAmount"],
            )
            due_date = self._get_value(
                invoice,
                ["dueDate", "due_date"],
            )

            if (
                invoice_id is None
                or supplier_id is None
                or amount is None
                or due_date is None
            ):
                return self._action_result(
                    success=False,
                    useful_action=False,
                    error_type="MISSING_REQUIRED_FIELD",
                    message=(
                        "Invoice is missing data required " "for duplicate checking."
                    ),
                )

            response = self.api_client.check_invoice_duplicate(
                invoice_id,
                supplier_id,
                amount,
                due_date,
            )

            if self._is_environment_error(response):
                return self._environment_error_result(
                    response,
                    "Environment error during duplicate checking.",
                )

            if not self._response_succeeded(response):
                return self._action_result(
                    success=False,
                    useful_action=False,
                    error_type=self._response_error_type(
                        response,
                        "INVALID_ACTION",
                    ),
                    message=self._response_message(
                        response,
                        "Duplicate check failed.",
                    ),
                )

            data = self._response_data(response)
            duplicate = bool(
                data.get("duplicate", False) if isinstance(data, dict) else False
            )

            processed_count += 1

            if duplicate:
                duplicate_ids.add(str(invoice_id))

        self.duplicate_invoice_ids.update(duplicate_ids)
        self._apply_known_exclusions_to_local_frames()

        self.state["check_duplicate"] = True
        self._update_invoice_states()

        return self._action_result(
            success=True,
            useful_action=(processed_count > 0),
            message="Duplicate invoices checked.",
            processed_count=processed_count,
            skipped_count=len(duplicate_ids),
        )

    # ==========================================================
    # Action 2 - CHECK_SUPPLIER
    # ==========================================================

    def check_supplier(self):
        if not self.state["get_invoices"]:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded before " "supplier validation."),
            )

        if self.state["check_supplier"]:
            return self._action_result(
                success=True,
                useful_action=False,
                message="Supplier validation already completed.",
            )

        candidates = self._candidate_invoices()

        if candidates.empty:
            self.state["check_supplier"] = True
            return self._action_result(
                success=True,
                useful_action=False,
                message="No invoices available for supplier checking.",
            )

        processed_count = 0
        invalid_ids = set()

        for _, invoice in candidates.iterrows():
            invoice_id = self._invoice_id(invoice)
            supplier_id = self._get_value(
                invoice,
                ["supplierId", "supplier", "supplier_id"],
            )

            if invoice_id is None or supplier_id is None:
                return self._action_result(
                    success=False,
                    useful_action=False,
                    error_type="MISSING_REQUIRED_FIELD",
                    message=(
                        "Invoice is missing supplier data " "required for validation."
                    ),
                )

            response = self.api_client.validate_supplier(supplier_id)

            if self._is_environment_error(response):
                return self._environment_error_result(
                    response,
                    "Environment error during supplier validation.",
                )

            if not self._response_succeeded(response):
                return self._action_result(
                    success=False,
                    useful_action=False,
                    error_type=self._response_error_type(
                        response,
                        "INVALID_ACTION",
                    ),
                    message=self._response_message(
                        response,
                        "Supplier validation failed.",
                    ),
                )

            data = self._response_data(response)

            valid = True
            if isinstance(data, dict):
                valid = bool(
                    data.get(
                        "valid",
                        data.get("eligible", True),
                    )
                )

            processed_count += 1

            if not valid:
                invalid_ids.add(str(invoice_id))

        self.invalid_supplier_invoice_ids.update(invalid_ids)
        self._apply_known_exclusions_to_local_frames()

        self.state["check_supplier"] = True
        self._update_invoice_states()

        return self._action_result(
            success=True,
            useful_action=(processed_count > 0),
            message="Suppliers checked.",
            processed_count=processed_count,
            skipped_count=len(invalid_ids),
        )

    # ==========================================================
    # Action 3 - APPROVE_INVOICES
    # ==========================================================

    def approve_invoices(self):
        if not self.state["get_invoices"]:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message="Invoices must be loaded before approval.",
            )

        if self.pending_approval_invoices.empty:
            self.state["approve_invoices"] = True
            return self._action_result(
                success=True,
                useful_action=False,
                message="No pending invoices to approve.",
            )

        approved_indices = []
        approved_rows = []
        errors = []

        for index, invoice in self.pending_approval_invoices.iterrows():
            invoice_id = self._invoice_id(invoice)

            if invoice_id is None:
                errors.append("MISSING_REQUIRED_FIELD")
                continue

            response = self.api_client.approve_invoice(invoice_id)

            if self._is_environment_error(response):
                return self._environment_error_result(
                    response,
                    "Environment error during invoice approval.",
                )

            if self._response_succeeded(response):
                approved_indices.append(index)
                approved_row = invoice.copy()

                if "status" in approved_row.index:
                    approved_row["status"] = "APPROVED"

                approved_rows.append(approved_row)
            else:
                errors.append(
                    self._response_error_type(
                        response,
                        "INVALID_WORKFLOW",
                    )
                )

        if approved_rows:
            self.approved_invoices = pd.concat(
                [
                    self.approved_invoices,
                    pd.DataFrame(approved_rows),
                ],
                ignore_index=True,
            )

        self.pending_approval_invoices = self.pending_approval_invoices.drop(
            approved_indices,
            errors="ignore",
        )
        self.pending_approval_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        # Stage flag means the action was successfully able to make
        # progress, or there was simply nothing to process.
        if approved_indices:
            self.state["approve_invoices"] = True

        self._update_invoice_states()

        # Critical correction: successful approvals are not erased by
        # one or more other invoice failures.
        if approved_indices:
            return self._action_result(
                success=True,
                useful_action=True,
                message=(
                    f"Approved {len(approved_indices)} invoice(s); "
                    f"{len(errors)} invoice(s) were skipped."
                ),
                processed_count=len(approved_indices),
                skipped_count=len(errors),
            )

        if errors:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type=self._select_error_type(errors),
                message="No pending invoice could be approved.",
                skipped_count=len(errors),
            )

        self.state["approve_invoices"] = True
        return self._action_result(
            success=True,
            useful_action=False,
            message="No pending invoice required approval.",
        )

    # ==========================================================
    # Action 4 - PAY_INVOICES
    # ==========================================================

    def pay_invoices(self):
        if not self.state["get_invoices"]:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message="Invoices must be loaded before payment.",
            )

        if self.approved_invoices.empty:
            self.state["pay_invoices"] = True
            return self._action_result(
                success=True,
                useful_action=False,
                message="No approved invoices to pay.",
            )

        accounts_response = self.api_client.get_accounts()

        if self._is_environment_error(accounts_response):
            return self._environment_error_result(
                accounts_response,
                "Environment error while loading payment accounts.",
            )

        if not self._response_succeeded(accounts_response):
            return self._action_result(
                success=False,
                useful_action=False,
                error_type=self._response_error_type(
                    accounts_response,
                    "ACCOUNT_NOT_FOUND",
                ),
                message=self._response_message(
                    accounts_response,
                    "Unable to load payment account.",
                ),
            )

        account_data = self._response_data(accounts_response)
        accounts = (
            account_data.get("accounts", account_data.get("data", []))
            if isinstance(account_data, dict)
            else account_data
        )

        if not accounts:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="ACCOUNT_NOT_FOUND",
                message="No payment account available.",
            )

        default_account_id = self._get_value(
            pd.Series(accounts[0]),
            ["_id", "id", "accountId"],
        )

        if default_account_id is None:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="ACCOUNT_NOT_FOUND",
                message="Payment account ID unavailable.",
            )

        paid_indices = []
        paid_rows = []
        errors = []

        for index, invoice in self.approved_invoices.iterrows():
            invoice_id = self._invoice_id(invoice)

            if invoice_id is None:
                errors.append("MISSING_REQUIRED_FIELD")
                continue

            account_id = (
                self._get_value(
                    invoice,
                    ["accountId", "account", "account_id"],
                )
                or default_account_id
            )

            response = self.api_client.pay_invoice(
                invoice_id,
                account_id,
            )

            if self._is_environment_error(response):
                return self._environment_error_result(
                    response,
                    "Environment error during invoice payment.",
                )

            if self._response_succeeded(response):
                paid_indices.append(index)
                paid_row = invoice.copy()

                if "status" in paid_row.index:
                    paid_row["status"] = "PAID"

                paid_rows.append(paid_row)
            else:
                errors.append(
                    self._response_error_type(
                        response,
                        "PAYMENT_FAILED",
                    )
                )

        if paid_rows:
            self.paid_invoices = pd.concat(
                [self.paid_invoices, pd.DataFrame(paid_rows)],
                ignore_index=True,
            )

        self.approved_invoices = self.approved_invoices.drop(
            paid_indices,
            errors="ignore",
        )
        self.approved_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        if paid_indices:
            self.state["pay_invoices"] = True

        self._update_invoice_states()

        # Critical correction: partial payment success stays positive.
        if paid_indices:
            return self._action_result(
                success=True,
                useful_action=True,
                message=(
                    f"Paid {len(paid_indices)} invoice(s); "
                    f"{len(errors)} invoice(s) were skipped."
                ),
                processed_count=len(paid_indices),
                skipped_count=len(errors),
            )

        if errors:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type=self._select_error_type(errors),
                message="No approved invoice could be paid.",
                skipped_count=len(errors),
            )

        self.state["pay_invoices"] = True
        return self._action_result(
            success=True,
            useful_action=False,
            message="No approved invoice required payment.",
        )

    # ==========================================================
    # Action 5 - CHECK_BUDGET
    # ==========================================================

    def check_budget(self):
        if not self.state["get_invoices"]:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message="Invoices must be loaded before budget checking.",
            )

        if self.state["check_budget"]:
            return self._action_result(
                success=True,
                useful_action=False,
                message="Budget validation already completed.",
            )

        if self.approved_invoices.empty:
            self.state["check_budget"] = True
            return self._action_result(
                success=True,
                useful_action=False,
                message="No approved invoices to check against budget.",
            )

        allocation = self._allocate_budget(
            self.approved_invoices,
            replace_budget_exclusions=True,
        )

        if allocation["environment_error"]:
            return allocation["result"]

        if allocation["error_result"] is not None:
            return allocation["error_result"]

        self.approved_invoices = allocation["kept"].copy()
        self.approved_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        self.state["check_budget"] = True
        self._update_invoice_states()

        return self._action_result(
            success=True,
            useful_action=(allocation["processed_count"] > 0),
            message="Budget checked.",
            processed_count=allocation["processed_count"],
            skipped_count=allocation["excluded_count"],
            budget_results=allocation["budget_results"],
        )

    # ==========================================================
    # Action 6 - GENERATE_REPORT
    # ==========================================================

    def generate_report(self):
        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=365)

        response = self.api_client.generate_report(
            type="transaction_summary",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if self._is_environment_error(response):
            return self._environment_error_result(
                response,
                "Environment error while generating report.",
            )

        if not self._response_succeeded(response):
            return self._action_result(
                success=False,
                useful_action=False,
                error_type=self._response_error_type(
                    response,
                    "REPORT_FAILED",
                ),
                message=self._response_message(
                    response,
                    "Report generation failed.",
                ),
            )

        data = self._response_data(response)
        report = (
            data.get("report", data.get("data", [])) if isinstance(data, dict) else data
        )

        if isinstance(report, dict):
            report = [report]

        self.report_df = pd.DataFrame(report)
        was_generated = self.state["generate_report"]
        self.state["generate_report"] = True

        return self._action_result(
            success=True,
            useful_action=(not was_generated),
            message="One-year transaction report generated.",
            processed_count=len(self.report_df),
        )

    # ==========================================================
    # Action 7 - CHECK_PAYMENT_COMPLETED
    # ==========================================================

    def check_payment_completed(self):
        if not self.state["get_invoices"]:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded before checking " "completion."),
            )

        was_checked = self.state["check_payment_completed"]
        self.state["check_payment_completed"] = True

        response = self.api_client.get_invoices()

        if self._is_environment_error(response):
            return self._environment_error_result(
                response,
                "Environment error while checking payment completion.",
            )

        if not self._response_succeeded(response):
            return self._action_result(
                success=False,
                useful_action=False,
                error_type=self._response_error_type(
                    response,
                    "INVALID_ACTION",
                ),
                message=self._response_message(
                    response,
                    "Unable to check payment completion.",
                ),
            )

        data = self._response_data(response)
        invoices = (
            data.get("invoices", data.get("data", []))
            if isinstance(data, dict)
            else data
        )

        current_invoices = pd.DataFrame(invoices)

        evaluation = self._evaluate_outstanding_eligibility(current_invoices)

        if evaluation["environment_error"]:
            return evaluation["result"]

        if evaluation["error_result"] is not None:
            return evaluation["error_result"]

        payable = evaluation["payable"]

        # Synchronize local candidate frames to what is genuinely
        # still payable after duplicate/supplier/budget validation.
        if payable.empty:
            self.pending_approval_invoices = pd.DataFrame()
            self.approved_invoices = pd.DataFrame()
        else:
            status_column = self._find_column(
                payable,
                ["status", "invoiceStatus"],
            )

            self.pending_approval_invoices = payable[
                payable[status_column] == "PENDING_APPROVAL"
            ].copy()
            self.approved_invoices = payable[
                payable[status_column] == "APPROVED"
            ].copy()

        self.state["task_completed"] = payable.empty
        self._update_invoice_states()

        if self.state["task_completed"]:
            return self._action_result(
                success=True,
                useful_action=True,
                message=(
                    "Task completed. No valid/payable outstanding " "invoice remains."
                ),
                processed_count=evaluation["processed_count"],
                skipped_count=evaluation["excluded_count"],
            )

        return self._action_result(
            success=True,
            useful_action=(not was_checked),
            message=("Valid/payable invoices still require processing."),
            processed_count=evaluation["processed_count"],
            skipped_count=evaluation["excluded_count"],
        )

    # ==========================================================
    # Eligibility / budget helpers
    # ==========================================================

    def _evaluate_outstanding_eligibility(self, current_invoices):
        """
        Re-evaluate backend outstanding invoices to decide whether the
        task is actually complete.

        Duplicate, supplier and budget validation findings are treated
        as exclusions. They do not make the task incomplete.
        """

        if current_invoices.empty:
            return {
                "payable": pd.DataFrame(),
                "processed_count": 0,
                "excluded_count": len(self._all_excluded_ids()),
                "environment_error": False,
                "result": None,
                "error_result": None,
            }

        status_column = self._find_column(
            current_invoices,
            ["status", "invoiceStatus"],
        )

        if status_column is None:
            return {
                "payable": pd.DataFrame(),
                "processed_count": 0,
                "excluded_count": 0,
                "environment_error": False,
                "result": None,
                "error_result": self._action_result(
                    success=False,
                    useful_action=False,
                    error_type="INVALID_STATE_DATA",
                    message="Invoice status unavailable.",
                ),
            }

        outstanding = current_invoices[
            current_invoices[status_column].isin(["PENDING_APPROVAL", "APPROVED"])
        ].copy()

        if outstanding.empty:
            return {
                "payable": pd.DataFrame(),
                "processed_count": 0,
                "excluded_count": len(self._all_excluded_ids()),
                "environment_error": False,
                "result": None,
                "error_result": None,
            }

        # Start with known duplicate/supplier exclusions, then verify
        # any currently unknown outstanding invoice.
        candidate_rows = []
        processed_count = 0

        for _, invoice in outstanding.iterrows():
            invoice_id = self._invoice_id(invoice)

            if invoice_id is None:
                continue

            invoice_id_str = str(invoice_id)

            if (
                invoice_id_str in self.duplicate_invoice_ids
                or invoice_id_str in self.invalid_supplier_invoice_ids
            ):
                continue

            supplier_id = self._get_value(
                invoice,
                ["supplierId", "supplier", "supplier_id"],
            )
            amount = self._get_value(
                invoice,
                ["amount", "totalAmount"],
            )
            due_date = self._get_value(
                invoice,
                ["dueDate", "due_date"],
            )

            if supplier_id is None or amount is None or due_date is None:
                continue

            duplicate_response = self.api_client.check_invoice_duplicate(
                invoice_id,
                supplier_id,
                amount,
                due_date,
            )

            if self._is_environment_error(duplicate_response):
                return {
                    "payable": pd.DataFrame(),
                    "processed_count": processed_count,
                    "excluded_count": len(self._all_excluded_ids()),
                    "environment_error": True,
                    "result": self._environment_error_result(
                        duplicate_response,
                        "Environment error during completion duplicate check.",
                    ),
                    "error_result": None,
                }

            if not self._response_succeeded(duplicate_response):
                return {
                    "payable": pd.DataFrame(),
                    "processed_count": processed_count,
                    "excluded_count": len(self._all_excluded_ids()),
                    "environment_error": False,
                    "result": None,
                    "error_result": self._action_result(
                        success=False,
                        useful_action=False,
                        error_type=self._response_error_type(
                            duplicate_response,
                            "INVALID_ACTION",
                        ),
                        message="Completion duplicate check failed.",
                    ),
                }

            duplicate_data = self._response_data(duplicate_response)

            if bool(duplicate_data.get("duplicate", False)):
                self.duplicate_invoice_ids.add(invoice_id_str)
                processed_count += 1
                continue

            supplier_response = self.api_client.validate_supplier(supplier_id)

            if self._is_environment_error(supplier_response):
                return {
                    "payable": pd.DataFrame(),
                    "processed_count": processed_count,
                    "excluded_count": len(self._all_excluded_ids()),
                    "environment_error": True,
                    "result": self._environment_error_result(
                        supplier_response,
                        "Environment error during completion supplier check.",
                    ),
                    "error_result": None,
                }

            if not self._response_succeeded(supplier_response):
                return {
                    "payable": pd.DataFrame(),
                    "processed_count": processed_count,
                    "excluded_count": len(self._all_excluded_ids()),
                    "environment_error": False,
                    "result": None,
                    "error_result": self._action_result(
                        success=False,
                        useful_action=False,
                        error_type=self._response_error_type(
                            supplier_response,
                            "INVALID_ACTION",
                        ),
                        message="Completion supplier validation failed.",
                    ),
                }

            supplier_data = self._response_data(supplier_response)
            valid_supplier = bool(
                supplier_data.get(
                    "valid",
                    supplier_data.get("eligible", True),
                )
            )

            if not valid_supplier:
                self.invalid_supplier_invoice_ids.add(invoice_id_str)
                processed_count += 1
                continue

            candidate_rows.append(invoice)
            processed_count += 1

        candidates = (
            pd.DataFrame(candidate_rows)
            if candidate_rows
            else pd.DataFrame(columns=outstanding.columns)
        )

        if candidates.empty:
            # No supplier/duplicate-valid outstanding invoice exists.
            self.budget_excluded_invoice_ids.clear()
            self._update_invoice_states()
            return {
                "payable": candidates,
                "processed_count": processed_count,
                "excluded_count": len(self._all_excluded_ids()),
                "environment_error": False,
                "result": None,
                "error_result": None,
            }

        allocation = self._allocate_budget(
            candidates,
            replace_budget_exclusions=True,
        )

        if allocation["environment_error"]:
            return {
                "payable": pd.DataFrame(),
                "processed_count": processed_count,
                "excluded_count": len(self._all_excluded_ids()),
                "environment_error": True,
                "result": allocation["result"],
                "error_result": None,
            }

        if allocation["error_result"] is not None:
            return {
                "payable": pd.DataFrame(),
                "processed_count": processed_count,
                "excluded_count": len(self._all_excluded_ids()),
                "environment_error": False,
                "result": None,
                "error_result": allocation["error_result"],
            }

        self._update_invoice_states()

        return {
            "payable": allocation["kept"],
            "processed_count": (processed_count + allocation["processed_count"]),
            "excluded_count": len(self._all_excluded_ids()),
            "environment_error": False,
            "result": None,
            "error_result": None,
        }

    def _allocate_budget(
        self,
        invoices,
        replace_budget_exclusions=False,
    ):
        """
        Sequentially allocate each department's remaining budget.

        Example with remaining budget 10,000 and invoices in order:
            5,000 -> keep, remaining allocation capacity 5,000
            6,000 -> skip
            3,000 -> keep, remaining allocation capacity 2,000

        The backend budget is NOT deducted here. Actual deduction only
        occurs when PAY_INVOICES succeeds.
        """

        if replace_budget_exclusions:
            self.budget_excluded_invoice_ids.clear()

        if invoices.empty:
            return {
                "kept": invoices.copy(),
                "processed_count": 0,
                "excluded_count": 0,
                "budget_results": {},
                "environment_error": False,
                "result": None,
                "error_result": None,
            }

        department_column = self._find_column(
            invoices,
            ["department", "category", "function"],
        )
        amount_column = self._find_column(
            invoices,
            ["amount", "totalAmount"],
        )

        if department_column is None or amount_column is None:
            return {
                "kept": pd.DataFrame(),
                "processed_count": 0,
                "excluded_count": 0,
                "budget_results": {},
                "environment_error": False,
                "result": None,
                "error_result": self._action_result(
                    success=False,
                    useful_action=False,
                    error_type="INVALID_STATE_DATA",
                    message=("Department/category or invoice amount is missing."),
                ),
            }

        valid_indices = []
        budget_results = {}
        processed_count = 0
        excluded_count = 0

        departments = invoices[department_column].dropna().unique().tolist()

        for department in departments:
            department_invoices = invoices[invoices[department_column] == department]

            # New backend contract: amount is optional when the
            # environment only needs current remainingBudget.
            response = self.api_client.check_budget(department=department)

            if self._is_environment_error(response):
                return {
                    "kept": pd.DataFrame(),
                    "processed_count": processed_count,
                    "excluded_count": excluded_count,
                    "budget_results": budget_results,
                    "environment_error": True,
                    "result": self._environment_error_result(
                        response,
                        "Environment error during budget checking.",
                    ),
                    "error_result": None,
                }

            if not self._response_succeeded(response):
                return {
                    "kept": pd.DataFrame(),
                    "processed_count": processed_count,
                    "excluded_count": excluded_count,
                    "budget_results": budget_results,
                    "environment_error": False,
                    "result": None,
                    "error_result": self._action_result(
                        success=False,
                        useful_action=False,
                        error_type=self._response_error_type(
                            response,
                            "INVALID_ACTION",
                        ),
                        message="Budget request failed.",
                    ),
                }

            data = self._response_data(response)
            found = bool(data.get("found", data.get("budget") is not None))
            budget = data.get("budget")

            # Missing budget is a successful business finding. All
            # invoices in that department are excluded.
            if not found or not isinstance(budget, dict):
                removed = []

                for index, invoice in department_invoices.iterrows():
                    invoice_id = self._invoice_id(invoice)
                    if invoice_id is not None:
                        self.budget_excluded_invoice_ids.add(str(invoice_id))
                    removed.append(
                        {
                            "index": index,
                            "reason": "BUDGET_NOT_FOUND",
                        }
                    )
                    processed_count += 1
                    excluded_count += 1

                budget_results[department] = {
                    "remaining_budget": None,
                    "kept_total": 0.0,
                    "kept_invoices": [],
                    "removed_invoices": removed,
                }
                continue

            remaining_budget = budget.get("remainingBudget")

            if remaining_budget is None:
                return {
                    "kept": pd.DataFrame(),
                    "processed_count": processed_count,
                    "excluded_count": excluded_count,
                    "budget_results": budget_results,
                    "environment_error": False,
                    "result": None,
                    "error_result": self._action_result(
                        success=False,
                        useful_action=False,
                        error_type="INVALID_RESPONSE",
                        message=(f"Remaining budget missing for {department}."),
                    ),
                }

            remaining_budget = float(remaining_budget)
            current_total = 0.0
            kept = []
            removed = []

            for index, invoice in department_invoices.iterrows():
                amount = pd.to_numeric(
                    invoice[amount_column],
                    errors="coerce",
                )

                processed_count += 1
                invoice_id = self._invoice_id(invoice)

                if pd.isna(amount):
                    if invoice_id is not None:
                        self.budget_excluded_invoice_ids.add(str(invoice_id))
                    removed.append(
                        {
                            "index": index,
                            "reason": "INVALID_AMOUNT",
                        }
                    )
                    excluded_count += 1
                    continue

                amount = float(amount)

                if current_total + amount <= remaining_budget:
                    valid_indices.append(index)
                    current_total += amount
                    kept.append(
                        {
                            "index": index,
                            "amount": amount,
                        }
                    )
                else:
                    if invoice_id is not None:
                        self.budget_excluded_invoice_ids.add(str(invoice_id))
                    removed.append(
                        {
                            "index": index,
                            "amount": amount,
                            "reason": "BUDGET_EXCEEDED",
                        }
                    )
                    excluded_count += 1

            budget_results[department] = {
                "remaining_budget": remaining_budget,
                "kept_total": current_total,
                "kept_invoices": kept,
                "removed_invoices": removed,
            }

        kept_df = invoices.loc[valid_indices].copy()
        kept_df.reset_index(drop=True, inplace=True)

        return {
            "kept": kept_df,
            "processed_count": processed_count,
            "excluded_count": excluded_count,
            "budget_results": budget_results,
            "environment_error": False,
            "result": None,
            "error_result": None,
        }

    # ==========================================================
    # DataFrame / exclusion helpers
    # ==========================================================

    def _candidate_invoices(self):
        frames = []

        if not self.pending_approval_invoices.empty:
            frames.append(self.pending_approval_invoices.copy())

        if not self.approved_invoices.empty:
            frames.append(self.approved_invoices.copy())

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _all_excluded_ids(self):
        return (
            set(self.duplicate_invoice_ids)
            | set(self.invalid_supplier_invoice_ids)
            | set(self.budget_excluded_invoice_ids)
        )

    def _filter_known_exclusions(self, df):
        if df.empty:
            return df.copy()

        excluded = self._all_excluded_ids()

        if not excluded:
            return df.copy()

        id_column = self._find_column(
            df,
            ["_id", "id", "invoiceId"],
        )

        if id_column is None:
            return df.copy()

        mask = ~df[id_column].astype(str).isin(excluded)
        return df.loc[mask].copy()

    def _apply_known_exclusions_to_local_frames(self):
        self.pending_approval_invoices = self._filter_known_exclusions(
            self.pending_approval_invoices
        )
        self.approved_invoices = self._filter_known_exclusions(self.approved_invoices)

        self.pending_approval_invoices.reset_index(
            drop=True,
            inplace=True,
        )
        self.approved_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        self._update_invoice_states()

    def _split_invoice_frames(
        self,
        invoices,
        apply_exclusions=True,
    ):
        if invoices.empty:
            self.paid_invoices = pd.DataFrame()
            self.rejected_invoices = pd.DataFrame()
            self.pending_approval_invoices = pd.DataFrame()
            self.approved_invoices = pd.DataFrame()
            return None

        status_column = self._find_column(
            invoices,
            ["status", "invoiceStatus"],
        )

        if status_column is None:
            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_STATE_DATA",
                message=("Invoice response does not contain a status column."),
            )

        self.paid_invoices = invoices[invoices[status_column] == "PAID"].copy()
        self.rejected_invoices = invoices[invoices[status_column] == "REJECTED"].copy()
        self.pending_approval_invoices = invoices[
            invoices[status_column] == "PENDING_APPROVAL"
        ].copy()
        self.approved_invoices = invoices[invoices[status_column] == "APPROVED"].copy()

        if apply_exclusions:
            self._apply_known_exclusions_to_local_frames()

        return None

    @staticmethod
    def _find_column(df, candidates):
        for column in candidates:
            if column in df.columns:
                return column
        return None

    def _invoice_id(self, row):
        return self._get_value(
            row,
            ["_id", "id", "invoiceId"],
        )

    @staticmethod
    def _get_value(row, candidates):
        for column in candidates:
            if column not in row.index:
                continue

            value = row[column]

            if isinstance(value, dict):
                nested_id = value.get("_id") or value.get("id")
                if nested_id is not None:
                    return nested_id

            try:
                if pd.notna(value):
                    return value
            except (TypeError, ValueError):
                return value

        return None

    # ==========================================================
    # API response helpers
    # ==========================================================

    @staticmethod
    def _is_environment_error(response):
        if not isinstance(response, dict):
            return False

        return bool(
            response.get(
                "environment_error",
                response.get("environmentError", False),
            )
        )

    @staticmethod
    def _response_data(response):
        if not isinstance(response, dict):
            return {}

        data = response.get("data", {})

        if (
            isinstance(data, dict)
            and "data" in data
            and isinstance(data["data"], (dict, list))
        ):
            return data["data"]

        return data

    @staticmethod
    def _response_error_type(response, default=None):
        if not isinstance(response, dict):
            return default

        payload = response.get("data", {})

        if isinstance(payload, dict):
            error_type = payload.get("errorType") or payload.get("error_type")

            if error_type:
                return str(error_type).upper()

            # Some validation endpoints use reason rather than
            # errorType because the API action itself succeeded.
            reason = payload.get("reason")
            if reason and payload.get("success") is False:
                return str(reason).upper()

            nested = payload.get("data")
            if isinstance(nested, dict):
                error_type = nested.get("errorType") or nested.get("error_type")
                if error_type:
                    return str(error_type).upper()

        return default

    @staticmethod
    def _response_message(response, default=""):
        if not isinstance(response, dict):
            return default

        payload = response.get("data", {})

        if isinstance(payload, dict):
            message = payload.get("message")
            if message:
                return str(message)

        return str(response.get("message", default))

    @classmethod
    def _response_succeeded(cls, response):
        if not isinstance(response, dict):
            return False

        if not response.get("success", False):
            return False

        payload = response.get("data")

        if isinstance(payload, dict) and "success" in payload:
            return bool(payload["success"])

        return not cls._is_environment_error(response)

    def _select_error_type(self, errors):
        errors = [error for error in errors if error]

        if not errors:
            return "UNKNOWN_BUSINESS_ERROR"

        return min(
            errors,
            key=lambda error: (self.reward_processor.get_error_penalty(error)),
        )

    # ==========================================================
    # Close
    # ==========================================================

    def close(self, terminated_reason="FAILED"):
        if self.episode_active:
            self._end_episode(terminated_reason)

        if hasattr(self.api_client, "close"):
            self.api_client.close()
        elif hasattr(self.api_client, "session"):
            self.api_client.session.close()