"""
finance_env.py

Finance reinforcement-learning environment.

Supports:

    - PPO baseline
    - LLM + PPO input guidance
    - LLM + PPO reward shaping
    - LLM + PPO input + reward guidance

Business actions report outcomes.

RewardProcessor owns reward calculation.
ProcedureTracker owns LLM procedure progress.
"""

from datetime import (
    datetime,
    timedelta,
    UTC,
)

import random
import time

import numpy as np
import pandas as pd

from config.config import config

from environment.action_space import (
    ActionSpace,
    FinanceAction,
)

from environment.state_encoder import (
    StateEncoder,
)

from environment.reward_processor import (
    RewardProcessor,
)

from environment.procedure_tracker import (
    ProcedureTracker,
)


class FinanceEnvironment:

    def __init__(
        self,
        api_client,
        llm_plan=None,
    ):

        self.api_client = api_client

        # ==========================================================
        # Environment Configuration
        # ==========================================================

        self.max_steps = config.environment.MAX_STEPS_PER_EPISODE

        self.seed = config.environment.RANDOM_SEED

        self.observation_type = config.environment.OBSERVATION_TYPE

        # ==========================================================
        # Agent Configuration
        # ==========================================================

        self.agent_type = config.agent.AGENT_TYPE

        self.algorithm = config.agent.ALGORITHM

        self.goal = config.agent.TASK

        # ==========================================================
        # Experiment Configuration
        # ==========================================================

        self.phase = config.experiment.PHASE

        self.experiment_name = config.experiment.EXPERIMENT_NAME

        self.guidance_mode = config.experiment.GUIDANCE_MODE

        self.guidance_bonus = config.experiment.GUIDANCE_BONUS

        self.llm_model = config.llm.MODEL if self.agent_type == "LLM_RL" else None

        # ==========================================================
        # Environment Components
        # ==========================================================

        self.action_space_handler = ActionSpace()

        self.state_encoder = StateEncoder()

        # ==========================================================
        # Reward Processor
        # ==========================================================

        self.reward_processor = RewardProcessor(
            use_guidance=self._uses_reward_guidance(),
            guidance_bonus=self.guidance_bonus,
        )

        # ==========================================================
        # LLM Procedure
        # ==========================================================

        self.llm_plan = []

        self.llm_procedure = []

        self.procedure_tracker = ProcedureTracker(
            procedure=[],
            action_dim=self.action_space_handler.action_count,
        )

        self.set_llm_plan(llm_plan or [])

        # ==========================================================
        # Rewarded Actions
        #
        # Prevent positive reward farming.
        # ==========================================================

        self.rewarded_actions = set()

        # ==========================================================
        # Episode Tracking
        # ==========================================================

        self.episode_id = None

        self.episode_number = None

        self.episode_active = False

        # ==========================================================
        # Step Tracking
        # ==========================================================

        self.current_step = 0

        # ==========================================================
        # DataFrames
        # ==========================================================

        self.all_invoices = pd.DataFrame()

        self.paid_invoices = pd.DataFrame()

        self.rejected_invoices = pd.DataFrame()

        self.pending_approval_invoices = pd.DataFrame()

        self.approved_invoices = pd.DataFrame()

        self.report_df = pd.DataFrame()

        # ==========================================================
        # State
        # ==========================================================

        self.state = self._initial_state()

    # ==========================================================
    # Guidance Modes
    # ==========================================================

    def _uses_reward_guidance(
        self,
    ):

        return self.agent_type == "LLM_RL" and self.guidance_mode in {
            "REWARD_SHAPING",
            "INPUT_AND_REWARD",
        }

    def _uses_input_guidance(
        self,
    ):

        return self.agent_type == "LLM_RL" and self.guidance_mode in {
            "INPUT",
            "INPUT_AND_REWARD",
        }

    # ==========================================================
    # LLM Plan
    # ==========================================================

    def set_llm_plan(
        self,
        plan,
    ):
        """
        Accept either:

            action IDs:
                [0, 1, 2, 3, 5, 4, 7]

        or:

            action names:
                [
                    "GET_INVOICES",
                    ...
                ]

        Backend Episode.llmPlan stores action names.

        ProcedureTracker stores action IDs.
        """

        names = []

        action_ids = []

        for item in plan or []:

            if isinstance(
                item,
                str,
            ):

                name = (
                    item.strip()
                    .upper()
                    .replace(
                        "-",
                        "_",
                    )
                    .replace(
                        " ",
                        "_",
                    )
                )

                try:

                    action = FinanceAction[name]

                except KeyError as exc:

                    raise ValueError("Unknown LLM action: " f"{item}") from exc

            else:

                try:

                    action = FinanceAction(int(item))

                except (
                    TypeError,
                    ValueError,
                ) as exc:

                    raise ValueError("Invalid LLM action: " f"{item}") from exc

            names.append(action.name)

            action_ids.append(int(action.value))

        self.llm_plan = names

        self.llm_procedure = action_ids

        self.procedure_tracker.set_procedure(action_ids)

    # ==========================================================
    # State
    # ==========================================================

    def _initial_state(
        self,
    ):

        return {
            "get_invoices": False,
            "check_duplicate": False,
            "check_supplier": False,
            "approve_invoices": False,
            "pay_invoices": False,
            "check_budget": False,
            "generate_report": False,
            "check_payment_completed": False,
            "has_paid_invoices": False,
            "has_rejected_invoices": False,
            "has_pending_approval_invoices": False,
            "has_approved_invoices": False,
            "task_completed": False,
        }

    def _update_invoice_states(
        self,
    ):

        self.state["has_paid_invoices"] = not self.paid_invoices.empty

        self.state["has_rejected_invoices"] = not self.rejected_invoices.empty

        self.state["has_pending_approval_invoices"] = (
            not self.pending_approval_invoices.empty
        )

        self.state["has_approved_invoices"] = not self.approved_invoices.empty

    def get_state(
        self,
    ):

        self._update_invoice_states()

        return self.state.copy()

    # ==========================================================
    # PPO Observation
    # ==========================================================

    def get_observation(
        self,
    ):

        return self.state_encoder.encode(self.get_state())

    # ==========================================================
    # LLM Guidance Input
    # ==========================================================

    def get_guidance_vector(
        self,
    ):

        return np.asarray(
            self.procedure_tracker.get_guidance(),
            dtype=np.float32,
        )

    def get_guided_observation(
        self,
    ):
        """
        Used by future LLM_RLAgent when guidance mode is:

            INPUT
            INPUT_AND_REWARD

        Base state = 13 values
        Guidance   = 8 values

        Combined = 21 values.
        """

        base = self.get_observation()

        guidance = self.get_guidance_vector()

        return np.concatenate(
            [
                base,
                guidance,
            ]
        ).astype(np.float32)

    # ==========================================================
    # Standard Action Result
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
            # Filled by step() after ProcedureTracker checks it.
            "procedure_followed": None,
            "error_type": error_type,
            "environment_error": bool(environment_error),
            "message": message,
        }

        result.update(extra)

        return result

    # ==========================================================
    # Environment Error Result
    # ==========================================================

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

    # ==========================================================
    # Action Endpoint
    # ==========================================================

    @staticmethod
    def _get_action_endpoint(
        action_name,
    ):

        endpoints = {
            "GET_INVOICES": "GET /invoice",
            "CHECK_DUPLICATE": "POST /invoice/duplicate-check",
            "CHECK_SUPPLIER": "POST /supplier/validate",
            "APPROVE_INVOICES": "PATCH /approval/approve",
            "PAY_INVOICES": "POST /payment/pay",
            "CHECK_BUDGET": "POST /account/budget/check",
            "GENERATE_REPORT": "POST /report/generate-report",
            "CHECK_PAYMENT_COMPLETED": ("GET /invoice + " "POST /account/budget/check"),
        }

        return endpoints.get(action_name)

    # ==========================================================
    # Episode Start
    # ==========================================================

    def _start_episode(
        self,
    ):

        payload = {
            "agentType": self.agent_type,
            "algorithm": self.algorithm,
            "goal": self.goal,
            "phase": self.phase,
            "experimentName": self.experiment_name,
            "seed": self.seed,
            "llmModel": self.llm_model,
            "guidanceMode": self.guidance_mode,
            "llmPlan": self.llm_plan,
            "initialState": self.get_state().copy(),
        }

        response = self.api_client.start_episode(payload)

        if self._is_environment_error(response):

            raise RuntimeError("Environment error while " "starting episode.")

        if not self._response_succeeded(response):

            raise RuntimeError("Failed to start episode.")

        data = self._response_data(response)

        self.episode_id = data.get("episodeId")

        self.episode_number = data.get("episodeNumber")

        if not self.episode_id:

            raise RuntimeError("Episode started but no " "episodeId was returned.")

        self.episode_active = True

    # ==========================================================
    # Record Episode Step
    # ==========================================================

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

    # ==========================================================
    # End Episode
    # ==========================================================

    def _end_episode(
        self,
        terminated_reason,
    ):

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

    def reset(
        self,
        seed=None,
        options=None,
    ):

        options = options or {}

        if self.episode_active:

            self._end_episode("RESET")

        # ======================================================
        # Metadata
        # ======================================================

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

        self.goal = options.get(
            "goal",
            self.goal,
        )

        self.phase = options.get(
            "phase",
            self.phase,
        )

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

        # ======================================================
        # Reward Guidance Configuration
        # ======================================================

        self.reward_processor.configure_guidance(
            use_guidance=self._uses_reward_guidance(),
            guidance_bonus=self.guidance_bonus,
        )

        # ======================================================
        # LLM Procedure
        # ======================================================

        if "llm_plan" in options:

            self.set_llm_plan(options["llm_plan"] or [])

        else:

            self.procedure_tracker.reset()

        if self.agent_type != "LLM_RL":

            self.set_llm_plan([])

            self.llm_model = None

        # ======================================================
        # Reset Reward Tracking
        # ======================================================

        self.rewarded_actions.clear()

        # ======================================================
        # Reset Backend
        # ======================================================

        response = self.api_client.reset_environment()

        if self._is_environment_error(response):

            raise RuntimeError("Backend environment reset failed.")

        if not self._response_succeeded(response):

            raise RuntimeError("Backend rejected environment reset.")

        # ======================================================
        # Reset Local Environment
        # ======================================================

        self.current_step = 0

        self.all_invoices = pd.DataFrame()

        self.paid_invoices = pd.DataFrame()

        self.rejected_invoices = pd.DataFrame()

        self.pending_approval_invoices = pd.DataFrame()

        self.approved_invoices = pd.DataFrame()

        self.report_df = pd.DataFrame()

        self.state = self._initial_state()

        # ======================================================
        # Backend Episode
        # ======================================================

        self._start_episode()

        return self.get_observation()

    # ==========================================================
    # STEP
    # ==========================================================

    def step(
        self,
        action,
    ):

        # ======================================================
        # Already Terminated
        # ======================================================

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
                    "episode_id": self.episode_id,
                    "episode_number": self.episode_number,
                },
            )

        # ======================================================
        # State Before
        # ======================================================

        state_before = self.get_state().copy()

        self.current_step += 1

        start_time = time.perf_counter()

        # ======================================================
        # Resolve Action
        # ======================================================

        try:

            action_name = self.action_space_handler.get_action_name(action)

            endpoint = self._get_action_endpoint(action_name)

            result = self.action_space_handler.execute(
                self,
                action,
            )

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

            action_name = locals().get(
                "action_name",
                str(action),
            )

            endpoint = locals().get("endpoint")

            result = self._action_result(
                success=False,
                useful_action=False,
                environment_error=True,
                message=str(exc),
            )

        duration_ms = (time.perf_counter() - start_time) * 1000.0

        # ======================================================
        # Outcome
        # ======================================================

        environment_error = bool(
            result.get(
                "environment_error",
                False,
            )
        )

        action_success = bool(
            result.get(
                "success",
                False,
            )
        )

        useful_action = bool(
            result.get(
                "useful_action",
                False,
            )
        )

        error_type = result.get("error_type")

        # ======================================================
        # LLM Procedure
        # ======================================================

        if self.agent_type == "LLM_RL" and self.procedure_tracker.has_procedure():

            procedure_followed = self.procedure_tracker.check_action(
                action=action,
                action_succeeded=(action_success and not environment_error),
            )

        else:

            procedure_followed = None

        result["procedure_followed"] = procedure_followed

        # ======================================================
        # Repeated Positive Action Detection
        # ======================================================

        repeated_action = (
            action_name in self.rewarded_actions and action_success and useful_action
        )

        # ======================================================
        # Task State After Action
        # ======================================================

        task_completed = bool(self.state["task_completed"])

        # ======================================================
        # Reward
        # ======================================================

        reward_result = self.reward_processor.process(
            action_name=action_name,
            action_success=action_success,
            useful_action=useful_action,
            environment_error=environment_error,
            error_type=error_type,
            task_completed=task_completed,
            procedure_followed=procedure_followed,
            repeated_action=repeated_action,
        )

        reward = reward_result["reward"]

        base_reward = reward_result["base_reward"]

        guidance_bonus = reward_result["guidance_bonus"]

        completion_bonus = reward_result["completion_bonus"]

        # ======================================================
        # Prevent Reward Farming
        # ======================================================

        if base_reward > 0 and action_success and useful_action and not repeated_action:

            self.rewarded_actions.add(action_name)

        # ======================================================
        # State After
        # ======================================================

        state_after = self.get_state().copy()

        # ======================================================
        # Episode Step Logging
        # ======================================================

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

        # ======================================================
        # Done
        # ======================================================

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

        # ======================================================
        # Info
        # ======================================================

        info = {
            "action": action_name,
            "step": self.current_step,
            "success": action_success,
            "useful_action": useful_action,
            "error_type": error_type,
            "environment_error": environment_error,
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
            "procedure_status": self.procedure_tracker.get_status(),
        }

        return (
            self.get_observation(),
            float(reward),
            done,
            info,
        )

    # ==========================================================
    # ACTION 0
    # GET INVOICES
    # ==========================================================

    def get_invoices(
        self,
    ):

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
            data.get(
                "invoices",
                data.get(
                    "data",
                    [],
                ),
            )
            if isinstance(
                data,
                dict,
            )
            else data
        )

        self.all_invoices = pd.DataFrame(invoices)

        if self.all_invoices.empty:

            self.paid_invoices = pd.DataFrame()

            self.rejected_invoices = pd.DataFrame()

            self.pending_approval_invoices = pd.DataFrame()

            self.approved_invoices = pd.DataFrame()

        else:

            status_column = self._find_column(
                self.all_invoices,
                [
                    "status",
                    "invoiceStatus",
                ],
            )

            if status_column is None:

                return self._action_result(
                    success=False,
                    useful_action=False,
                    error_type="INVALID_STATE_DATA",
                    message=("Invoice response does not " "contain a status column."),
                )

            self.paid_invoices = self.all_invoices[
                self.all_invoices[status_column] == "PAID"
            ].copy()

            self.rejected_invoices = self.all_invoices[
                self.all_invoices[status_column] == "REJECTED"
            ].copy()

            self.pending_approval_invoices = self.all_invoices[
                self.all_invoices[status_column] == "PENDING_APPROVAL"
            ].copy()

            self.approved_invoices = self.all_invoices[
                self.all_invoices[status_column] == "APPROVED"
            ].copy()

        self.state["get_invoices"] = True

        self._update_invoice_states()

        return self._action_result(
            success=True,
            useful_action=True,
            message="Invoices loaded.",
        )

    # ==========================================================
    # ACTION 1
    # CHECK DUPLICATE
    # ==========================================================

    def check_duplicate(
        self,
    ):

        if not self.state["get_invoices"]:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded " "before duplicate checking."),
            )

        if self.pending_approval_invoices.empty and self.approved_invoices.empty:

            self.state["check_duplicate"] = True

            return self._action_result(
                success=True,
                useful_action=False,
                message=("No invoices available " "for duplicate checking."),
            )

        frames = []

        if not self.pending_approval_invoices.empty:

            frames.append(
                (
                    "pending",
                    self.pending_approval_invoices.copy(),
                )
            )

        if not self.approved_invoices.empty:

            frames.append(
                (
                    "approved",
                    self.approved_invoices.copy(),
                )
            )

        for (
            frame_name,
            frame,
        ) in frames:

            remove_indices = []

            for (
                index,
                invoice,
            ) in frame.iterrows():

                supplier_id = self._get_value(
                    invoice,
                    [
                        "supplierId",
                        "supplier",
                        "supplier_id",
                    ],
                )

                amount = self._get_value(
                    invoice,
                    [
                        "amount",
                        "totalAmount",
                    ],
                )

                due_date = self._get_value(
                    invoice,
                    [
                        "dueDate",
                        "due_date",
                    ],
                )

                if supplier_id is None or amount is None or due_date is None:

                    return self._action_result(
                        success=False,
                        useful_action=False,
                        error_type="MISSING_REQUIRED_FIELD",
                        message=(
                            "Invoice is missing data "
                            "required for duplicate checking."
                        ),
                    )

                response = self.api_client.invoice_dupplicate_check(
                    supplier_id,
                    amount,
                    due_date,
                )

                if self._is_environment_error(response):

                    return self._environment_error_result(
                        response,
                        ("Environment error during " "duplicate checking."),
                    )

                data = self._response_data(response)

                error_type = self._response_error_type(response)

                # Some backends may represent duplicate
                # detection using a business error response.
                if error_type == "DUPLICATE_INVOICE":

                    duplicate = True

                elif not self._response_succeeded(response):

                    return self._action_result(
                        success=False,
                        useful_action=False,
                        error_type=error_type or "INVALID_ACTION",
                        message=self._response_message(
                            response,
                            "Duplicate check failed.",
                        ),
                    )

                else:

                    duplicate = bool(
                        data.get(
                            "duplicate",
                            False,
                        )
                        if isinstance(
                            data,
                            dict,
                        )
                        else False
                    )

                if duplicate:

                    remove_indices.append(index)

            if frame_name == "pending":

                self.pending_approval_invoices = frame.drop(remove_indices)

            else:

                self.approved_invoices = frame.drop(remove_indices)

        self.state["check_duplicate"] = True

        self._update_invoice_states()

        return self._action_result(
            success=True,
            useful_action=True,
            message="Duplicate invoices checked.",
        )

    # ==========================================================
    # ACTION 2
    # CHECK SUPPLIER
    # ==========================================================

    def check_supplier(
        self,
    ):

        if not self.state["get_invoices"]:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded " "before supplier validation."),
            )

        if self.pending_approval_invoices.empty and self.approved_invoices.empty:

            self.state["check_supplier"] = True

            return self._action_result(
                success=True,
                useful_action=False,
                message=("No invoices available " "for supplier checking."),
            )

        frames = []

        if not self.pending_approval_invoices.empty:

            frames.append(
                (
                    "pending",
                    self.pending_approval_invoices.copy(),
                )
            )

        if not self.approved_invoices.empty:

            frames.append(
                (
                    "approved",
                    self.approved_invoices.copy(),
                )
            )

        supplier_rejection_errors = {
            "SUPPLIER_NOT_FOUND",
            "SUPPLIER_INACTIVE",
            "SUPPLIER_HIGH_RISK",
        }

        for (
            frame_name,
            frame,
        ) in frames:

            remove_indices = []

            for (
                index,
                invoice,
            ) in frame.iterrows():

                supplier_id = self._get_value(
                    invoice,
                    [
                        "supplierId",
                        "supplier",
                        "supplier_id",
                    ],
                )

                if supplier_id is None:

                    remove_indices.append(index)

                    continue

                response = self.api_client.validate_supplier(supplier_id)

                if self._is_environment_error(response):

                    return self._environment_error_result(
                        response,
                        ("Environment error during " "supplier validation."),
                    )

                error_type = self._response_error_type(response)

                if error_type in supplier_rejection_errors:

                    valid = False

                elif not self._response_succeeded(response):

                    return self._action_result(
                        success=False,
                        useful_action=False,
                        error_type=error_type or "INVALID_ACTION",
                        message=self._response_message(
                            response,
                            "Supplier validation failed.",
                        ),
                    )

                else:

                    data = self._response_data(response)

                    valid = (
                        data.get(
                            "valid",
                            data.get(
                                "isValid",
                                True,
                            ),
                        )
                        if isinstance(
                            data,
                            dict,
                        )
                        else True
                    )

                if not valid:

                    remove_indices.append(index)

            if frame_name == "pending":

                self.pending_approval_invoices = frame.drop(remove_indices)

            else:

                self.approved_invoices = frame.drop(remove_indices)

        self.state["check_supplier"] = True

        self._update_invoice_states()

        return self._action_result(
            success=True,
            useful_action=True,
            message="Suppliers checked.",
        )

    # ==========================================================
    # ACTION 3
    # APPROVE INVOICES
    # ==========================================================

    def approve_invoices(
        self,
    ):

        if not self.state["get_invoices"]:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded " "before approval."),
            )

        if self.pending_approval_invoices.empty:

            self.state["approve_invoices"] = True

            return self._action_result(
                success=True,
                useful_action=False,
                message=("No pending invoices " "to approve."),
            )

        approved_indices = []

        approved_rows = []

        errors = []

        for (
            index,
            invoice,
        ) in self.pending_approval_invoices.iterrows():

            invoice_id = self._get_value(
                invoice,
                [
                    "_id",
                    "id",
                    "invoiceId",
                ],
            )

            if invoice_id is None:

                errors.append("MISSING_REQUIRED_FIELD")

                continue

            response = self.api_client.approve_invoice(invoice_id)

            if self._is_environment_error(response):

                return self._environment_error_result(
                    response,
                    ("Environment error during " "invoice approval."),
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

        # IMPORTANT:
        # Only remove successfully approved invoices.
        self.pending_approval_invoices = self.pending_approval_invoices.drop(
            approved_indices,
            errors="ignore",
        )

        self.pending_approval_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        self.state["approve_invoices"] = True

        self._update_invoice_states()

        if errors:

            return self._action_result(
                success=False,
                useful_action=bool(approved_indices),
                error_type=self._select_error_type(errors),
                message=("One or more invoices " "could not be approved."),
            )

        return self._action_result(
            success=True,
            useful_action=bool(approved_indices),
            message="Invoices approved.",
        )

    # ==========================================================
    # ACTION 4
    # PAY INVOICES
    # ==========================================================

    def pay_invoices(
        self,
    ):

        if not self.state["get_invoices"]:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded " "before payment."),
            )

        if self.approved_invoices.empty:

            self.state["pay_invoices"] = True

            return self._action_result(
                success=True,
                useful_action=False,
                message=("No approved invoices " "to pay."),
            )

        paid_indices = []

        paid_rows = []

        errors = []

        accounts_response = self.api_client.get_accounts()

        if self._is_environment_error(accounts_response):

            return self._environment_error_result(
                accounts_response,
                ("Environment error while " "loading payment accounts."),
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
            account_data.get(
                "accounts",
                account_data.get(
                    "data",
                    [],
                ),
            )
            if isinstance(
                account_data,
                dict,
            )
            else account_data
        )

        if not accounts:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="ACCOUNT_NOT_FOUND",
                message=("No payment account available."),
            )

        default_account_id = self._get_value(
            pd.Series(accounts[0]),
            [
                "_id",
                "id",
                "accountId",
            ],
        )

        if default_account_id is None:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="ACCOUNT_NOT_FOUND",
                message=("Payment account ID unavailable."),
            )

        for (
            index,
            invoice,
        ) in self.approved_invoices.iterrows():

            invoice_id = self._get_value(
                invoice,
                [
                    "_id",
                    "id",
                    "invoiceId",
                ],
            )

            if invoice_id is None:

                errors.append("MISSING_REQUIRED_FIELD")

                continue

            account_id = (
                self._get_value(
                    invoice,
                    [
                        "accountId",
                        "account",
                        "account_id",
                    ],
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
                    ("Environment error during " "invoice payment."),
                )

            if self._response_succeeded(response):

                paid_indices.append(index)

                paid_row = invoice.copy()

                if "status" in paid_row.index:

                    paid_row["status"] = "PAID"

                paid_rows.append(paid_row)

            else:

                # Examples:
                #
                # BUDGET_EXCEEDED
                # INSUFFICIENT_BALANCE
                # INVOICE_NOT_APPROVED
                # DUPLICATE_INVOICE
                # SUPPLIER_INACTIVE
                # ACCOUNT_FROZEN
                errors.append(
                    self._response_error_type(
                        response,
                        "PAYMENT_FAILED",
                    )
                )

        if paid_rows:

            self.paid_invoices = pd.concat(
                [
                    self.paid_invoices,
                    pd.DataFrame(paid_rows),
                ],
                ignore_index=True,
            )

        # IMPORTANT:
        # Failed invoices remain approved.
        self.approved_invoices = self.approved_invoices.drop(
            paid_indices,
            errors="ignore",
        )

        self.approved_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        self.state["pay_invoices"] = True

        self._update_invoice_states()

        # ======================================================
        # Business Error
        #
        # We do NOT add one penalty for every failed invoice.
        #
        # One RL action -> one reward.
        #
        # Use the most severe error encountered.
        # ======================================================

        if errors:

            selected_error = self._select_error_type(errors)

            return self._action_result(
                success=False,
                useful_action=bool(paid_indices),
                error_type=selected_error,
                message=("Payment action encountered " f"{selected_error}."),
                paid_count=len(paid_indices),
            )

        if not paid_indices:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="PAYMENT_FAILED",
                message=("No approved invoice " "could be paid."),
            )

        return self._action_result(
            success=True,
            useful_action=True,
            message=("Approved invoices " "processed for payment."),
            paid_count=len(paid_indices),
        )

    # ==========================================================
    # ACTION 5
    # CHECK BUDGET
    # ==========================================================

    def check_budget(
        self,
    ):

        if not self.state["get_invoices"]:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded " "before budget checking."),
            )

        if self.approved_invoices.empty:

            self.state["check_budget"] = True

            return self._action_result(
                success=True,
                useful_action=False,
                message=("No approved invoices " "to check against budget."),
            )

        department_column = self._find_column(
            self.approved_invoices,
            [
                "department",
                "category",
                "function",
            ],
        )

        amount_column = self._find_column(
            self.approved_invoices,
            [
                "amount",
                "totalAmount",
            ],
        )

        if department_column is None:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_STATE_DATA",
                message=("Department/category " "is missing."),
            )

        if amount_column is None:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_STATE_DATA",
                message=("Invoice amount field " "is missing."),
            )

        departments = (
            self.approved_invoices[department_column].dropna().unique().tolist()
        )

        valid_indices = []

        budget_results = {}

        for department in departments:

            department_invoices = self.approved_invoices[
                self.approved_invoices[department_column] == department
            ]

            random_amount = round(
                random.uniform(
                    1.0,
                    100.0,
                ),
                2,
            )

            response = self.api_client.check_budget(
                random_amount,
                department,
            )

            if self._is_environment_error(response):

                return self._environment_error_result(
                    response,
                    ("Environment error during " "budget checking."),
                )

            data = self._response_data(response)

            # Even if the HTTP-level response represents
            # BUDGET_EXCEEDED, use returned budget information
            # when available.
            budget = (
                data.get("budget")
                if isinstance(
                    data,
                    dict,
                )
                else None
            )

            if not isinstance(
                budget,
                dict,
            ):

                return self._action_result(
                    success=False,
                    useful_action=False,
                    error_type=self._response_error_type(
                        response,
                        "BUDGET_NOT_FOUND",
                    ),
                    message=("Budget information missing " f"for {department}."),
                )

            remaining_budget = budget.get("remainingBudget")

            if remaining_budget is None:

                return self._action_result(
                    success=False,
                    useful_action=False,
                    error_type="INVALID_RESPONSE",
                    message=("Remaining budget missing " f"for {department}."),
                )

            remaining_budget = float(remaining_budget)

            current_total = 0.0

            kept = []

            removed = []

            for (
                index,
                invoice,
            ) in department_invoices.iterrows():

                amount = pd.to_numeric(
                    invoice[amount_column],
                    errors="coerce",
                )

                if pd.isna(amount):

                    removed.append(
                        {
                            "index": index,
                            "reason": "INVALID_AMOUNT",
                        }
                    )

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

                    removed.append(
                        {
                            "index": index,
                            "amount": amount,
                            "reason": "BUDGET_EXCEEDED",
                        }
                    )

            budget_results[department] = {
                "remaining_budget": remaining_budget,
                "kept_total": current_total,
                "kept_invoices": kept,
                "removed_invoices": removed,
            }

        self.approved_invoices = self.approved_invoices.loc[valid_indices].copy()

        self.approved_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        self.state["check_budget"] = True

        self._update_invoice_states()

        # CHECK_BUDGET successfully finding invoices that exceed
        # budget is NOT an agent error.
        #
        # It performed exactly the intended validation.
        return self._action_result(
            success=True,
            useful_action=True,
            message="Budget checked.",
            budget_results=budget_results,
        )

    # ==========================================================
    # ACTION 6
    # GENERATE REPORT
    # ==========================================================

    def generate_report(
        self,
    ):

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
                ("Environment error while " "generating report."),
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
            data.get(
                "report",
                data.get(
                    "data",
                    [],
                ),
            )
            if isinstance(
                data,
                dict,
            )
            else data
        )

        if isinstance(
            report,
            dict,
        ):

            report = [report]

        self.report_df = pd.DataFrame(report)

        self.state["generate_report"] = True

        return self._action_result(
            success=True,
            useful_action=True,
            message=("One-year transaction " "report generated."),
        )

    # ==========================================================
    # ACTION 7
    # CHECK PAYMENT COMPLETED
    # ==========================================================

    def check_payment_completed(
        self,
    ):

        if not self.state["get_invoices"]:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_WORKFLOW",
                message=("Invoices must be loaded " "before checking completion."),
            )

        self.state["check_payment_completed"] = True

        # ------------------------------------------------------
        # Locally valid pending invoices still exist.
        #
        # Therefore the task cannot yet be complete.
        # ------------------------------------------------------

        if not self.pending_approval_invoices.empty:

            return self._action_result(
                success=True,
                useful_action=False,
                message=("Valid pending invoices " "still require approval."),
            )

        response = self.api_client.get_invoices()

        if self._is_environment_error(response):

            return self._environment_error_result(
                response,
                ("Environment error while " "checking payment completion."),
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
                    ("Unable to check " "payment completion."),
                ),
            )

        data = self._response_data(response)

        invoices = (
            data.get(
                "invoices",
                data.get(
                    "data",
                    [],
                ),
            )
            if isinstance(
                data,
                dict,
            )
            else data
        )

        current_invoices = pd.DataFrame(invoices)

        if current_invoices.empty:

            self.approved_invoices = pd.DataFrame()

            self.state["task_completed"] = True

            self._update_invoice_states()

            return self._action_result(
                success=True,
                useful_action=True,
                message=("Payment task completed."),
            )

        status_column = self._find_column(
            current_invoices,
            [
                "status",
                "invoiceStatus",
            ],
        )

        if status_column is None:

            return self._action_result(
                success=False,
                useful_action=False,
                error_type="INVALID_STATE_DATA",
                message=("Invoice status unavailable."),
            )

        approved = current_invoices[
            current_invoices[status_column] == "APPROVED"
        ].copy()

        self.approved_invoices = approved

        # ======================================================
        # No payable approved invoices remain.
        # ======================================================

        if approved.empty:

            self.state["task_completed"] = True

            self._update_invoice_states()

            return self._action_result(
                success=True,
                useful_action=True,
                message=("All valid invoices " "have been processed."),
            )

        # ======================================================
        # Determine whether remaining approved invoices are
        # actually payable according to remaining budget.
        # ======================================================

        previous_budget_state = self.state["check_budget"]

        budget_result = self.check_budget()

        # Internal budget check should not pretend that the
        # agent selected CHECK_BUDGET at this step.
        self.state["check_budget"] = previous_budget_state

        if budget_result.get(
            "environment_error",
            False,
        ):

            return budget_result

        if not budget_result.get(
            "success",
            False,
        ):

            return self._action_result(
                success=False,
                useful_action=False,
                error_type=budget_result.get(
                    "error_type",
                    "INVALID_ACTION",
                ),
                message=budget_result.get(
                    "message",
                    ("Unable to determine " "task completion."),
                ),
            )

        # check_budget keeps only currently payable invoices.
        self.state["task_completed"] = self.approved_invoices.empty

        self._update_invoice_states()

        if self.state["task_completed"]:

            return self._action_result(
                success=True,
                useful_action=True,
                message=(
                    "Task completed. Remaining "
                    "approved invoices cannot be "
                    "paid within available budgets."
                ),
            )

        return self._action_result(
            success=True,
            useful_action=False,
            message=("Valid approved invoices " "still require payment."),
        )

    # ==========================================================
    # Helpers
    # ==========================================================

    @staticmethod
    def _is_environment_error(
        response,
    ):

        if not isinstance(
            response,
            dict,
        ):

            return False

        return bool(
            response.get(
                "environment_error",
                response.get(
                    "environmentError",
                    False,
                ),
            )
        )

    @staticmethod
    def _response_data(
        response,
    ):

        if not isinstance(
            response,
            dict,
        ):

            return {}

        data = response.get(
            "data",
            {},
        )

        if (
            isinstance(
                data,
                dict,
            )
            and "data" in data
            and isinstance(
                data["data"],
                (
                    dict,
                    list,
                ),
            )
        ):

            return data["data"]

        return data

    @staticmethod
    def _response_error_type(
        response,
        default=None,
    ):

        if not isinstance(
            response,
            dict,
        ):

            return default

        payload = response.get("data", {})

        if isinstance(
            payload,
            dict,
        ):

            error_type = payload.get("errorType") or payload.get("error_type")

            if error_type:

                return str(error_type).upper()

            nested = payload.get("data")

            if isinstance(
                nested,
                dict,
            ):

                error_type = nested.get("errorType") or nested.get("error_type")

                if error_type:

                    return str(error_type).upper()

        return default

    @staticmethod
    def _response_message(
        response,
        default="",
    ):

        if not isinstance(
            response,
            dict,
        ):

            return default

        payload = response.get("data", {})

        if isinstance(
            payload,
            dict,
        ):

            message = payload.get("message")

            if message:

                return str(message)

        return str(
            response.get(
                "message",
                default,
            )
        )

    @classmethod
    def _response_succeeded(
        cls,
        response,
    ):

        if not isinstance(
            response,
            dict,
        ):

            return False

        if not response.get(
            "success",
            False,
        ):

            return False

        payload = response.get("data")

        if (
            isinstance(
                payload,
                dict,
            )
            and "success" in payload
        ):

            return bool(payload["success"])

        return not (cls._is_environment_error(response))

    def _select_error_type(
        self,
        errors,
    ):
        """
        One high-level action may internally call the backend
        several times.

        We still generate ONE RL reward.

        Therefore use the most severe business error instead
        of summing penalties by record count.
        """

        errors = [error for error in errors if error]

        if not errors:

            return "UNKNOWN_BUSINESS_ERROR"

        return min(
            errors,
            key=lambda error: self.reward_processor.get_error_penalty(error),
        )

    @staticmethod
    def _find_column(
        df,
        candidates,
    ):

        for column in candidates:

            if column in df.columns:

                return column

        return None

    @staticmethod
    def _get_value(
        row,
        candidates,
    ):

        for column in candidates:

            if column not in row.index:

                continue

            value = row[column]

            # Mongo populate can leave supplier as a dictionary.
            if isinstance(
                value,
                dict,
            ):

                nested_id = value.get("_id") or value.get("id")

                if nested_id is not None:

                    return nested_id

            try:

                if pd.notna(value):

                    return value

            except (
                TypeError,
                ValueError,
            ):

                return value

        return None

    # ==========================================================
    # Close
    # ==========================================================

    def close(
        self,
        terminated_reason="FAILED",
    ):

        if self.episode_active:

            self._end_episode(terminated_reason)

        if hasattr(
            self.api_client,
            "session",
        ):

            self.api_client.session.close()
