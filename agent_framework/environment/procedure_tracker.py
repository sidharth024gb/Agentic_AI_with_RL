"""
Prerequisite-aware progress tracker for an LLM-generated procedure.

Used by LLM + PPO for:

    - guidance input
    - procedure-following reward shaping
    - experiment logging

Important design
----------------
The tracker maintains its OWN procedural completion state.

It does not trust FinanceEnvironment action flags as proof that a
procedure step has been completed correctly.

A procedure action becomes complete only when:

    1. the environment action succeeds; and
    2. all prerequisites for that action were already complete
       inside this ProcedureTracker.

PPO is never blocked.

An agent may select any action from the full action space. If an
out-of-order action is nevertheless valid according to its prerequisite
set, the tracker may mark it complete, but ``procedure_followed`` remains
False unless it was also the action currently recommended by the LLM
procedure.

Example
-------

Procedure:

    [0, 1, 2, 3, 5, 4, 7]

Prerequisites:

    {
        0: [],
        1: [0],
        2: [0, 1],
        3: [0, 1, 2],
        5: [0, 1, 2, 3],
        4: [0, 1, 2, 3, 5],
        7: [0, 1, 2, 3, 5, 4],
    }

If action 3 succeeds before action 2 is procedure-complete, action 3 is
NOT marked complete by this tracker. Later guidance will still recommend
the missing prerequisite/action path rather than incorrectly advancing
to action 5.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional


class ProcedureTracker:
    """
    Track prerequisite-valid progress through an LLM procedure.
    """

    def __init__(
        self,
        procedure: Optional[Iterable[int]] = None,
        action_dim: int = 0,
        prerequisites: Optional[Mapping] = None,
    ):
        self.action_dim = int(action_dim)

        self.procedure: List[int] = []
        self.prerequisites: Dict[int, List[int]] = {}
        self.completed: Dict[int, bool] = {}

        self.set_procedure(
            procedure or [],
            prerequisites=prerequisites,
        )

    # ==========================================================
    # Procedure Setup
    # ==========================================================

    def set_procedure(
        self,
        procedure: Iterable[int],
        prerequisites: Optional[Mapping] = None,
    ):
        """
        Install a procedure and reset tracker-owned completion state.

        ``prerequisites`` may contain integer keys or JSON string keys.

        If prerequisites are omitted, a cumulative prerequisite graph
        is derived from the procedure for backwards compatibility:

            [0, 1, 2]

        becomes:

            0: []
            1: [0]
            2: [0, 1]
        """

        procedure = [int(action) for action in procedure]

        self._validate_procedure_actions(procedure)

        if len(set(procedure)) != len(procedure):
            raise ValueError(
                "ProcedureTracker requires unique action IDs. "
                "Duplicate actions should be removed by the plan parser."
            )

        normalized_prerequisites = self._normalize_prerequisites(
            procedure=procedure,
            prerequisites=prerequisites,
        )

        self._validate_prerequisite_graph(
            procedure=procedure,
            prerequisites=normalized_prerequisites,
        )

        self.procedure = procedure
        self.prerequisites = normalized_prerequisites
        self.completed = {action: False for action in self.procedure}

    def _validate_procedure_actions(
        self,
        procedure: List[int],
    ):
        for action in procedure:
            if action < 0 or action >= self.action_dim:
                raise ValueError(
                    "Invalid action "
                    f"{action} in LLM procedure. "
                    f"Action dimension is {self.action_dim}."
                )

    def _normalize_prerequisites(
        self,
        procedure: List[int],
        prerequisites: Optional[Mapping],
    ) -> Dict[int, List[int]]:
        # ------------------------------------------------------
        # Backwards-compatible cumulative graph
        # ------------------------------------------------------

        if prerequisites is None:
            normalized = {}
            completed_before = []

            for action in procedure:
                normalized[action] = list(completed_before)
                completed_before.append(action)

            return normalized

        if not isinstance(prerequisites, Mapping):
            raise ValueError("Procedure prerequisites must be a mapping.")

        normalized = {}

        for raw_action, raw_requirements in prerequisites.items():
            try:
                action = int(raw_action)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid prerequisite action key: {raw_action}"
                ) from exc

            if raw_requirements is None:
                raw_requirements = []

            if not isinstance(
                raw_requirements,
                (list, tuple, set),
            ):
                raise ValueError(
                    "Prerequisites for action "
                    f"{action} must be a list-like collection."
                )

            requirements = []

            for raw_requirement in raw_requirements:
                try:
                    requirement = int(raw_requirement)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "Invalid prerequisite "
                        f"{raw_requirement} for action {action}."
                    ) from exc

                if requirement not in requirements:
                    requirements.append(requirement)

            normalized[action] = requirements

        # Every procedure action must have an explicit entry.
        # Missing entries are interpreted as having no prerequisites.
        for action in procedure:
            normalized.setdefault(action, [])

        # Ignore nothing silently: extra graph nodes are likely a stale
        # or incompatible cached plan.
        extra_actions = set(normalized) - set(procedure)

        if extra_actions:
            raise ValueError(
                "Prerequisite graph contains actions not present "
                f"in the procedure: {sorted(extra_actions)}"
            )

        return normalized

    def _validate_prerequisite_graph(
        self,
        procedure: List[int],
        prerequisites: Dict[int, List[int]],
    ):
        procedure_set = set(procedure)

        for action, requirements in prerequisites.items():
            if action not in procedure_set:
                raise ValueError(f"Action {action} is not present in the procedure.")

            for requirement in requirements:
                if requirement not in procedure_set:
                    raise ValueError(
                        "Prerequisite "
                        f"{requirement} for action {action} "
                        "is not present in the procedure."
                    )

                if requirement == action:
                    raise ValueError(f"Action {action} cannot depend on itself.")

        # ------------------------------------------------------
        # Cycle detection
        # ------------------------------------------------------

        visiting = set()
        visited = set()

        def visit(action):
            if action in visited:
                return

            if action in visiting:
                raise ValueError("Procedure prerequisite graph contains a cycle.")

            visiting.add(action)

            for requirement in prerequisites.get(action, []):
                visit(requirement)

            visiting.remove(action)
            visited.add(action)

        for action in procedure:
            visit(action)

    # ==========================================================
    # Reset / Clear
    # ==========================================================

    def reset(
        self,
        procedure=None,
        prerequisites=None,
    ):
        if procedure is not None:
            self.set_procedure(
                procedure,
                prerequisites=prerequisites,
            )
            return

        self.completed = {action: False for action in self.procedure}

    def clear(
        self,
    ):
        self.procedure = []
        self.prerequisites = {}
        self.completed = {}

    # ==========================================================
    # Basic Information
    # ==========================================================

    def has_procedure(
        self,
    ):
        return bool(self.procedure)

    def get_procedure(
        self,
    ):
        return self.procedure.copy()

    def get_prerequisites(
        self,
    ):
        return {
            action: requirements.copy()
            for action, requirements in self.prerequisites.items()
        }

    def get_completion_map(
        self,
    ):
        return dict(self.completed)

    def get_execution_status(
        self,
    ):
        """
        Backwards-compatible list aligned with procedure order.
        """

        return [bool(self.completed.get(action, False)) for action in self.procedure]

    # ==========================================================
    # Prerequisite State
    # ==========================================================

    def prerequisites_satisfied(
        self,
        action,
    ) -> bool:
        try:
            action = int(action)
        except (TypeError, ValueError):
            return False

        if action not in self.completed:
            return False

        return all(
            self.completed.get(requirement, False)
            for requirement in self.prerequisites.get(action, [])
        )

    def get_eligible_actions(
        self,
    ) -> List[int]:
        """
        Return incomplete actions whose prerequisites are complete.

        Actions are returned in LLM procedure order. This order is used
        only to choose one recommendation; it does not block PPO from
        selecting another eligible or ineligible action.
        """

        return [
            action
            for action in self.procedure
            if (
                not self.completed.get(action, False)
                and self.prerequisites_satisfied(action)
            )
        ]

    def get_blocked_actions(
        self,
    ) -> List[int]:
        return [
            action
            for action in self.procedure
            if (
                not self.completed.get(action, False)
                and not self.prerequisites_satisfied(action)
            )
        ]

    # ==========================================================
    # Current Recommendation
    # ==========================================================

    def get_next_action(
        self,
    ) -> Optional[int]:
        eligible = self.get_eligible_actions()

        if not eligible:
            return None

        return eligible[0]

    def get_next_index(
        self,
    ) -> Optional[int]:
        action = self.get_next_action()

        if action is None:
            return None

        return self.procedure.index(action)

    def is_expected_action(
        self,
        action,
    ) -> Optional[bool]:
        """
        Returns:

            None
                no LLM procedure exists

            True
                action is the current recommendation

            False
                action differs from the current recommendation
        """

        if not self.has_procedure():
            return None

        expected_action = self.get_next_action()

        if expected_action is None:
            return False

        try:
            selected_action = int(action)
        except (TypeError, ValueError):
            return False

        return selected_action == expected_action

    # ==========================================================
    # Mark Complete
    # ==========================================================

    def mark_completed(
        self,
        action,
    ) -> bool:
        """
        Mark an action complete only when its tracker prerequisites
        are currently satisfied.

        This method does NOT inspect FinanceEnvironment state.
        """

        try:
            action = int(action)
        except (TypeError, ValueError):
            return False

        if action not in self.completed:
            return False

        if self.completed[action]:
            return False

        if not self.prerequisites_satisfied(action):
            return False

        self.completed[action] = True
        return True

    # ==========================================================
    # Check Action
    # ==========================================================

    def check_action(
        self,
        action,
        action_succeeded,
    ) -> Optional[bool]:
        """
        Process one PPO-selected action.

        Two concepts are intentionally separated:

        procedure validity
            A successful action whose tracker prerequisites were already
            complete may become procedure-complete.

        procedure adherence / procedure_followed
            True only when PPO selected the action that the tracker was
            currently recommending AND that action succeeded validly.

        Therefore an eligible out-of-order action can be recorded as
        complete without receiving the "followed recommendation" signal.

        Failed actions never become complete.

        Successful no-ops may complete a procedure step if its
        prerequisites are satisfied. RewardProcessor independently
        decides whether a no-op deserves any base/guidance reward.
        """

        if not self.has_procedure():
            return None

        expected_before = self.get_next_action()

        try:
            selected_action = int(action)
        except (TypeError, ValueError):
            return False

        if selected_action not in self.completed:
            return False

        prerequisites_met = self.prerequisites_satisfied(selected_action)

        action_succeeded = bool(action_succeeded)

        procedure_followed = bool(
            expected_before is not None
            and selected_action == expected_before
            and action_succeeded
            and prerequisites_met
        )

        # ------------------------------------------------------
        # Tracker-owned progress
        #
        # Completion is based on prerequisites, NOT on whether this
        # action happened to be the single current recommendation.
        # ------------------------------------------------------

        if action_succeeded and prerequisites_met:
            self.mark_completed(selected_action)

        return procedure_followed

    # ==========================================================
    # Guidance Vector
    # ==========================================================

    def get_guidance(
        self,
    ) -> List[float]:
        """
        One-hot vector representing the current recommendation.

        If the procedure is complete (or empty), returns all zeros.
        """

        guidance = [0.0 for _ in range(self.action_dim)]

        next_action = self.get_next_action()

        if next_action is None:
            return guidance

        guidance[next_action] = 1.0

        return guidance

    # ==========================================================
    # Progress
    # ==========================================================

    def completed_count(
        self,
    ):
        return sum(1 for action in self.procedure if self.completed.get(action, False))

    def remaining_count(
        self,
    ):
        return len(self.procedure) - self.completed_count()

    def is_complete(
        self,
    ):
        return bool(self.procedure) and self.remaining_count() == 0

    def progress_ratio(
        self,
    ):
        if not self.procedure:
            return 0.0

        return self.completed_count() / len(self.procedure)

    # ==========================================================
    # Status
    # ==========================================================

    def get_status(
        self,
    ):
        return {
            "procedure": self.get_procedure(),
            "prerequisites": self.get_prerequisites(),
            "executed": self.get_execution_status(),
            "completedByAction": self.get_completion_map(),
            "eligibleActions": self.get_eligible_actions(),
            "blockedActions": self.get_blocked_actions(),
            "nextAction": self.get_next_action(),
            "nextIndex": self.get_next_index(),
            "guidance": self.get_guidance(),
            "completed": self.is_complete(),
            "completedCount": self.completed_count(),
            "remainingCount": self.remaining_count(),
            "progress": self.progress_ratio(),
        }
