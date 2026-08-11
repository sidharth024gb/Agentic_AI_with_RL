from __future__ import annotations

from typing import Iterable, List, Optional


class ProcedureTracker:
    """
    Tracks an LLM-generated action procedure during one episode.

    Example procedure:

        [0, 1, 2, 3]

    Means:

        0 -> 1 -> 2 -> 3

    The tracker maintains which procedure actions have been
    successfully executed and provides a one-hot guidance vector
    for the PPO policy.

    The tracker does NOT determine the environment reward.
    It only determines whether an action follows the LLM-generated
    procedure and provides guidance information.
    """

    def __init__(
        self,
        procedure: Optional[Iterable[int]] = None,
        action_dim: int = 0,
    ):
        self.action_dim = action_dim
        self.procedure: List[int] = []

        # True means the corresponding procedure step was completed.
        self.executed: List[bool] = []

        self.reset(procedure)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        procedure: Optional[Iterable[int]] = None,
    ) -> None:
        """
        Reset the tracker for a new episode.

        All procedure execution flags are reset to False.
        """

        if procedure is not None:
            self.procedure = [int(action) for action in procedure]

        self.executed = [False for _ in self.procedure]

    # ------------------------------------------------------------------
    # Procedure
    # ------------------------------------------------------------------

    def set_procedure(
        self,
        procedure: Iterable[int],
    ) -> None:
        """
        Set a new LLM-generated procedure and reset progress.
        """

        self.procedure = [int(action) for action in procedure]

        self.executed = [False for _ in self.procedure]

    def get_procedure(self) -> List[int]:
        """
        Return a copy of the current procedure.
        """

        return self.procedure.copy()

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def get_next_action(self) -> Optional[int]:
        """
        Return the next expected action in the LLM procedure.

        Example:

            procedure = [0, 1, 2, 3]
            executed  = [True, False, False, False]

            returns 1
        """

        for index, completed in enumerate(self.executed):

            if not completed:
                return self.procedure[index]

        return None

    def get_next_index(self) -> Optional[int]:
        """
        Return the index of the next expected procedure action.
        """

        for index, completed in enumerate(self.executed):

            if not completed:
                return index

        return None

    def mark_completed(self, action: int) -> bool:
        """
        Mark the current expected action as completed.

        The action is only marked as completed if it is the
        next expected action in the procedure.

        Returns:
            True  -> procedure advanced
            False -> action was not the expected next action
        """

        next_action = self.get_next_action()

        if next_action is None:
            return False

        if int(action) != next_action:
            return False

        next_index = self.get_next_index()

        if next_index is None:
            return False

        self.executed[next_index] = True

        return True

    # ------------------------------------------------------------------
    # Guidance
    # ------------------------------------------------------------------

    def get_guidance(self) -> List[float]:
        """
        Return a one-hot guidance vector for PPO.

        Example:

            procedure = [0, 1, 2, 3]
            next action = 1

            guidance =
                [0, 1, 0, 0]

        If the procedure is complete, all values are zero.
        """

        guidance = [0.0 for _ in range(self.action_dim)]

        next_action = self.get_next_action()

        if next_action is None:
            return guidance

        if 0 <= next_action < self.action_dim:
            guidance[next_action] = 1.0

        return guidance

    # ------------------------------------------------------------------
    # Procedure checking
    # ------------------------------------------------------------------

    def is_expected_action(self, action: int) -> bool:
        """
        Check whether an action is the next action recommended
        by the LLM procedure.
        """

        next_action = self.get_next_action()

        if next_action is None:
            return False

        return int(action) == next_action

    def check_action(
        self,
        action: int,
        action_succeeded: bool,
    ) -> bool:
        """
        Check and optionally advance the procedure.

        The procedure advances only when:

            1. The selected action is the expected next action.
            2. The environment reports that the action succeeded.

        Returns:
            True  -> action followed the procedure
            False -> action did not follow the procedure
        """

        follows_procedure = self.is_expected_action(action)

        if follows_procedure and action_succeeded:
            self.mark_completed(action)

        return follows_procedure

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def is_complete(self) -> bool:
        """
        Return True when all procedure steps are completed.
        """

        if not self.executed:
            return False

        return all(self.executed)

    def completed_count(self) -> int:
        """
        Number of completed procedure steps.
        """

        return sum(self.executed)

    def remaining_count(self) -> int:
        """
        Number of remaining procedure steps.
        """

        return len(self.procedure) - self.completed_count()

    def get_execution_status(self) -> List[bool]:
        """
        Return a copy of the execution flags.
        """

        return self.executed.copy()

    # ------------------------------------------------------------------
    # Debugging / logging
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """
        Return the complete tracker status.
        """

        return {
            "procedure": self.get_procedure(),
            "executed": self.get_execution_status(),
            "nextAction": self.get_next_action(),
            "nextIndex": self.get_next_index(),
            "guidance": self.get_guidance(),
            "completed": self.is_complete(),
            "completedCount": self.completed_count(),
            "remainingCount": self.remaining_count(),
        }
