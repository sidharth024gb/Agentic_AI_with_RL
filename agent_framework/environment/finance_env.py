from datetime import datetime, timedelta, UTC

import numpy as np
import pandas as pd
import random

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None

from environment.action_space import ActionSpace
from environment.state_encoder import StateEncoder


class FinanceEnvironment:
    """
    Gymnasium-like Finance RL Environment.

    The backend is the simulated enterprise system.
    DataFrames represent the agent's working state inside the environment.
    """

    def __init__(self, api_client, max_steps=20):

        self.api_client = api_client

        self.max_steps = max_steps

        self.action_space_handler = ActionSpace()
        self.state_encoder = StateEncoder()

        self.current_step = 0

        self.all_invoices = pd.DataFrame()

        self.paid_invoices = pd.DataFrame()
        self.rejected_invoices = pd.DataFrame()
        self.pending_approval_invoices = pd.DataFrame()
        self.approved_invoices = pd.DataFrame()

        self.report_df = pd.DataFrame()

        self.state = self._initial_state()

    # ==========================================================
    # State
    # ==========================================================

    def _initial_state(self):
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

    def _update_invoice_states(self):

        self.state["has_paid_invoices"] = not self.paid_invoices.empty

        self.state["has_rejected_invoices"] = not self.rejected_invoices.empty

        self.state["has_pending_approval_invoices"] = (
            not self.pending_approval_invoices.empty
        )

        self.state["has_approved_invoices"] = not self.approved_invoices.empty

    def get_state(self):

        self._update_invoice_states()

        return self.state.copy()

    def get_observation(self):

        return self.state_encoder.encode(self.get_state())

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self, seed=None, options=None):

        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0

        self.all_invoices = pd.DataFrame()

        self.paid_invoices = pd.DataFrame()
        self.rejected_invoices = pd.DataFrame()
        self.pending_approval_invoices = pd.DataFrame()
        self.approved_invoices = pd.DataFrame()

        self.report_df = pd.DataFrame()

        self.state = self._initial_state()

        # Backend environment reset
        response = self.api_client.reset_environment()

        if response.get("environmentError"):
            raise RuntimeError(
                response.get("message", "Failed to reset backend environment.")
            )

        return self.get_observation()

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, action):

        if self.current_step >= self.max_steps:
            return (
                self.get_observation(),
                0.0,
                True,
                {"reason": "MAX_STEPS_REACHED"},
            )

        action_name = self.action_space_handler.get_action_name(action)

        self.current_step += 1

        try:

            result = self.action_space_handler.execute(self, action)

        except Exception as exc:

            return (
                self.get_observation(),
                0.0,
                True,
                {
                    "environmentError": True,
                    "error": str(exc),
                    "action": action_name,
                },
            )

        reward = result.get("reward", 0)

        if reward is None:
            reward = 0

        done = self.state["task_completed"] or self.current_step >= self.max_steps

        info = {
            "action": action_name,
            "step": self.current_step,
            "success": result.get("success", True),
            "environmentError": result.get("environmentError", False),
            "message": result.get("message"),
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

    def get_invoices(self):

        response = self.api_client.get_invoices()

        if response.get("environmentError"):
            return response

        invoices = response.get("invoices", response.get("data", []))

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
                raise ValueError("Invoice response does not contain a status column.")

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

        return {
            "success": True,
            "reward": response.get("reward", 5),
            "message": "Invoices loaded.",
        }

    # ==========================================================
    # ACTION 1
    # CHECK DUPLICATE
    # ==========================================================

    def check_duplicate(self):

        frames = []

        if not self.pending_approval_invoices.empty:
            frames.append(("pending_approval", self.pending_approval_invoices.copy()))

        if not self.approved_invoices.empty:
            frames.append(("approved", self.approved_invoices.copy()))

        for frame_name, frame in frames:

            if frame.empty:
                continue

            rows_to_remove = []

            for index, invoice in frame.iterrows():

                supplier_id = self._get_value(
                    invoice, ["supplierId", "supplier", "supplier_id"]
                )

                amount = self._get_value(invoice, ["amount", "totalAmount"])

                due_date = self._get_value(invoice, ["dueDate", "due_date"])

                if supplier_id is None or amount is None:
                    continue

                response = self.api_client.invoice_dupplicate_check(
                    supplier_id, amount, due_date
                )

                if response.get("environmentError"):
                    return response

                duplicate = bool(response.get("duplicate", False))

                if duplicate:

                    rows_to_remove.append(index)

            if frame_name == "pending_approval":

                self.pending_approval_invoices = frame.drop(rows_to_remove)

            elif frame_name == "approved":

                self.approved_invoices = frame.drop(rows_to_remove)

        self.state["check_duplicate"] = True

        self._update_invoice_states()

        return {
            "success": True,
            "reward": 10,
            "message": "Duplicate invoices checked.",
        }

    # ==========================================================
    # ACTION 2
    # CHECK SUPPLIER
    # ==========================================================

    def check_supplier(self):

        frames = []

        if not self.pending_approval_invoices.empty:
            frames.append(("pending_approval", self.pending_approval_invoices.copy()))

        if not self.approved_invoices.empty:
            frames.append(("approved", self.approved_invoices.copy()))

        for frame_name, frame in frames:

            rows_to_remove = []

            for index, invoice in frame.iterrows():

                supplier_id = self._get_value(
                    invoice, ["supplierId", "supplier", "supplier_id"]
                )

                if supplier_id is None:
                    rows_to_remove.append(index)
                    continue

                response = self.api_client.validate_supplier(supplier_id)

                if response.get("environmentError"):
                    return response

                valid = response.get(
                    "valid", response.get("isValid", response.get("success", False))
                )

                if not valid:
                    rows_to_remove.append(index)

            if frame_name == "pending_approval":

                self.pending_approval_invoices = frame.drop(rows_to_remove)

            elif frame_name == "approved":

                self.approved_invoices = frame.drop(rows_to_remove)

        self.state["check_supplier"] = True

        self._update_invoice_states()

        return {
            "success": True,
            "reward": 10,
            "message": "Suppliers checked.",
        }

    # ==========================================================
    # ACTION 3
    # APPROVE INVOICES
    # ==========================================================

    def approve_invoices(self):

        if self.pending_approval_invoices.empty:

            self.state["approve_invoices"] = True

            return {
                "success": True,
                "reward": 0,
                "message": "No pending invoices to approve.",
            }

        approved_rows = []

        for _, invoice in self.pending_approval_invoices.iterrows():

            invoice_id = self._get_value(invoice, ["_id", "id", "invoiceId"])

            if invoice_id is None:
                continue

            response = self.api_client.approve_invoice(invoice_id)

            if response.get("environmentError"):
                return response

            if response.get("success", False):

                approved_rows.append(invoice)

        if approved_rows:

            approved_df = pd.DataFrame(approved_rows)

            self.approved_invoices = pd.concat(
                [self.approved_invoices, approved_df], ignore_index=True
            )

        self.pending_approval_invoices = pd.DataFrame()

        self.state["approve_invoices"] = True

        self._update_invoice_states()

        return {
            "success": True,
            "reward": 10,
            "message": "Invoices approved.",
        }

    # ==========================================================
    # ACTION 4
    # PAY INVOICES
    # ==========================================================

    def pay_invoices(self):

        if self.approved_invoices.empty:

            self.state["pay_invoices"] = True

            return {
                "success": True,
                "reward": 0,
                "message": "No approved invoices to pay.",
            }

        paid_rows = []

        for _, invoice in self.approved_invoices.iterrows():

            invoice_id = self._get_value(invoice, ["_id", "id", "invoiceId"])

            if invoice_id is None:
                continue

            account_id = self._get_value(
                invoice, ["accountId", "account", "account_id"]
            )

            if account_id is None:

                accounts_response = self.api_client.get_accounts()

                if accounts_response.get("environmentError"):
                    return accounts_response

                accounts = accounts_response.get(
                    "accounts", accounts_response.get("data", [])
                )

                if not accounts:
                    continue

                account_id = self._get_value(
                    pd.Series(accounts[0]), ["_id", "id", "accountId"]
                )

            response = self.api_client.pay_invoice(invoice_id, account_id)

            if response.get("environmentError"):
                return response

            if response.get("success", False):

                paid_rows.append(invoice)

        if paid_rows:

            paid_df = pd.DataFrame(paid_rows)

            self.paid_invoices = pd.concat(
                [self.paid_invoices, paid_df], ignore_index=True
            )

        self.approved_invoices = pd.DataFrame()

        self.state["pay_invoices"] = True

        self._update_invoice_states()

        return {
            "success": True,
            "reward": 30,
            "message": "Approved invoices processed for payment.",
        }

    # ==========================================================
    # ACTION 5
    # CHECK BUDGET
    # ==========================================================


    def check_budget(self):

        if self.approved_invoices.empty:

            self.state["check_budget"] = True

            return {
                "success": True,
                "reward": 0,
                "message": "No approved invoices to check.",
            }

        # ----------------------------------------------------------
        # Identify department and amount columns
        # ----------------------------------------------------------

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

            return {
                "success": False,
                "reward": 0,
                "message": ("No department/category found " "in approved invoices."),
            }

        if amount_column is None:

            return {
                "success": False,
                "reward": 0,
                "message": ("No amount field found " "in approved invoices."),
            }

        # ----------------------------------------------------------
        # Find unique departments
        # ----------------------------------------------------------

        departments = self.approved_invoices[department_column].dropna().unique().tolist()

        valid_invoice_indices = []

        budget_results = {}

        # ----------------------------------------------------------
        # Check each department separately
        # ----------------------------------------------------------

        for department in departments:

            department_invoices = self.approved_invoices[
                self.approved_invoices[department_column] == department
            ]

            # ------------------------------------------------------
            # Get the remaining budget for this department.
            #
            # The amount sent to the API is only used to perform
            # the budget lookup. The returned remainingBudget is
            # what we use for filtering.
            # ------------------------------------------------------

            random_amount = round(
                random.uniform(1.0, 100.0),
                2,
            )

            response = self.api_client.check_budget(
                random_amount,
                department,
            )

            # ------------------------------------------------------
            # Environment/backend error
            # ------------------------------------------------------

            if response.get("environment_error"):

                return {
                    "success": False,
                    "reward": 0,
                    "environment_error": True,
                    "message": response.get(
                        "message",
                        "Environment error during budget check.",
                    ),
                }

            # ------------------------------------------------------
            # Actual backend response is inside "data"
            # ------------------------------------------------------

            data = response.get("data", {})

            if not isinstance(data, dict):

                return {
                    "success": False,
                    "reward": 0,
                    "message": (
                        f"Invalid budget response for " f"department {department}."
                    ),
                }

            budget = data.get("budget")

            if not isinstance(budget, dict):

                return {
                    "success": False,
                    "reward": 0,
                    "message": (
                        f"Budget information missing for " f"department {department}."
                    ),
                }

            remaining_budget = budget.get("remainingBudget")

            if remaining_budget is None:

                return {
                    "success": False,
                    "reward": 0,
                    "message": (
                        f"Remaining budget missing for " f"department {department}."
                    ),
                }

            remaining_budget = float(remaining_budget)

            # ------------------------------------------------------
            # Keep invoices individually while staying within the
            # department's remaining budget.
            # ------------------------------------------------------

            current_total = 0.0

            department_kept = []
            department_removed = []

            for index, invoice in department_invoices.iterrows():

                amount = pd.to_numeric(
                    invoice[amount_column],
                    errors="coerce",
                )

                if pd.isna(amount):

                    department_removed.append(index)

                    continue

                amount = float(amount)

                # --------------------------------------------------
                # Keep invoice if adding it does not exceed budget.
                # --------------------------------------------------

                if current_total + amount <= remaining_budget:

                    valid_invoice_indices.append(index)

                    department_kept.append(
                        {
                            "index": index,
                            "amount": amount,
                        }
                    )

                    current_total += amount

                else:

                    department_removed.append(
                        {
                            "index": index,
                            "amount": amount,
                        }
                    )

            # ------------------------------------------------------
            # Store budget information for logging/debugging.
            # ------------------------------------------------------

            budget_results[department] = {
                "remaining_budget": remaining_budget,
                "kept_total": current_total,
                "kept_invoices": department_kept,
                "removed_invoices": department_removed,
            }

        # ----------------------------------------------------------
        # Keep only invoices that fit within their department
        # budget.
        # ----------------------------------------------------------

        self.approved_invoices = self.approved_invoices.loc[valid_invoice_indices].copy()

        # ----------------------------------------------------------
        # Reset dataframe index after filtering.
        # ----------------------------------------------------------

        self.approved_invoices.reset_index(
            drop=True,
            inplace=True,
        )

        # ----------------------------------------------------------
        # Update state
        # ----------------------------------------------------------

        self.state["check_budget"] = True

        self._update_invoice_states()

        return {
            "success": True,
            "reward": 10,
            "message": "Budget checked.",
            "budget_results": budget_results,
        }

    # ==========================================================
    # ACTION 6
    # GENERATE REPORT
    # ==========================================================

    def generate_report(self):

        end_date = datetime.now(UTC)
        start_date = end_date - timedelta(days=365)

        response = self.api_client.generate_report(
            type="transaction_summary",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
        )

        if response.get("environmentError"):
            return response

        report = response.get("report", response.get("data", []))

        if isinstance(report, dict):
            report = [report]

        self.report_df = pd.DataFrame(report)

        self.state["generate_report"] = True

        return {
            "success": True,
            "reward": 5,
            "message": "One-year transaction report generated.",
        }

    # ==========================================================
    # ACTION 7
    # CHECK PAYMENT COMPLETED
    # ==========================================================

    def check_payment_completed(self):

        response = self.api_client.get_invoices()

        if response.get("environmentError"):
            return response

        invoices = response.get("invoices", response.get("data", []))

        current_invoices = pd.DataFrame(invoices)

        if current_invoices.empty:

            self.approved_invoices = pd.DataFrame()

            self.state["check_payment_completed"] = True
            self.state["task_completed"] = True

            return {
                "success": True,
                "reward": 10,
                "message": "Payment task completed.",
            }

        status_column = self._find_column(current_invoices, ["status", "invoiceStatus"])

        if status_column is None:

            return {
                "success": False,
                "reward": 0,
                "message": "Invoice status unavailable.",
            }

        approved = current_invoices[
            current_invoices[status_column] == "APPROVED"
        ].copy()

        self.approved_invoices = approved

        self.state["check_payment_completed"] = True

        # No approved invoices means all invoices
        # requiring payment have been paid.
        if approved.empty:

            self.state["task_completed"] = True

            self._update_invoice_states()

            return {
                "success": True,
                "reward": 10,
                "message": "All valid invoices have been paid.",
            }

        # There are still approved invoices.
        # Re-check their budgets.
        valid_rows = []

        for _, invoice in approved.iterrows():

            amount = self._get_value(invoice, ["amount", "totalAmount"])

            department = self._get_value(
                invoice, ["department", "category", "function"]
            )

            if amount is None or department is None:
                continue

            budget_response = self.api_client.check_budget(amount, department)

            if budget_response.get("environmentError"):
                return budget_response

            available = budget_response.get(
                "available",
                budget_response.get(
                    "withinBudget", budget_response.get("success", False)
                ),
            )

            if available:
                valid_rows.append(invoice)

        # Valid approved invoices still remain.
        # Therefore the task is not complete.
        self.approved_invoices = pd.DataFrame(valid_rows)

        self.state["task_completed"] = self.approved_invoices.empty

        self._update_invoice_states()

        return {
            "success": True,
            "reward": 10 if self.state["task_completed"] else 0,
            "message": (
                "Payment task completed."
                if self.state["task_completed"]
                else "Approved invoices still require payment."
            ),
        }

    # ==========================================================
    # HELPERS
    # ==========================================================

    @staticmethod
    def _find_column(df, candidates):

        for column in candidates:

            if column in df.columns:
                return column

        return None

    @staticmethod
    def _get_value(row, candidates):

        for column in candidates:

            if column in row.index:

                value = row[column]

                if pd.notna(value):
                    return value

        return None
