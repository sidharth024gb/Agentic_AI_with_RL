"""
action_space.py

State-aware action execution for the Finance RL Environment.
"""

from enum import IntEnum

from environment.api_client import APIClient


class Action(IntEnum):

    GET_INVOICES = 0

    VALIDATE_SUPPLIER = 1

    CHECK_BUDGET = 2

    APPROVE_INVOICE = 3

    PAY_INVOICE = 4

    GENERATE_REPORT = 5


class ActionSpace:

    def __init__(self, client: APIClient):

        self.client = client

    def size(self):

        return len(Action)

    def execute(self, action: int, state: dict):

        action = Action(action)

        invoices = state.get("invoices", [])

        suppliers = state.get("suppliers", [])

        accounts = state.get("accounts", [])

        # -------------------------------------------------------
        # Get Invoices
        # -------------------------------------------------------

        if action == Action.GET_INVOICES:

            return self.client.get_invoices()

        # -------------------------------------------------------
        # Validate Supplier
        # -------------------------------------------------------

        elif action == Action.VALIDATE_SUPPLIER:

            invoice = next(
                (invoice for invoice in invoices if invoice["status"] == "PENDING"),
                None,
            )

            if invoice is None:

                return {"success": False, "message": "No pending invoice found."}

            supplier = invoice["supplier"]

            supplier_id = supplier["_id"] if isinstance(supplier, dict) else supplier

            return self.client.validate_supplier(supplier_id)

        # -------------------------------------------------------
        # Budget Check
        # -------------------------------------------------------

        elif action == Action.CHECK_BUDGET:

            invoice = next(
                (invoice for invoice in invoices if invoice["status"] == "PENDING"),
                None,
            )

            if invoice is None:

                return {"success": False, "message": "No pending invoice found."}

            return self.client.check_budget(
                amount=invoice["amount"], department=invoice["department"]
            )

        # -------------------------------------------------------
        # Approve
        # -------------------------------------------------------

        elif action == Action.APPROVE_INVOICE:

            invoice = next(
                (invoice for invoice in invoices if invoice["status"] == "PENDING"),
                None,
            )

            if invoice is None:

                return {"success": False, "message": "No invoice available."}

            return self.client.approve_invoice(invoice["_id"])

        # -------------------------------------------------------
        # Pay
        # -------------------------------------------------------

        elif action == Action.PAY_INVOICE:

            invoice = next(
                (invoice for invoice in invoices if invoice["status"] == "APPROVED"),
                None,
            )

            if invoice is None:

                return {"success": False, "message": "No approved invoice."}

            account = max(accounts, key=lambda x: x["balance"], default=None)

            if account is None:

                return {"success": False, "message": "No account available."}

            return self.client.pay_invoice(
                invoice_id=invoice["_id"], account_id=account["_id"]
            )

        # -------------------------------------------------------
        # Report
        # -------------------------------------------------------

        elif action == Action.GENERATE_REPORT:

            return self.client.generate_report(type="SUMMARY")

        raise ValueError(f"Unknown action {action}")
