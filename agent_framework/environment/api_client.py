"""
api_client.py

Centralized API client for communicating with the
Finance RL Backend.

All HTTP communication between the RL agent and the
backend should go through this class.
"""

from typing import (
    Any,
    Dict,
    Optional,
)

from urllib.parse import urlencode

import requests

from config.config import config


class APIClient:
    """
    Wrapper around the Finance Backend REST API.
    """

    def __init__(self):

        self.base_url = config.backend.BASE_URL.rstrip("/")

        self.timeout = config.backend.TIMEOUT

        self.session = requests.Session()

        self.token: Optional[str] = None

        self._update_headers()

    # ==========================================================
    # Internal Helpers
    # ==========================================================

    def _update_headers(
        self,
    ):

        headers = {
            "Content-Type": "application/json",
        }

        if self.token:

            headers["Authorization"] = f"Bearer {self.token}"

        self.session.headers.update(headers)

    # ==========================================================
    # Request
    # ==========================================================

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs,
    ) -> Dict[str, Any]:

        url = f"{self.base_url}" f"{endpoint}"

        try:

            response = self.session.request(
                method=method,
                url=url,
                timeout=self.timeout,
                **kwargs,
            )

            # ==================================================
            # Parse JSON
            # ==================================================

            try:

                data = response.json()

            except Exception:

                data = {}

            # ==================================================
            # Environment Error
            #
            # Infrastructure failures can be signalled through:
            #
            # 1. HTTP >= 500
            # 2. backend environmentError flag
            # ==================================================

            backend_environment_error = False

            if isinstance(
                data,
                dict,
            ):

                backend_environment_error = bool(
                    data.get(
                        "environmentError",
                        data.get(
                            "environment_error",
                            False,
                        ),
                    )
                )

            environment_error = response.status_code >= 500 or backend_environment_error

            return {
                # HTTP request itself succeeded.
                #
                # FinanceEnvironment additionally inspects the
                # payload's own success field for business
                # outcomes.
                "success": bool(response.ok),
                "endpoint": endpoint,
                "status_code": response.status_code,
                "data": data,
                "environment_error": environment_error,
            }

        except requests.exceptions.RequestException as exc:

            return {
                "success": False,
                "status_code": None,
                "endpoint": endpoint,
                "data": {},
                "environment_error": True,
                "message": str(exc),
            }

    # ==========================================================
    # HTTP Methods
    # ==========================================================

    def _get(
        self,
        endpoint,
        **kwargs,
    ):

        return self._request(
            "GET",
            endpoint,
            **kwargs,
        )

    def _post(
        self,
        endpoint,
        **kwargs,
    ):

        return self._request(
            "POST",
            endpoint,
            **kwargs,
        )

    def _patch(
        self,
        endpoint,
        **kwargs,
    ):

        return self._request(
            "PATCH",
            endpoint,
            **kwargs,
        )

    def _delete(
        self,
        endpoint,
        **kwargs,
    ):

        return self._request(
            "DELETE",
            endpoint,
            **kwargs,
        )

    # ==========================================================
    # Authentication
    # ==========================================================

    def login(
        self,
    ):

        payload = {
            "email": config.backend.EMAIL,
            "password": config.backend.PASSWORD,
        }

        response = self._post(
            config.backend.LOGIN_ENDPOINT,
            json=payload,
        )

        if response["success"]:

            data = response.get(
                "data",
                {},
            )

            token = (
                data.get("token")
                if isinstance(
                    data,
                    dict,
                )
                else None
            )

            if token:

                self.token = token

                self._update_headers()

        return response

    def get_profile(
        self,
    ):

        return self._get(config.backend.PROFILE_ENDPOINT)

    # ==========================================================
    # Sandbox
    # ==========================================================

    def reset_environment(
        self,
        seed=None,
    ):
        """
        Reset the backend sandbox using an optional deterministic
        episode seed.

        FinanceEnvironment supplies the per-episode seed so that
        repeated experiments with the same base seed receive the
        same sequence of backend scenarios.
        """

        payload = {}

        if seed is not None:
            payload["seed"] = int(seed)

        return self._post(
            config.backend.SANDBOX_RESET,
            json=payload,
        )

    def get_state(
        self,
    ):

        return self._get(config.backend.SANDBOX_STATE)

    def get_reward(
        self,
        episode_id,
    ):

        return self._get(f"{config.backend.SANDBOX_REWARD}/" f"{episode_id}")

    # ==========================================================
    # Episodes
    # ==========================================================

    def get_episode(
        self,
        episode_id,
    ):

        return self._get(f"{config.backend.EPISODE}/" f"{episode_id}")

    def start_episode(
        self,
        payload,
    ):

        return self._post(
            config.backend.EPISODE_START,
            json=payload,
        )

    def record_step(
        self,
        episode_id,
        payload,
    ):

        return self._post(
            (
                f"{config.backend.EPISODE}/"
                f"{episode_id}"
                f"{config.backend.EPISODE_STEP}"
            ),
            json=payload,
        )

    def end_episode(
        self,
        episode_id,
        payload,
    ):

        return self._post(
            (
                f"{config.backend.EPISODE}/"
                f"{episode_id}"
                f"{config.backend.EPISODE_END}"
            ),
            json=payload,
        )

    def get_episodes(
        self,
        experiment_name=None,
        phase=None,
        agent_type=None,
        algorithm=None,
    ):

        query = {}

        if experiment_name:

            query["experimentName"] = experiment_name

        if phase:

            query["phase"] = phase

        if agent_type:

            query["agentType"] = agent_type

        if algorithm:

            query["algorithm"] = algorithm

        endpoint = config.backend.EPISODE

        if query:

            endpoint = f"{endpoint}?" f"{urlencode(query)}"

        return self._get(endpoint)

    # ==========================================================
    # Invoice
    # ==========================================================

    def get_invoices(
        self,
    ):

        return self._get(config.backend.INVOICE)

    def get_invoice(
        self,
        invoice_id,
    ):

        return self._get((f"{config.backend.INVOICE}/" f"{invoice_id}"))

    # ==========================================================
    # Duplicate Check
    # ==========================================================

    def check_invoice_duplicate(
        self,
        invoice_id,
        supplier_id,
        amount,
        due_date,
    ):
        """
        Check whether an existing invoice is a duplicate.

        invoice_id is included so the backend can exclude
        the current invoice from its duplicate query.
        """

        return self._post(
            config.backend.INVOICE_DUPLICATE_CHECK,
            json={
                "invoiceId": invoice_id,
                "supplierId": supplier_id,
                "amount": amount,
                "dueDate": due_date,
            },
        )

    # ==========================================================
    # Backwards Compatibility
    #
    # Remove this after finance_env.py has been updated.
    # ==========================================================

    def invoice_dupplicate_check(
        self,
        supplier_id,
        amount,
        due_date,
        invoice_id=None,
    ):
        """
        Deprecated compatibility wrapper.

        New code should use:

            check_invoice_duplicate(...)
        """

        payload = {
            "supplierId": supplier_id,
            "amount": amount,
            "dueDate": due_date,
        }

        if invoice_id is not None:

            payload["invoiceId"] = invoice_id

        return self._post(
            config.backend.INVOICE_DUPLICATE_CHECK,
            json=payload,
        )

    # ==========================================================
    # Invoice Approval
    # ==========================================================

    def approve_invoice(
        self,
        invoice_id,
    ):

        return self._patch(
            config.backend.APPROVAL_APPROVE,
            json={"invoiceId": invoice_id},
        )

    # ==========================================================
    # Payment
    # ==========================================================

    def pay_invoice(
        self,
        invoice_id,
        account_id,
    ):

        return self._post(
            config.backend.PAYMENT_PAY,
            json={
                "invoiceId": invoice_id,
                "accountId": account_id,
            },
        )

    def cancel_payment(
        self,
        transaction_id,
    ):

        return self._post(
            config.backend.PAYMENT_CANCEL_PAYMENT,
            json={"transactionId": transaction_id},
        )

    def retry_payment(
        self,
        invoice_id,
    ):

        return self._post(
            config.backend.PAYMENT_RETRY_PAYMENT,
            json={"invoiceId": invoice_id},
        )

    # ==========================================================
    # Supplier
    # ==========================================================

    def get_suppliers(
        self,
    ):

        return self._get(config.backend.SUPPLIER)

    def validate_supplier(
        self,
        supplier_id,
    ):

        return self._post(
            config.backend.SUPPLIER_VALIDATE,
            json={"supplierId": supplier_id},
        )

    # ==========================================================
    # Accounts
    # ==========================================================

    def get_accounts(
        self,
    ):

        return self._get(config.backend.ACCOUNT)

    def check_budget(
        self,
        amount=None,
        department=None,
    ):
        """
        Retrieve/check budget state.

        amount is optional.

        When supplied, the backend additionally reports whether
        that amount fits inside the remaining budget.
        """

        payload = {"department": department}

        if amount is not None:

            payload["amount"] = amount

        return self._post(
            config.backend.ACCOUNT_BUDGET_CHECK,
            json=payload,
        )

    def cash_position(
        self,
    ):

        return self._get(config.backend.ACCOUNT_CASH_POSITION)

    # ==========================================================
    # Reports
    # ==========================================================

    def get_transactions(
        self,
        status=None,
        invoice=None,
    ):

        query = {}

        if status is not None:

            query["status"] = status

        if invoice is not None:

            query["invoice"] = invoice

        endpoint = config.backend.REPORT_TRANSACTIONS

        if query:

            endpoint = f"{endpoint}?" f"{urlencode(query)}"

        return self._get(endpoint)

    def generate_report(
        self,
        type,
        start_date=None,
        end_date=None,
    ):

        return self._post(
            config.backend.REPORT_GENERATE_REPORT,
            json={
                "startDate": start_date,
                "endDate": end_date,
                "type": type,
            },
        )

    # ==========================================================
    # Close
    # ==========================================================

    def close(
        self,
    ):

        self.session.close()
