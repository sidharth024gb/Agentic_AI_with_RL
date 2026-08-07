"""
state_encoder.py

Converts backend environment state (JSON)
into numerical vectors for RL algorithms.
"""

import numpy as np


class StateEncoder:
    """
    Encodes Finance Backend state into RL observation vector.
    """

    def __init__(self):

        # Keep feature order fixed.
        # The neural network depends on this order.

        self.feature_names = [
            "total_invoices",
            "pending_invoices",
            "approved_invoices",
            "paid_invoices",
            "total_invoice_amount",
            "total_accounts",
            "available_balance",
            "total_suppliers",
            "active_suppliers",
            "average_supplier_risk",
        ]

    # ==========================================================
    # Main Encoder
    # ==========================================================

    def encode(self, state: dict):
        """
        Convert backend JSON state into numpy vector.

        Parameters
        ----------
        state:
            Raw backend state dictionary


        Returns
        -------
        np.ndarray
            RL observation vector
        """

        invoices = state.get("invoices", [])

        accounts = state.get("accounts", [])

        suppliers = state.get("suppliers", [])

        observation = [
            self._invoice_count(invoices),
            self._invoice_status_count(invoices, "PENDING"),
            self._invoice_status_count(invoices, "APPROVED"),
            self._invoice_status_count(invoices, "PAID"),
            self._total_invoice_amount(invoices),
            len(accounts),
            self._total_balance(accounts),
            len(suppliers),
            self._active_supplier_count(suppliers),
            self._average_supplier_risk(suppliers),
        ]

        return np.array(observation, dtype=np.float32)

    # ==========================================================
    # Invoice Features
    # ==========================================================

    def _invoice_count(self, invoices):

        return len(invoices)

    def _invoice_status_count(self, invoices, status):

        return sum(1 for invoice in invoices if invoice.get("status") == status)

    def _total_invoice_amount(self, invoices):

        return sum(invoice.get("amount", 0) for invoice in invoices)

    # ==========================================================
    # Account Features
    # ==========================================================

    def _total_balance(self, accounts):

        return sum(account.get("balance", 0) for account in accounts)

    # ==========================================================
    # Supplier Features
    # ==========================================================

    def _active_supplier_count(self, suppliers):

        return sum(1 for supplier in suppliers if supplier.get("active", False))

    def _average_supplier_risk(self, suppliers):

        if not suppliers:

            return 0

        total_risk = sum(supplier.get("riskScore", 0) for supplier in suppliers)

        return total_risk / len(suppliers)

    # ==========================================================
    # Utilities
    # ==========================================================

    def get_feature_names(self):

        return self.feature_names

    def observation_size(self):

        return len(self.feature_names)
