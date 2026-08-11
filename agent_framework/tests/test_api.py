"""
tests/test_api.py

Unit tests for the Finance API client.

These tests verify that APIClient:

    - builds the correct requests
    - uses the correct HTTP methods
    - sends the expected payloads
    - handles authentication
    - stores the JWT token
    - builds endpoint URLs correctly
    - exposes only agent-permitted operations

The real backend is NOT required.

HTTP requests are mocked so that this test file can run
independently of the backend server.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ==============================================================
# Project Root
# ==============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ==============================================================
# Project Imports
# ==============================================================

from environment.api_client import APIClient

# ==============================================================
# Mock Response
# ==============================================================


class MockResponse:
    """
    Minimal requests.Response replacement.
    """

    def __init__(
        self,
        payload,
        status_code=200,
    ):

        self._payload = payload
        self.status_code = status_code

    def json(self):

        return self._payload

    def raise_for_status(self):

        if self.status_code >= 400:

            raise Exception(f"HTTP {self.status_code}")


# ==============================================================
# Fixtures
# ==============================================================


@pytest.fixture
def client():

    api_client = APIClient()

    # Prevent the tests from making real HTTP requests.
    api_client.session = MagicMock()

    return api_client


@pytest.fixture
def successful_response():

    return {
        "success": True,
        "data": {},
        "message": "Success",
    }


# ==============================================================
# Authentication Tests
# ==============================================================


def test_login(
    client,
):
    """
    Verify login sends username/password to the login endpoint
    and stores the returned JWT token.
    """

    response = MockResponse(
        {
            "success": True,
            "data": {
                "token": "test-jwt-token",
            },
            "message": "Login successful",
        }
    )

    client.session.post.return_value = response

    result = client.login()

    assert result["success"] is True

    assert client.token == ("test-jwt-token")

    client.session.post.assert_called_once()


def test_login_updates_authorization_header(
    client,
):
    """
    Verify that a successful login updates the client headers
    with the JWT token.
    """

    response = MockResponse(
        {
            "success": True,
            "data": {
                "token": "abc123",
            },
        }
    )

    client.session.post.return_value = response

    client.login()

    headers = client.session.headers

    assert headers.get("Authorization") == "Bearer abc123"


# ==============================================================
# Profile
# ==============================================================


def test_get_profile(
    client,
):
    """
    Verify GET profile request.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "id": "agent-1",
            },
        }
    )

    result = client.get_profile()

    assert result["success"] is True

    client.session.get.assert_called_once()


# ==============================================================
# Sandbox
# ==============================================================


def test_reset_environment(
    client,
):
    """
    Verify sandbox reset endpoint.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.reset_environment()

    assert result["success"] is True

    client.session.post.assert_called_once()


def test_get_state(
    client,
):
    """
    Verify sandbox state endpoint.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "invoices": [],
            },
        }
    )

    result = client.get_state()

    assert result["success"] is True

    client.session.get.assert_called_once()


def test_get_reward(
    client,
):
    """
    Verify reward endpoint URL contains episode ID.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "reward": 10,
            },
        }
    )

    result = client.get_reward("episode-123")

    assert result["success"] is True

    called_url = client.session.get.call_args.args[0]

    assert "episode-123" in called_url


# ==============================================================
# Episode Tests
# ==============================================================


def test_get_episode(
    client,
):
    """
    Verify retrieving an episode.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "episodeNumber": 1,
            },
        }
    )

    result = client.get_episode("episode-1")

    assert result["success"] is True


def test_start_episode(
    client,
):
    """
    Verify episode creation/start request.
    """

    payload = {
        "agentType": "RL",
        "algorithm": "PPO",
        "goal": "Pay invoice",
    }

    client.session.post.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "episodeId": "episode-1",
            },
        }
    )

    result = client.start_episode(payload)

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == payload


def test_record_step(
    client,
):
    """
    Verify episode step recording.
    """

    payload = {
        "step": 1,
        "action": "get_invoice",
        "endpoint": "/invoices/1",
        "reward": 5,
        "success": True,
        "stateBefore": {},
        "stateAfter": {},
        "message": "Invoice found",
    }

    client.session.post.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.record_step(
        "episode-1",
        payload,
    )

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == payload


def test_end_episode(
    client,
):
    """
    Verify episode completion request.
    """

    payload = {
        "finalState": {},
        "completed": True,
        "terminatedReason": "GOAL_ACHIEVED",
    }

    client.session.post.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.end_episode(
        "episode-1",
        payload,
    )

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == payload


# ==============================================================
# Invoice Tests
# ==============================================================


def test_get_invoices(
    client,
):
    """
    Verify retrieving invoices.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": [],
        }
    )

    result = client.get_invoices()

    assert result["success"] is True


def test_get_invoice(
    client,
):
    """
    Verify retrieving a specific invoice.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "_id": "invoice-1",
            },
        }
    )

    result = client.get_invoice("invoice-1")

    assert result["success"] is True

    called_url = client.session.get.call_args.args[0]

    assert "invoice-1" in called_url


def test_invoice_duplicate_check(
    client,
):
    """
    Verify duplicate invoice checking.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "duplicate": False,
            },
        }
    )

    result = client.invoice_dupplicate_check(
        "supplier-1",
        1000,
        "2026-08-10",
    )

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == {
        "supplierId": "supplier-1",
        "amount": 1000,
        "dueDate": "2026-08-10",
    }


# ==============================================================
# Invoice Approval
# ==============================================================


def test_approve_invoice(
    client,
):
    """
    Verify invoice approval uses PATCH.
    """

    client.session.patch.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.approve_invoice("invoice-1")

    assert result["success"] is True

    client.session.patch.assert_called_once()

    called_kwargs = client.session.patch.call_args.kwargs

    assert called_kwargs["json"] == {
        "invoiceId": "invoice-1",
    }


# ==============================================================
# Payment Tests
# ==============================================================


def test_pay_invoice(
    client,
):
    """
    Verify invoice payment request.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.pay_invoice(
        "invoice-1",
        "account-1",
    )

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == {
        "invoiceId": "invoice-1",
        "accountId": "account-1",
    }


def test_cancel_payment(
    client,
):
    """
    Verify payment cancellation.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.cancel_payment("transaction-1")

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == {
        "transactionId": "transaction-1",
    }


def test_retry_payment(
    client,
):
    """
    Verify payment retry.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.retry_payment("invoice-1")

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == {
        "invoiceId": "invoice-1",
    }


# ==============================================================
# Supplier Tests
# ==============================================================


def test_get_suppliers(
    client,
):
    """
    Verify retrieving suppliers.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": [],
        }
    )

    result = client.get_suppliers()

    assert result["success"] is True


def test_validate_supplier(
    client,
):
    """
    Verify supplier validation.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
        }
    )

    result = client.validate_supplier("supplier-1")

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == {
        "supplierId": "supplier-1",
    }


# ==============================================================
# Account Tests
# ==============================================================


def test_get_accounts(
    client,
):
    """
    Verify retrieving accounts.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": [],
        }
    )

    result = client.get_accounts()

    assert result["success"] is True


def test_check_budget(
    client,
):
    """
    Verify budget checking request.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "approved": True,
            },
        }
    )

    result = client.check_budget(
        5000,
        "IT",
    )

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == {
        "amount": 5000,
        "department": "IT",
    }


def test_cash_position(
    client,
):
    """
    Verify cash position request.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": {
                "cash": 100000,
            },
        }
    )

    result = client.cash_position()

    assert result["success"] is True


# ==============================================================
# Report Tests
# ==============================================================


def test_get_transactions(
    client,
):
    """
    Verify transaction query parameters.
    """

    client.session.get.return_value = MockResponse(
        {
            "success": True,
            "data": [],
        }
    )

    result = client.get_transactions(
        "SUCCESS",
        "invoice-1",
    )

    assert result["success"] is True

    called_url = client.session.get.call_args.args[0]

    assert "status=SUCCESS" in called_url

    assert "invoice=invoice-1" in called_url


def test_generate_report(
    client,
):
    """
    Verify report generation.
    """

    client.session.post.return_value = MockResponse(
        {
            "success": True,
            "data": {},
        }
    )

    result = client.generate_report(
        "PAYMENT_REPORT",
        "2026-08-01",
        "2026-08-09",
    )

    assert result["success"] is True

    called_kwargs = client.session.post.call_args.kwargs

    assert called_kwargs["json"] == {
        "startDate": "2026-08-01",
        "endDate": "2026-08-09",
        "type": "PAYMENT_REPORT",
    }


# ==============================================================
# Agent Permission Tests
# ==============================================================


def test_agent_client_does_not_expose_blacklist(
    client,
):
    """
    Blacklisting suppliers is not an agent-permitted action.

    Therefore APIClient must not expose it.
    """

    assert not hasattr(
        client,
        "blacklist_supplier",
    )


def test_agent_client_does_not_expose_create_invoice(
    client,
):
    """
    Creating invoices is not an agent-permitted action.

    Therefore APIClient must not expose it.
    """

    assert not hasattr(
        client,
        "create_invoice",
    )


def test_agent_client_does_not_expose_randomize(
    client,
):
    """
    Randomization is part of reset_environment().

    There should be no separate randomize endpoint.
    """

    assert not hasattr(
        client,
        "randomize_environment",
    )

    assert not hasattr(
        client,
        "randomize",
    )


# ==============================================================
# Session Cleanup
# ==============================================================


def test_session_can_be_closed(
    client,
):
    """
    Verify the HTTP session can be closed.
    """

    client.session.close()

    client.session.close.assert_called_once()
