import json
import os
import time
from datetime import datetime
import dotenv
from openpyxl import Workbook, load_workbook
import requests

dotenv.load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

BASE_URL = f"http://localhost:{os.getenv('PORT', 5000)}"

HEADERS = {"Content-Type": "application/json"}


# ============================================================
# TEST USER
# ============================================================

TEST_USER = {
    "username": "RL_Test_Agent",
    "email": "rl_agent_test@sandbox.com",
    "password": "Password123!",
    "role": "AGENT_BOT",
}


# ============================================================
# GLOBAL TEST STATE
# ============================================================

TOKEN = None

EPISODE_ID = None

INVOICE_ID = None

RETRY_INVOICE_ID = None

UPDATE_INVOICE_ID = None

ARCHIVE_INVOICE_ID = None

REJECT_INVOICE_ID = None

SUPPLIER_ID = None

BLACKLIST_SUPPLIER_ID = None

TRANSACTION_ID = None

CANCEL_TRANSACTION_ID = None

ACCOUNT_ID = None

TO_ACCOUNT_ID = None

STATS = {
    "total_actions": 0,
    "successful_actions": 0,
    "failed_actions": 0,
    "environment_errors": 0,
    "total_reward": 0,
}

# ============================================================
# LOG FILE CONFIGURATION
# ============================================================

LOG_FILE = f"results/logs/test/rl_environment_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"


def init_log():
    os.makedirs("logs", exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Endpoint Tests"
    sheet.append(
        [
            "Timestamp",
            "Phase",
            "Endpoint",
            "Method",
            "Status Code",
            "Success",
            "Reward",
            "Execution Time(ms)",
            "Response",
            "Error",
        ]
    )
    workbook.save(LOG_FILE)


def write_log(endpoint, method, response, execution_time=0.0, phase="API Execution"):
    if not os.path.exists(LOG_FILE):
        init_log()

    workbook = load_workbook(LOG_FILE)
    sheet = workbook["Endpoint Tests"]

    try:
        response_json = response.json()
    except Exception:
        response_json = {"raw_response": getattr(response, "text", str(response))}

    status_code = getattr(response, "status_code", "N/A")
    success = (
        response_json.get("success", False)
        if isinstance(response_json, dict)
        else False
    )
    reward = (
        response_json.get("reward", None) if isinstance(response_json, dict) else None
    )
    message = (
        response_json.get("message", "") if isinstance(response_json, dict) else ""
    )

    sheet.append(
        [
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            phase,
            endpoint,
            method,
            status_code,
            success,
            reward,
            round(execution_time, 2),
            (
                json.dumps(response_json, indent=2)
                if isinstance(response_json, dict)
                else str(response_json)
            ),
            message,
        ]
    )
    workbook.save(LOG_FILE)


# ============================================================
# LOGGING
# ============================================================


def print_header(title):
    # Suppressed for clean summarized terminal output
    pass


def print_result(name, success, reward=None):
    # Individual endpoint results are silent in terminal; captured in Excel logs
    pass


# ============================================================
# REWARD TRACKING
# ============================================================


def update_statistics(response):
    """
    Updates RL evaluation metrics.

    Agent mistakes:
        counted as failed actions

    Environment errors:
        not counted as failures
    """
    try:
        data = response.json()
    except Exception:
        STATS["environment_errors"] += 1
        return

    if data.get("environmentError"):
        STATS["environment_errors"] += 1
        return

    reward = data.get("reward")
    if reward is not None:
        STATS["total_reward"] += reward

    STATS["total_actions"] += 1

    if data.get("success"):
        STATS["successful_actions"] += 1
    else:
        STATS["failed_actions"] += 1


# ============================================================
# HTTP REQUEST WRAPPER
# ============================================================


def api_request(method, endpoint, body=None, auth=False):
    """
    Common API caller.

    Automatically:
    - Adds JWT token
    - Tracks rewards
    - Handles server errors
    - Logs detailed results into Excel
    """
    headers = HEADERS.copy()

    if auth and TOKEN:
        headers["Authorization"] = f"Bearer {TOKEN}"

    url = BASE_URL + endpoint
    start_time = time.time()

    try:
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=body)
        elif method == "PATCH":
            response = requests.patch(url, headers=headers, json=body)
        else:
            raise Exception("Unsupported HTTP method")

        execution_time = (time.time() - start_time) * 1000
        update_statistics(response)

        # Write detailed request & response data to Excel
        write_log(endpoint, method, response, execution_time)

        return response

    except requests.exceptions.ConnectionError:
        STATS["environment_errors"] += 1
        return None


# ============================================================
# 1. HEALTH CHECK
# ============================================================


def test_health():
    print_header("SERVER HEALTH CHECK")
    response = api_request("GET", "/")

    if response is None:
        print_result("Server availability", False)
        return False

    data = response.json()
    success = response.status_code == 200 and data.get("success") is True
    print_result("GET /", success)
    return success


# ============================================================
# 2. AUTHENTICATION TESTING
# ============================================================


def test_register():
    print_header("AUTHENTICATION")
    response = api_request("POST", "/api/auth/register", TEST_USER)

    if response is None:
        return False

    data = response.json()
    success = response.status_code in [200, 201] and data.get("success") is True
    print_result("Register RL Agent", success)
    return success


def test_login():
    global TOKEN
    response = api_request(
        "POST",
        "/api/auth/login",
        {"email": TEST_USER["email"], "password": TEST_USER["password"]},
    )

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and "token" in data

    if success:
        TOKEN = data["token"]

    print_result("Login Agent", success)
    return success


def test_profile():
    response = api_request("GET", "/api/auth/me", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("GET Agent Profile", success)
    return success


def test_authentication():
    register = test_register()
    login = test_login()
    profile = False

    if login:
        profile = test_profile()

    return register and login and profile


# ============================================================
# 3. SANDBOX ENVIRONMENT TESTING
# ============================================================


def test_environment_reset():
    print_header("SANDBOX ENVIRONMENT")
    response = api_request("POST", "/api/sandbox/reset", {}, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and data.get("reward") is not None
    print_result("Reset Environment", success, data.get("reward"))
    return success


def test_environment_state():
    response = api_request("GET", "/api/sandbox/state", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and "observation" in data
    print_result("Get Environment State", success, data.get("reward"))
    return success


def test_environment_reward():
    global EPISODE_ID

    response = api_request("GET", f"/api/sandbox/reward/{EPISODE_ID}", auth=True)

    if response is None:
        return False

    data = response.json()
    success = "reward" in data
    print_result("Get Current Reward", success, data.get("reward"))
    return success


def test_sandbox():
    reset = test_environment_reset()
    state = test_environment_state()
    reward = test_environment_reward()
    return reset and state and reward


# ============================================================
# 4. INVOICE WORKFLOW TESTING
# ============================================================


def test_create_invoice():
    global SUPPLIER_ID
    invoice_data = {
        "supplierId": SUPPLIER_ID,
        "amount": 2000,
        "category": "SOFTWARE",
        "description": "Cloud subscription invoice",
        "dueDate": "2026-12-31",
    }

    response = api_request("POST", "/api/invoice", invoice_data, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Create Invoice", success, data.get("reward"))
    return success


def test_get_invoices():
    global INVOICE_ID, RETRY_INVOICE_ID, UPDATE_INVOICE_ID, ARCHIVE_INVOICE_ID, REJECT_INVOICE_ID
    response = api_request("GET", "/api/invoice", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and isinstance(data.get("invoices"), list)

    if success:
        invoices = data.get("invoices")

        if isinstance(invoices, list):
            # 1. Filter out only the invoices that match your status condition
            pending_invoices = [
                inv for inv in invoices if inv.get("status") == "PENDING_APPROVAL"
            ]

            # 2. Safely extract up to 4 IDs (defaults to None if there are fewer than 4 matches)
            INVOICE_ID = (
                pending_invoices[0]["_id"] if len(pending_invoices) > 0 else None
            )
            RETRY_INVOICE_ID = (
                pending_invoices[1]["_id"] if len(pending_invoices) > 1 else None
            )
            UPDATE_INVOICE_ID = (
                pending_invoices[2]["_id"] if len(pending_invoices) > 2 else None
            )
            ARCHIVE_INVOICE_ID = (
                pending_invoices[3]["_id"] if len(pending_invoices) > 3 else None
            )
            REJECT_INVOICE_ID = (
                pending_invoices[4]["_id"] if len(pending_invoices) > 4 else None
            )

    print_result("Get Invoices", success, data.get("reward"))
    return success


def test_get_invoice_by_id():
    if not INVOICE_ID:
        print_result("Get Invoice By ID", False)
        return False

    response = api_request("GET", f"/api/invoice/{INVOICE_ID}", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Get Invoice By ID", success, data.get("reward"))
    return success


def test_update_invoice_status():
    if not UPDATE_INVOICE_ID:
        return False

    body = {"status": "APPROVED"}
    response = api_request(
        "PATCH", f"/api/invoice/{UPDATE_INVOICE_ID}/status", body, auth=True
    )

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Update Invoice Status", success, data.get("reward"))
    return success


def test_duplicate_check():
    global SUPPLIER_ID

    body = {"supplierId": SUPPLIER_ID, "amount": 2000, "dueDate": "2026-07-31"}
    response = api_request("POST", "/api/invoice/duplicate-check", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = "reward" in data
    print_result("Duplicate Invoice Check", success, data.get("reward"))
    return success


def test_archive_invoice():
    if not ARCHIVE_INVOICE_ID:
        return False

    body = {"invoiceId": ARCHIVE_INVOICE_ID}
    response = api_request("POST", "/api/invoice/archive", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Archive Invoice", success, data.get("reward"))
    return success


def test_invoice_workflow():
    create = test_create_invoice()
    get_all = test_get_invoices()
    get_one = test_get_invoice_by_id()
    update = test_update_invoice_status()
    duplicate = test_duplicate_check()
    archive = test_archive_invoice()
    return create and get_all and get_one and update and duplicate and archive


# ============================================================
# 5. APPROVAL WORKFLOW TESTING
# ============================================================


def test_approve_invoice():
    if not INVOICE_ID:
        print_result("Approve Invoice", False)
        return False

    body = {"invoiceId": INVOICE_ID, "comment": "Approved by RL agent"}
    response = api_request("POST", "/api/approval/approve", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Approve Invoice", success, data.get("reward"))
    return success


def test_reject_invoice():
    if not REJECT_INVOICE_ID:
        return False

    body = {"invoiceId": REJECT_INVOICE_ID, "reason": "Testing rejection workflow"}
    response = api_request("POST", "/api/approval/reject", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = "reward" in data
    print_result("Reject Invoice", success, data.get("reward"))
    return success


def test_approval_workflow():
    approve = test_approve_invoice()
    reject = test_reject_invoice()
    return approve or reject


# ============================================================
# 6. SUPPLIER WORKFLOW TESTING
# ============================================================


def test_get_suppliers():
    global SUPPLIER_ID, BLACKLIST_SUPPLIER_ID
    response = api_request("GET", "/api/supplier", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and isinstance(data.get("suppliers"), list)
    suppliers = data.get("suppliers")

    if isinstance(suppliers, list):
        # 1. ✅ Find the first active, low-risk supplier
        low_risk_supplier = next(
            (
                s
                for s in suppliers
                if isinstance(s.get("riskScore"), (int, float))
                and s.get("riskScore") < 50
                and s.get("active") is True
            ),
            None,
        )

        # 2. ❌ Find the first blacklisted supplier (checks if 'blacklisted' flag is True OR riskScore is very high)
        blacklisted_supplier = next(
            (
                s
                for s in suppliers
                if s.get("blacklisted") is True
                or (
                    isinstance(s.get("riskScore"), (int, float))
                    and s.get("riskScore") >= 80
                )
            ),
            None,
        )

        # 3. Extract the IDs safely with fallbacks
        SUPPLIER_ID = low_risk_supplier["_id"] if low_risk_supplier else None
        BLACKLIST_SUPPLIER_ID = (
            blacklisted_supplier["_id"] if blacklisted_supplier else None
        )

    print_result("Get Suppliers", success)
    return success


def test_validate_supplier():
    global SUPPLIER_ID

    body = {"supplierId": SUPPLIER_ID}
    response = api_request("POST", "/api/supplier/validate", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = "reward" in data
    print_result("Validate Supplier", success, data.get("reward"))
    return success


def test_blacklist_supplier():
    global BLACKLIST_SUPPLIER_ID
    body = {"supplierId": BLACKLIST_SUPPLIER_ID, "reason": "Testing blacklist action"}
    response = api_request("POST", "/api/supplier/blacklist", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = "reward" in data
    print_result("Blacklist Supplier", success, data.get("reward"))
    return success


def test_supplier_workflow():
    get = test_get_suppliers()
    validate = test_validate_supplier()
    blacklist = test_blacklist_supplier()
    return get and validate and blacklist


# ============================================================
# 7. ACCOUNT / TREASURY WORKFLOW TESTING
# ============================================================


def test_get_accounts():
    global ACCOUNT_ID, TO_ACCOUNT_ID

    response = api_request("GET", "/api/account", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and isinstance(data.get("accounts"), list)

    if isinstance(data.get("accounts"), list) and len(data.get("accounts")) > 0:
        ACCOUNT_ID = data.get("accounts")[0]["_id"]

        if len(data.get("accounts")) > 1:
            TO_ACCOUNT_ID = data.get("accounts")[1]["_id"]

    print_result("Get Accounts", success)
    return success


def test_budget_check():
    body = {"amount": 2000, "department": "SOFTWARE"}
    response = api_request("POST", "/api/account/budget/check", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = "reward" in data
    print_result("Budget Check", success, data.get("reward"))
    return success


def test_account_transfer():
    global ACCOUNT_ID, TO_ACCOUNT_ID

    body = {"fromAccount": ACCOUNT_ID, "toAccount": TO_ACCOUNT_ID, "amount": 1000}
    response = api_request("POST", "/api/account/transfer", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success")
    print_result("Account Transfer", success, data.get("reward"))
    return success


def test_cash_position():
    response = api_request("GET", "/api/account/cash-position", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Cash Position", success)
    return success


def test_account_workflow():
    accounts = test_get_accounts()
    budget = test_budget_check()
    transfer = test_account_transfer()
    cash = test_cash_position()
    return accounts and budget and transfer and cash


# ============================================================
# 8. PAYMENT WORKFLOW TESTING
# ============================================================


def test_pay_invoice():
    global INVOICE_ID, ACCOUNT_ID, TRANSACTION_ID

    if not INVOICE_ID or not ACCOUNT_ID:
        print_result("Pay Invoice", False)
        return False

    body = {"invoiceId": INVOICE_ID, "accountId": ACCOUNT_ID}
    response = api_request("POST", "/api/payment/pay", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success")
    if success:
        state = data.get("state")
        TRANSACTION_ID = state.get("transaction")["_id"]
    print_result("Pay Invoice", success, data.get("reward"))
    return success


def test_refund_payment():
    global TRANSACTION_ID

    body = {"transactionId": TRANSACTION_ID, "reason": "Testing refund workflow"}
    response = api_request("POST", "/api/payment/refund", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success")
    print_result("Refund Payment", success, data.get("reward"))
    return success


def test_cancel_payment():
    global CANCEL_TRANSACTION_ID, TRANSACTION_ID

    body = {
        "transactionId": (
            CANCEL_TRANSACTION_ID if CANCEL_TRANSACTION_ID else TRANSACTION_ID
        )
    }
    response = api_request("POST", "/api/payment/cancel-payment", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success")
    print_result("Cancel Payment", success, data.get("reward"))
    return success


def test_retry_payment():
    global RETRY_INVOICE_ID

    body = {"invoiceId": RETRY_INVOICE_ID}
    response = api_request("POST", "/api/payment/retry-payment", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success")
    print_result("Retry Payment", success, data.get("reward"))
    return success


def test_payment_workflow():
    pay = test_pay_invoice()
    refund = test_refund_payment()
    cancel = test_cancel_payment()
    retry = test_retry_payment()
    return pay or refund or cancel or retry


# ============================================================
# 9. REPORTING WORKFLOW TESTING
# ============================================================


def test_transactions_report():
    global CANCEL_TRANSACTION_ID

    response = api_request("GET", "/api/report/transactions?$status=PENDING", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success")

    if success:
        transactions = data.get("transactions")

        # ✅ Finds the first transaction where status is "PENDING"
        pending_transaction = next(
            (inv for inv in transactions if inv.get("status") == "PENDING"), None
        )

        if pending_transaction:
            CANCEL_TRANSACTION_ID = pending_transaction["_id"]

    print_result("Get Transactions", success)
    return success


def test_generate_report():
    body = {"type": "TRANSACTION_SUMMARY", "format": "JSON"}
    response = api_request("POST", "/api/report/generate-report", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Generate Report", success, data.get("reward"))
    return success


def test_audit_log():
    response = api_request("GET", "/api/report/audit-log", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and isinstance(data.get("logs"), list)
    print_result("Get Audit Log", success)
    return success


def test_reporting_workflow():
    transactions = test_transactions_report()
    report = test_generate_report()
    audit = test_audit_log()
    return transactions and report and audit


# ============================================================
# 10. EPISODE WORKFLOW TESTING
# ============================================================


def test_start_episode():
    global EPISODE_ID

    body = {
        "agentType": "RL",
        "algorithm": "PPO",
        "goal": "Pay all approved invoices",
        "initialState": {"invoices": [], "accounts": [], "suppliers": []},
    }

    response = api_request("POST", "/api/episode/start", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and "episodeId" in data

    if success:
        EPISODE_ID = data["episodeId"]

    print_result("Start Episode", success, data.get("reward"))
    return success


def test_episode_step():
    if not EPISODE_ID:
        return False

    body = {
        "action": "CHECK_BUDGET",
        "endpoint": "/api/account/budget/check",
        "success": True,
        "message": "Budget available",
        "stateBefore": {},
        "stateAfter": {},
        "reward": 10,
    }

    response = api_request("POST", f"/api/episode/{EPISODE_ID}/step", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("Record Episode Step", success, data.get("reward"))
    return success


def test_end_episode():
    if not EPISODE_ID:
        return False

    body = {
        "finalState": {"completedInvoices": 1},
        "terminatedReason": "GOAL_REACHED",
    }
    response = api_request("POST", f"/api/episode/{EPISODE_ID}/end", body, auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True
    print_result("End Episode", success, data.get("totalReward"))
    return success


def test_get_episode():
    if not EPISODE_ID:
        return False

    response = api_request("GET", f"/api/episode/{EPISODE_ID}", auth=True)

    if response is None:
        return False

    data = response.json()
    success = data.get("success") is True and "episode" in data
    print_result("Get Episode", success)
    return success


def test_episode_workflow():
    start = test_start_episode()
    step = test_episode_step()
    end = test_end_episode()
    get = test_get_episode()
    return start and step and end and get


# ============================================================
# REWARD STATISTICS
# ============================================================


def print_statistics():
    print("\n" + "=" * 60)
    print("RL ENVIRONMENT STATISTICS")
    print("=" * 60)
    print(f"Total Actions       : {STATS['total_actions']}")
    print(f"Successful Actions  : {STATS['successful_actions']}")
    print(f"Failed Actions      : {STATS['failed_actions']}")
    print(f"Environment Errors  : {STATS['environment_errors']}")
    print(f"Total Reward        : {STATS['total_reward']}")

    if STATS["total_actions"]:
        average = STATS["total_reward"] / STATS["total_actions"]
    else:
        average = 0

    print(f"Average Reward      : {average:.2f}")
    print("=" * 60 + "\n")


# ============================================================
# COMPLETE TEST EXECUTION
# ============================================================


def run_all_tests():
    init_log()

    print("\n" + "=" * 60)
    print("RL BACKEND ENVIRONMENT TEST STARTED")
    print(f"Logging detailed results to: {LOG_FILE}")
    print("=" * 60)

    results = {}
    results["Health"] = test_health()
    results["Authentication"] = test_authentication()
    results["Episode Tracking"] = test_episode_workflow()
    results["Sandbox"] = test_sandbox()
    results["Supplier Workflow"] = test_supplier_workflow()
    results["Invoice Workflow"] = test_invoice_workflow()
    results["Account Workflow"] = test_account_workflow()
    results["Approval Workflow"] = test_approval_workflow()
    results["Reporting"] = test_reporting_workflow()
    results["Payment Workflow"] = test_payment_workflow()

    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)

    for name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{name:<30}: {status}")

    print_statistics()


# ============================================================
# PROGRAM ENTRY
# ============================================================
if __name__ == "__main__":
    run_all_tests()
