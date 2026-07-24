import requests
import json
import sys
import os
import dotenv

dotenv.load_dotenv()

# Base configuration
BASE_URL = f"http://localhost:{os.getenv('PORT')}"  # Adjust port if running on a different port
HEADERS = {"Content-Type": "application/json"}

# Test user credentials
TEST_USER = {
    "name": "RL Agent Admin",
    "email": "rl_agent@sandbox.com",
    "password": "Password123!",
}

# Global state holders for testing
JWT_TOKEN = None
CREATED_INVOICE_ID = None
AGENT_ACCOUNT_NUMBER = None
INITIATED_TRANSACTION_ID = None

def print_status(test_name, response):
    """Helper function to print test results cleanly."""
    status = "SUCCESS" if response.status_code in [200, 201] else "FAILED"
    color_code = "\033[92m" if status == "SUCCESS" else "\033[91m"
    reset_code = "\033[0m"
    
    print(f"[{color_code}{status}{reset_code}] {test_name} (Status: {response.status_code})")
    try:
        data = response.json()
        print(f"   Response: {json.dumps(data, indent=2)[:300]}...")  # Truncate long responses
    except Exception:
        print(f"   Response: {response.text[:200]}")
    print("-" * 60)


def test_health_check():
    """1. Test root server health endpoint."""
    print("\n--- 1. Testing Server Health ---")
    try:
        res = requests.get(f"{BASE_URL}/")
        print_status("Health Check Endpoint GET /", res)
    except requests.exceptions.ConnectionError:
        print(f"\033[91m[ERROR] Could not connect to backend server at {BASE_URL}. Is server.js running?\033[0m")
        sys.exit(1)


def test_auth_endpoints():
    """2. Test Authentication Routes (Register & Login)."""
    print("\n--- 2. Testing Auth Routes (/api/auth) ---")

    # Register Test User (ignore failure if user already exists)
    reg_res = requests.post(f"{BASE_URL}/api/auth/register", json=TEST_USER, headers=HEADERS)
    print_status("User Registration POST /api/auth/register", reg_res)

    # Login Test User
    login_payload = {
        "email": TEST_USER["email"],
        "password": TEST_USER["password"]
    }
    login_res = requests.post(f"{BASE_URL}/api/auth/login", json=login_payload, headers=HEADERS)
    print_status("User Login POST /api/auth/login", login_res)

    if login_res.status_code == 200:
        JWT_TOKEN = login_res.json().get("token")
        HEADERS["Authorization"] = f"Bearer {JWT_TOKEN}"
        print(f"   Stored JWT Token successfully.")
    else:
        print("\033[91m[CRITICAL] Login failed. Authentication endpoints must work before testing protected routes.\033[0m")
        sys.exit(1)


def test_sandbox_endpoints():
    """3. Test Sandbox Endpoints (/api/sandbox)."""
    print("\n--- 3. Testing Sandbox Control Routes (/api/sandbox) ---")

    # Reset Sandbox Database State
    reset_res = requests.post(f"{BASE_URL}/api/sandbox/reset", headers=HEADERS)
    print_status("Sandbox Reset POST /api/sandbox/reset", reset_res)

    # Get Sandbox State Observation
    state_res = requests.get(f"{BASE_URL}/api/sandbox/state", headers=HEADERS)
    print_status("Sandbox State GET /api/sandbox/state", state_res)

    if state_res.status_code == 200:
        AGENT_ACCOUNT_NUMBER = state_res.json().get("account",{"accountNumber": ""})["accountNumber"]


def test_finance_endpoints():
    """4. Test Corporate Finance Operational Routes (/api/finance)."""
    print("\n--- 4. Testing Finance Routes (/api/finance) ---")

    # Fetch Invoices
    inv_res = requests.get(f"{BASE_URL}/api/finance/invoices", headers=HEADERS)
    print_status("Get Invoices GET /api/finance/invoices", inv_res)

    # Create New Invoice
    new_inv_payload = {
        "vendorName": "Acme Tech Solutions",
        "amount": 2500.00,
        "description": "Cloud Infrastructure Services",
        "dueDate": "2026-12-31"
    }
    create_inv_res = requests.post(f"{BASE_URL}/api/finance/invoices", json=new_inv_payload, headers=HEADERS)
    print_status("Create Invoice POST /api/finance/invoices", create_inv_res)

    if create_inv_res.status_code == 201:
        CREATED_INVOICE_ID = create_inv_res.json().get("_id") or create_inv_res.json().get("invoice", {}).get("_id")

    # Approve Invoice (if ID exists)
    if CREATED_INVOICE_ID:
        approve_res = requests.patch(f"{BASE_URL}/api/finance/invoices/{CREATED_INVOICE_ID}/status", json={"status": "APPROVED"}, headers=HEADERS)
        print_status(f"Approve Invoice PUT /api/finance/invoices/{CREATED_INVOICE_ID}/status", approve_res)

    # Pay Invoice / Initiate Transaction
    pay_payload = {
        "invoiceId": CREATED_INVOICE_ID,
        "paymentMethod": "Wire Transfer",
        "accountNumber": AGENT_ACCOUNT_NUMBER
    }
    pay_res = requests.post(f"{BASE_URL}/api/finance/pay", json=pay_payload, headers=HEADERS)
    print_status("Initiate Payment POST /api/finance/pay", pay_res)

    if pay_res.status_code == 200:
        INITIATED_TRANSACTION_ID = pay_res.json().get("_id") or pay_res.json().get("transaction", {}).get("_id")

    # Reconcile Transactions
    rec_res = requests.post(f"{BASE_URL}/api/finance/reconcile", json={"transactionId": INITIATED_TRANSACTION_ID}, headers=HEADERS)
    print_status("Reconcile Ledger POST /api/finance/reconcile", rec_res)


def run_all_tests():
    """Run all API endpoint validation tests in sequence."""
    print("=" * 60)
    print(" STARTING BACKEND SANDBOX API TEST SUITE ")
    print("=" * 60)

    test_health_check()
    test_auth_endpoints()
    test_sandbox_endpoints()
    test_finance_endpoints()

    print("\n")
    print("=" * 60)
    print(" API TESTING COMPLETE ")
    print(" If all endpoints above show [SUCCESS], your backend is ready for RL Gym environment integration!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()