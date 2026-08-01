import requests
import json
import sys
import os
import dotenv
import time


dotenv.load_dotenv()


BASE_URL = f"http://localhost:{os.getenv('PORT')}"

HEADERS = {
    "Content-Type": "application/json"
}


TEST_USER = {

    "username": "RL Agent Admin",

    "email": "rl_agent@sandbox.com",

    "password": "Password123!",

    "role": "AGENT_BOT"

}


JWT_TOKEN = None

EPISODE_ID = None

INVOICE_ID = None

TRANSACTION_ID = None


SUCCESS = 0
FAILED = 0



# ---------------------------------------
# Helper
# ---------------------------------------

def print_status(name,response):

    global SUCCESS,FAILED


    if response.status_code in [200,201]:

        SUCCESS += 1
        status="SUCCESS"

    else:

        FAILED += 1
        status="FAILED"



    colour = "\033[92m" if status=="SUCCESS" else "\033[91m"

    reset="\033[0m"


    print(
        f"[{colour}{status}{reset}] {name}"
        f" ({response.status_code})"
    )


    try:

        data=response.json()

        print(
            json.dumps(
                data,
                indent=2
            )[:500]
        )


    except:

        print(response.text[:300])


    print("-"*60)



# ---------------------------------------
# Health
# ---------------------------------------

def test_health():

    print("\n1. Health Check")


    try:

        res=requests.get(
            f"{BASE_URL}/"
        )


        print_status(
            "GET /",
            res
        )


    except requests.exceptions.ConnectionError:

        print(
            "Backend not running"
        )

        sys.exit()



# ---------------------------------------
# Authentication
# ---------------------------------------

def test_login():


    global JWT_TOKEN


    print("\n2. Authentication")


    requests.post(
        f"{BASE_URL}/api/auth/register",
        json=TEST_USER,
        headers=HEADERS
    )


    res=requests.post(

        f"{BASE_URL}/api/auth/login",

        json={

            "email":TEST_USER["email"],

            "password":TEST_USER["password"]

        },

        headers=HEADERS

    )


    print_status(
        "POST /api/auth/login",
        res
    )


    if res.status_code==200:


        JWT_TOKEN=res.json()["token"]


        HEADERS["Authorization"]=(
            f"Bearer {JWT_TOKEN}"
        )

    else:

        sys.exit()



# ---------------------------------------
# Sandbox
# ---------------------------------------

def test_sandbox():


    print("\n3. Sandbox")


    res=requests.post(

        f"{BASE_URL}/api/sandbox/reset",

        headers=HEADERS

    )


    print_status(
        "POST /api/sandbox/reset",
        res
    )



    res=requests.get(

        f"{BASE_URL}/api/sandbox/state",

        headers=HEADERS

    )


    print_status(
        "GET /api/sandbox/state",
        res
    )



# ---------------------------------------
# Episode
# ---------------------------------------

def start_episode():

    global EPISODE_ID


    print("\n4. Episode Start")


    payload={

        "agentType":"RL",

        "algorithm":"PPO",

        "goal":
        "Pay approved invoices",

        "initialState":{}

    }


    res=requests.post(

        f"{BASE_URL}/api/episode/start",

        json=payload,

        headers=HEADERS

    )


    print_status(
        "POST /api/episode/start",
        res
    )


    if res.status_code==201:

        EPISODE_ID=res.json()["episodeId"]




# ---------------------------------------
# Invoice
# ---------------------------------------

def test_invoice():


    global INVOICE_ID


    print("\n5. Invoice")


    payload={

        "supplierName":
        "Microsoft",

        "amount":
        2500,

        "description":
        "Cloud services",

        "dueDate":
        "2026-12-31"

    }


    res=requests.post(

        f"{BASE_URL}/api/invoice",

        json=payload,

        headers=HEADERS

    )


    print_status(
        "POST /api/invoice",
        res
    )


    if res.status_code==201:


        data=res.json()


        INVOICE_ID=(
            data.get("_id")
            or
            data.get("invoice",{}).get("_id")
        )



# ---------------------------------------
# Approval
# ---------------------------------------

def approve_invoice():


    print("\n6. Approval")


    res=requests.post(

        f"{BASE_URL}/api/approval/approve",

        json={

            "invoiceId":INVOICE_ID

        },

        headers=HEADERS

    )


    print_status(

        "POST /api/approval/approve",

        res

    )



# ---------------------------------------
# Supplier
# ---------------------------------------

def validate_supplier():


    print("\n7. Supplier")


    res=requests.post(

        f"{BASE_URL}/api/supplier/validate",

        json={

            "supplierId":"Microsoft"

        },

        headers=HEADERS

    )


    print_status(

        "POST /api/supplier/validate",

        res

    )



# ---------------------------------------
# Account
# ---------------------------------------

def check_budget():


    print("\n8. Budget")


    res=requests.post(

        f"{BASE_URL}/api/account/budget/check",

        json={

            "amount":2500

        },

        headers=HEADERS

    )


    print_status(

        "POST /api/account/budget/check",

        res

    )




# ---------------------------------------
# Payment
# ---------------------------------------

def pay_invoice():


    global TRANSACTION_ID


    print("\n9. Payment")


    res=requests.post(

        f"{BASE_URL}/api/payment/pay",

        json={

            "invoiceId":INVOICE_ID,

            "paymentMethod":"BANK"

        },

        headers=HEADERS

    )


    print_status(

        "POST /api/payment/pay",

        res

    )


    if res.status_code==200:


        data=res.json()


        TRANSACTION_ID=(

            data.get("_id")

            or

            data.get("transaction",{}).get("_id")

        )




# ---------------------------------------
# Reports
# ---------------------------------------

def test_reports():


    print("\n10. Reports")


    res=requests.get(

        f"{BASE_URL}/api/report/transactions",

        headers=HEADERS

    )


    print_status(

        "GET /api/report/transactions",

        res

    )




# ---------------------------------------
# End Episode
# ---------------------------------------

def end_episode():


    print("\n11. Episode End")


    res=requests.post(

        f"{BASE_URL}/api/episode/{EPISODE_ID}/end",

        json={

            "terminatedReason":
            "GOAL_REACHED",

            "finalState":{}

        },

        headers=HEADERS

    )


    print_status(

        "POST /api/episode/end",

        res

    )




# ---------------------------------------
# Run
# ---------------------------------------

def run():


    print("="*60)

    print(
        "RL FINANCE SANDBOX TEST"
    )

    print("="*60)



    test_health()

    test_login()

    test_sandbox()

    start_episode()

    test_invoice()

    approve_invoice()

    validate_supplier()

    check_budget()

    pay_invoice()

    test_reports()

    end_episode()



    print("\n")

    print("="*60)

    print(
        f"SUCCESS: {SUCCESS}"
    )

    print(
        f"FAILED: {FAILED}"
    )

    print("="*60)




if __name__=="__main__":

    run()