"""
test_llm.py

Simple integration test for:

    Ollama
    prompts.py
    planner.py
    parser.py
    cache.py
"""

from config.config import config

from llm.planner import LLMPlanner


def main():

    print("=" * 70)
    print("LLM PLANNER TEST")
    print("=" * 70)

    # ==========================================================
    # Configuration
    # ==========================================================

    print("\nConfiguration")

    print(f"Model       : {config.llm.MODEL}")

    print(f"Ollama URL  : {config.llm.BASE_URL}")

    print(f"Timeout     : {config.llm.TIMEOUT}")

    print(f"Temperature : {config.llm.TEMPERATURE}")

    print(f"Cache       : {config.llm.USE_CACHE}")

    # ==========================================================
    # Example initial environment state
    # ==========================================================

    state = {
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

    planner = LLMPlanner()

    try:

        # ======================================================
        # First Call
        # ======================================================

        print("\n" + "=" * 70)
        print("FIRST PLANNER CALL")
        print("=" * 70)

        result = planner.plan(
            goal=config.agent.TASK,
            state=state,
        )

        if not result.get(
            "success",
            False,
        ):

            print("\nPlanner failed:")

            print(result.get("error"))

            return

        print("\nCached:")

        print(result.get("cached"))

        print("\nAction Names:")

        for index, action in enumerate(
            result.get(
                "action_names",
                [],
            ),
            start=1,
        ):

            print(f"{index}. {action}")

        print("\nAction IDs:")

        print(result.get("action_ids"))

        print("\nPrerequisites:")

        for (
            action_id,
            prerequisites,
        ) in result.get("prerequisites", {}).items():

            print(f"{action_id}: " f"{prerequisites}")

        print("\nLatency:")

        print(f"{result.get('latency_ms', 0):.2f} ms")

        # ======================================================
        # Second Call
        #
        # Should come from cache.
        # ======================================================

        print("\n" + "=" * 70)
        print("SECOND PLANNER CALL")
        print("=" * 70)

        cached_result = planner.plan(
            goal=config.agent.TASK,
            state=state,
        )

        print("\nCached:")

        print(cached_result.get("cached"))

        print("\nAction Names:")

        print(cached_result.get("action_names"))

        # ======================================================
        # Metrics
        # ======================================================

        print("\n" + "=" * 70)
        print("PLANNER METRICS")
        print("=" * 70)

        metrics = planner.get_metrics()

        for key, value in metrics.items():

            print(f"{key}: {value}")

        print("\n" + "=" * 70)
        print("LLM TEST COMPLETED")
        print("=" * 70)

    finally:

        planner.close()


if __name__ == "__main__":

    main()
