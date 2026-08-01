import ollama
import json
import numpy as np
from config import Config


class LLMPlanner:
    """
    Uses a local Ollama model to provide action prior probabilities.
    Responses are cached to avoid repeated LLM calls for the same state.
    """

    def __init__(self, model_name=Config.OLLAMA_MODEL):
        self.model_name = model_name

        # Cache previous LLM responses
        self._cache = {}

        self.action_labels = [
            "0: Fetch Invoices",
            "1: Post Invoice",
            "2: Approve Invoice",
            "3: Execute Payment",
            "4: Reconcile Transaction"
        ]

    def get_action_priors(self, observation):
        """
        Returns a probability distribution over actions.
        """

        # Convert observation to tuple so it can be used as a cache key
        state_key = tuple(observation)

        # Return cached response if available
        if state_key in self._cache:
            return self._cache[state_key]

        pending_inv, unapproved, unpaid, balance = observation

        prompt = f"""
            You are an expert financial workflow automation assistant.

            Current State:
            - Pending Invoices: {pending_inv}
            - Unapproved Invoices: {unapproved}
            - Unpaid Invoices: {unpaid}
            - Bank Balance: £{balance}

            Available Actions:
            0: Fetch Invoices
            1: Post Invoice
            2: Approve Invoice
            3: Execute Payment
            4: Reconcile Transaction

            Goal:
            Process invoices in the correct sequence while avoiding invalid actions.

            Return ONLY a JSON object where the keys are action indices and the values
            are probabilities that sum to exactly 1.

            Example:
            {{"0":0.05,"1":0.10,"2":0.60,"3":0.20,"4":0.05}}
            """

        try:
            print(f"--- Querying LLM for state {state_key} ---")

            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = response["message"]["content"].strip()

            # Remove markdown if the model wraps JSON in ```json
            response_text = response_text.replace("```json", "")
            response_text = response_text.replace("```", "").strip()

            probs_dict = json.loads(response_text)
            print(f" Probability for Observation {observation}: {probs_dict}")

            priors = np.zeros(Config.ACTION_DIM, dtype=np.float32)

            for idx, prob in probs_dict.items():
                idx = int(idx)
                if 0 <= idx < Config.ACTION_DIM:
                    priors[idx] = float(prob)

            # Normalize
            total = priors.sum()

            if total > 0:
                priors /= total
            else:
                priors = np.ones(Config.ACTION_DIM, dtype=np.float32)
                priors /= Config.ACTION_DIM

        except Exception as e:
            print(f"Ollama Error: {e}")

            priors = np.ones(Config.ACTION_DIM, dtype=np.float32)
            priors /= Config.ACTION_DIM

        # Cache result
        self._cache[state_key] = priors

        return priors