"""
parser.py

Parser for LLM planning responses.

The parser converts the free-form response produced by the
LLM planner into a small structured planning hint.

Important:
    The parser does NOT execute actions.
    The parser does NOT call the backend.
    The parser does NOT override the PPO policy.

The PPO agent remains responsible for selecting the actual
environment action.
"""

import re
from typing import Any, Dict, List


class PlanningParser:
    """
    Converts an LLM response into a structured planning hint.
    """

    # ==========================================================
    # Known Planning Concepts
    # ==========================================================

    PLANNING_KEYWORDS = {
        "get_invoices": [
            "get invoice",
            "get invoices",
            "retrieve invoice",
            "retrieve invoices",
            "fetch invoice",
            "fetch invoices",
            "find invoice",
            "find invoices",
        ],
        "validate_supplier": [
            "validate supplier",
            "supplier validation",
            "verify supplier",
            "check supplier",
        ],
        "check_budget": [
            "check budget",
            "verify budget",
            "budget check",
            "available budget",
        ],
        "approve_invoice": [
            "approve invoice",
            "invoice approval",
            "approval",
            "approve the invoice",
        ],
        "pay_invoice": [
            "pay invoice",
            "make payment",
            "process payment",
            "payment",
        ],
        "generate_report": [
            "generate report",
            "create report",
            "produce report",
            "report",
        ],
    }

    # ==========================================================
    # Constructor
    # ==========================================================

    def __init__(self):
        """
        Initialize the parser.
        """

        self.keyword_map = self._build_keyword_map()

    # ==========================================================
    # Parse Response
    # ==========================================================

    def parse(
        self,
        response: str,
    ) -> Dict[str, Any]:
        """
        Parse an LLM planning response.

        Parameters
        ----------
        response : str
            Raw response returned by the LLM.

        Returns
        -------
        dict
            Structured planning information.
        """

        if not response:

            return {
                "recommendation": "",
                "keywords": [],
                "confidence": 0.0,
                "valid": False,
            }

        cleaned_response = self._clean_response(response)

        detected_keywords = self._detect_keywords(cleaned_response)

        confidence = self._estimate_confidence(detected_keywords)

        return {
            "recommendation": cleaned_response,
            "keywords": detected_keywords,
            "confidence": confidence,
            "valid": True,
        }

    # ==========================================================
    # Clean Response
    # ==========================================================

    @staticmethod
    def _clean_response(
        response: str,
    ) -> str:
        """
        Remove unnecessary formatting from the LLM response.
        """

        response = response.strip()

        # Remove common Markdown formatting.
        response = re.sub(
            r"^```.*?$",
            "",
            response,
            flags=re.MULTILINE,
        )

        response = response.replace(
            "Recommendation:",
            "",
        )

        response = response.strip()

        return response

    # ==========================================================
    # Detect Planning Keywords
    # ==========================================================

    def _detect_keywords(
        self,
        response: str,
    ) -> List[str]:
        """
        Detect known planning concepts in the response.

        Returns
        -------
        list[str]
            Detected planning concepts.
        """

        response_lower = response.lower()

        detected = []

        for action_name, phrases in self.keyword_map.items():

            for phrase in phrases:

                if phrase in response_lower:

                    detected.append(action_name)

                    break

        return detected

    # ==========================================================
    # Build Keyword Map
    # ==========================================================

    @staticmethod
    def _build_keyword_map():
        """
        Return the known planning keyword mapping.

        Kept as a separate method so the mapping can later
        be loaded from configuration if required.
        """

        return PlanningParser.PLANNING_KEYWORDS

    # ==========================================================
    # Confidence
    # ==========================================================

    @staticmethod
    def _estimate_confidence(
        detected_keywords: List[str],
    ) -> float:
        """
        Estimate confidence based on whether the parser
        detected a known planning concept.

        This is NOT an LLM confidence score.

        It simply indicates whether the parser found
        recognizable planning information.
        """

        if not detected_keywords:

            return 0.0

        if len(detected_keywords) == 1:

            return 0.75

        return 1.0
