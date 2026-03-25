
import os
import sys
import json
import time
import re
import requests
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import (GEMINI_MODEL, MAX_TOKENS, API_TEMPERATURE,
                    PROMPTS_DIR, INTEREST_SCORE_MAP, SALES_STAGES,
                    CONVERSION_LIKELIHOOD, BATCH_DELAY_SEC)

_SYSTEM_PROMPT_PATH = os.path.join(PROMPTS_DIR, "analysis_system_prompt.txt")
with open(_SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/"



def _default_result(error_msg: str = "") -> dict:
    return {
        "budget_range":          "50L - 1Cr",
        "budget_reasoning":      "Could not be determined.",
        "preferred_area":        "Unknown",
        "area_reasoning":        "Could not be determined.",
        "interest_level":        "Neutral",
        "sentiment_score":       0.5,
        "sales_stage":           "Brochure Sent",
        "conversion_likelihood": "Medium",
        "key_signals":           [],
        "customer_persona":      "Unknown",
        "urgency":               "Unknown",
        "pain_points":           [],
        "positive_signals":      [],
        "recommended_action":    "Follow up with standard brochure.",
        "summary":               error_msg or "Analysis unavailable.",
        "_error":                error_msg,
    }



def _extract_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not extract JSON from response: {text[:200]}")

def _normalise(result: dict) -> dict:
    from config import (BUDGET_LABELS, INTEREST_LABELS,
                        INTEREST_SCORE_MAP, SALES_STAGES, CONVERSION_LIKELIHOOD)

    if result.get("budget_range") not in BUDGET_LABELS:
        result["budget_range"] = "50L - 1Cr"
    if result.get("interest_level") not in INTEREST_LABELS:
        result["interest_level"] = "Neutral"
    il = result["interest_level"]
    result["sales_stage"]           = SALES_STAGES.get(il, "Brochure Sent")
    result["conversion_likelihood"] = CONVERSION_LIKELIHOOD.get(il, "Medium")
    try:
        score = float(result.get("sentiment_score", 0.5))
        result["sentiment_score"] = round(max(0.0, min(1.0, score)), 2)
    except (TypeError, ValueError):
        result["sentiment_score"] = INTEREST_SCORE_MAP.get(il, 0.5)
    for key in ("key_signals", "pain_points", "positive_signals"):
        if not isinstance(result.get(key), list):
            result[key] = []
    for key, default in [
        ("preferred_area", "Unknown"),
        ("customer_persona", "Unknown"),
        ("urgency", "Unknown"),
        ("recommended_action", "Follow up with customer."),
        ("summary", ""),
        ("budget_reasoning", ""),
        ("area_reasoning", ""),
    ]:
        if not result.get(key):
            result[key] = default

    return result


def analyse_conversation(conversation: str,
                          api_key: Optional[str] = None,
                          retries: int = 2) -> dict:
    
    key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        return _default_result("No API key provided. Set GEMINI_API_KEY.")

    headers = {
        "Content-Type": "application/json",
    }

    url = f"{GEMINI_API_URL}{GEMINI_MODEL}:generateContent?key={key}"

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"Analyse this real estate customer-salesman conversation and return ONLY a JSON object as specified.\n\nCONVERSATION:\n{conversation}"}]
            }
        ],
        "generationConfig": {
            "temperature": API_TEMPERATURE,
            "maxOutputTokens": MAX_TOKENS,
            "responseMimeType": "application/json"
        }
    }

    for attempt in range(retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return _default_result("No content returned from Gemini API.")
            
            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")

            result = _extract_json(text)
            return _normalise(result)

        except requests.exceptions.HTTPError as e:
            if resp.status_code == 429 and attempt < retries:
                time.sleep(2 ** attempt * 2)
                continue
            return _default_result(f"API error {resp.status_code}: {str(e)}")
        except requests.exceptions.Timeout:
            if attempt < retries:
                time.sleep(2)
                continue
            return _default_result("API request timed out.")
        except Exception as e:
            return _default_result(f"Error: {str(e)}")

    return _default_result("Max retries exceeded.")


def analyse_batch(conversations: list[str],
                  api_key: Optional[str] = None,
                  progress_callback=None,
                  delay: float = BATCH_DELAY_SEC) -> list[dict]:
    results = []
    total   = len(conversations)
    for i, conv in enumerate(conversations):
        result = analyse_conversation(conv, api_key=api_key)
        results.append(result)
        if progress_callback:
            progress_callback(i + 1, total, result)
        if i < total - 1:
            time.sleep(delay)
    return results
