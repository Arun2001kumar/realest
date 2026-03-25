import os
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
DATA_PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
PROMPTS_DIR   = os.path.join(BASE_DIR, "prompts")

RAW_DATA_FILE = os.path.join(DATA_RAW_DIR, "RealEstate_Conversation_110Records.xlsx")
CACHE_FILE    = os.path.join(DATA_PROC_DIR, "genai_results_cache.json")
GEMINI_MODEL      = "models/gemini-2.5-flash"
MAX_TOKENS        = 4096
API_TEMPERATURE   = 0.1   
BATCH_DELAY_SEC   = 0.3 
BUDGET_LABELS    = ["Below 50L", "50L - 1Cr", "1Cr - 2Cr", "Above 2Cr"]
AREA_LABELS = [
    "Whitefield", "Koramangala", "Electronic City", "Indiranagar",
    "HSR Layout", "Marathahalli", "JP Nagar", "Sarjapur",
    "Bellandur", "Yelahanka", "Devanahalli", "Kanakapura",
    "Varthur", "Thanisandra", "Hebbal", "Other"
]
INTEREST_LABELS  = [
    "Not Interested", "Low Interest", "Neutral", "Interested", "Highly Interested"
]
SALES_STAGES = {
    "Not Interested":    "Closed - No Sale",
    "Low Interest":      "Not Responding",
    "Neutral":           "Brochure Sent",
    "Interested":        "Follow-up Pending",
    "Highly Interested": "Site Visit Scheduled",
}
CONVERSION_LIKELIHOOD = {
    "Not Interested":    "Very Low",
    "Low Interest":      "Low",
    "Neutral":           "Medium",
    "Interested":        "High",
    "Highly Interested": "Very High",
}

INTEREST_SCORE_MAP = {
    "Not Interested":    0.05,
    "Low Interest":      0.25,
    "Neutral":           0.50,
    "Interested":        0.72,
    "Highly Interested": 0.92,
}

INTEREST_COLORS = {
    "Highly Interested": "#1a7a4a",
    "Interested":        "#2E75B6",
    "Neutral":           "#d97706",
    "Low Interest":      "#ea580c",
    "Not Interested":    "#dc2626",
}
INTEREST_BG = {
    "Highly Interested": "#d1fae5",
    "Interested":        "#dbeafe",
    "Neutral":           "#fef3c7",
    "Low Interest":      "#ffedd5",
    "Not Interested":    "#fee2e2",
}
