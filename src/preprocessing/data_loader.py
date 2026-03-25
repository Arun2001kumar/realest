
import os
import sys
import pandas as pd # type: ignore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import RAW_DATA_FILE
def load_conversations(path: str = RAW_DATA_FILE) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Conversation Data", header=1)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Conversation"]).reset_index(drop=True)

    required = ["Conv ID", "Customer Name", "Salesman Name",
                "Property Name", "Location", "Conversation"]
    for col in required:
        if col not in df.columns:
            df[col] = "Unknown"

    return df


def get_conversation_list(df: pd.DataFrame) -> list[dict]:
    return [
        {"id": row["Conv ID"], "conversation": row["Conversation"]}
        for _, row in df.iterrows()
    ]
