import os
import sys
import json
import hashlib
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import CACHE_FILE
def _hash(text: str) -> str:
    return hashlib.md5(text.strip().encode()).hexdigest()
def load_cache() -> dict:
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
def save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)
def get_cached(conversation: str, cache: Optional[dict] = None) -> Optional[dict]:
    c = cache if cache is not None else load_cache()
    return c.get(_hash(conversation))
def set_cached(conversation: str, result: dict, cache: Optional[dict] = None) -> dict:
    c = cache if cache is not None else load_cache()
    c[_hash(conversation)] = result
    save_cache(c)
    return c
def clear_cache():
    if os.path.exists(CACHE_FILE):
        os.remove(CACHE_FILE)
def cache_size() -> int:
    return len(load_cache())
