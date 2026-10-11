"""Entity extraction: dates/times/names/phones.

Rule-based — parsed, not trained. Same logic as the Colab build.
"""
import re
from dateparser.search import search_dates


def extract_datetime(text):
    results = search_dates(text, settings={'PREFER_DATES_FROM': 'future'})
    if not results:
        return []
    out = []
    junk = {"to", "a", "the", "and", "for", "on", "at", "in"}
    for m, d in results:
        mc = m.strip().lower()
        if len(mc) < 3:
            continue
        if mc in junk:
            continue
        out.append({"text": m, "value": d.isoformat()})
    return out


def extract_name(text):
    patterns = [
        r"this is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"i'm ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"my name is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return None


def extract_phone(text):
    m = re.search(r"(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})", text)
    return m.group(1) if m else None


def extract_entities(text):
    return {
        "datetimes": extract_datetime(text),
        "name": extract_name(text),
        "phone": extract_phone(text),
    }
