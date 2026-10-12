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
    junk = {"to", "a", "the", "and", "for", "on", "at", "in", "of"}
    for m, d in results:
        # strip trailing junk words: "Tuesday and" -> "Tuesday"
        words = m.strip().split()
        while len(words) > 1 and words[-1].lower().rstrip(",.") in junk:
            words.pop()
        clean = " ".join(words).strip()
        mc = clean.lower()
        if len(mc) < 3:
            continue
        if mc in junk:
            continue
        out.append({"text": clean, "value": d.isoformat()})
    return out


# words that are never a person's name — stops "this is ridiculous"
# from extracting name="ridiculous"
NOT_A_NAME = {
    "ridiculous", "terrible", "horrible", "awful", "wrong", "right", "fine",
    "okay", "ok", "great", "good", "bad", "stupid", "crazy", "insane",
    "difficult", "hard", "easy", "important", "urgent", "possible",
    "true", "false", "real", "big", "small", "new", "old", "all", "it",
    "this", "that",
}


def extract_name(text):
    patterns = [
        r"this is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"i'm ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        r"my name is ([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            name = m.group(1).strip()
            if name.lower() in NOT_A_NAME:
                continue
            return name
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
