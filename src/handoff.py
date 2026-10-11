"""Warm-handoff brief: who the caller is, what they wanted, what was covered."""


def summarize_handoff(history):
    if not history:
        return "No conversation yet."

    name = None
    for h in history:
        n = h["entities"].get("name")
        if n:
            name = n
            break

    intents = [h["intent"] for h in history]
    main_intent = max(set(intents), key=intents.count)

    details = []
    for h in history:
        for d in h["entities"].get("datetimes", []):
            details.append(d["text"])
        if h["entities"].get("phone"):
            details.append(f"phone {h['entities']['phone']}")

    covered = [f"- Caller: '{h['user']}' -> intent={h['intent']}" for h in history]

    return (
        "WARM HANDOFF BRIEF\n"
        f"Caller: {name or 'unknown'}\n"
        f"Main need: {main_intent}\n"
        f"Key details: {', '.join(details) if details else 'none given yet'}\n"
        "Covered so far:\n"
        + "\n".join(covered)
        + "\nRecommendation: pick up from last intent, no need to re-ask."
    )
