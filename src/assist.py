"""Assist mode: the visitor plays Nikki (office staff), the app plays a
scripted caller, and the AI whispers help to Nikki in a private panel.

The caller is scripted per scenario; the AI's assistance (account lookup,
frustration detection, suggested replies, wrap-up notes) runs live on the
actual caller lines. Frustration detection here is a keyword scorer —
production would use a trained sentiment classifier. Everything the AI shows
Nikki is visually separated from what the caller "hears."
"""

from .demo_data import find_customer

import re

# ------------------------------------------------------------------
# Frustration detection — demo stand-in for a production sentiment model
# ------------------------------------------------------------------
FRUSTRATION_LEXICON = {
    "ridiculous": 3, "furious": 3, "scam": 3,
    "unacceptable": 2, "angry": 2, "terrible": 2, "awful": 2,
    "hate": 2, "worst": 2, "robbed": 2, "disgusting": 2,
    "money back": 2,
    "don't want": 1, "stop": 1, "now": 1, "today": 1,
    "immediately": 1, "manager": 1, "excuse": 1, "excuses": 1, "twice": 1,
}


def frustration_level(text):
    t = text.lower()
    score = sum(pts for phrase, pts in FRUSTRATION_LEXICON.items() if phrase in t)
    if score >= 4:
        return "HIGH"
    if score >= 2:
        return "MEDIUM"
    return "LOW"


CARD_ICONS = {
    "invoice": "🧾",
    "suggestion": "💬",
    "alert": "⚠️",
    "record": "📋",
    "note": "ℹ️",
    "next_step": "➡️",
    "brief": "🔥",
    "wrapup": "📝",
}


def _invoice_body(customer_name):
    c = find_customer(customer_name)
    inv = c["current_invoice"]
    parts = [f"**{c['name']}** — invoice {inv['id']} ({inv['period']})",
             f"Balance due: **${inv['amount_due']:.2f}** — due {inv['due_date']}"]
    if c.get("past_due"):
        parts.append(f"Past due: **${c['past_due']:.2f}**")
    parts.append("Items: " + ", ".join(f"{l} (${a:.2f})" for l, a in inv["line_items"]))
    return "\n\n".join(parts)


def build_cards(step, caller_text):
    """Resolve a scenario step's card specs into renderable cards.

    Invoice cards are built live from demo data; frustration alerts are
    computed live from the caller line.
    """
    cards = []
    for spec in step.get("cards", []):
        spec = dict(spec)
        if spec.get("kind") == "invoice":
            spec["body"] = _invoice_body(spec.pop("customer"))
        cards.append(spec)
    if step.get("sentiment_watch"):
        level = frustration_level(caller_text)
        if level in ("MEDIUM", "HIGH"):
            cards.insert(0, {
                "kind": "alert",
                "title": f"Sentiment alert — frustration {level}",
                "body": ("Acknowledge the feeling before the facts. "
                         "Don't debate the charge or promise outcomes."),
            })
    return cards


# ------------------------------------------------------------------
# Scenarios — scripted caller lines + the AI assistance each one triggers
# ------------------------------------------------------------------
ASSIST_SCENARIOS = {
    "billing": {
        "label": "Billing question",
        "caller_name": "Sarah",
        "steps": [
            {
                "caller": ("Hi, this is Sarah — I'm looking at my bill and I'm a little "
                           "confused. What's my current balance?"),
                "sentiment_watch": True,
                "cards": [
                    {"kind": "invoice", "customer": "Sarah", "title": "Account lookup"},
                    {"kind": "suggestion", "title": "Suggested reply",
                     "body": ("Happy to check that for you, Sarah! Your current balance is $85, "
                              "due November 1st. I also see $45 past due from September — "
                              "want to go over it together?")},
                ],
            },
            {
                "caller": "Wait, $45 past due? I could've sworn I paid that back in September.",
                "sentiment_watch": True,
                "cards": [
                    {"kind": "note", "title": "Payment history",
                     "body": "No payment recorded since **Aug 12** ($85). Nothing in September."},
                    {"kind": "suggestion", "title": "Suggested reply",
                     "body": ("I don't see a September payment on my end — it may just not have "
                              "posted yet. I can have our billing team double-check, or we can "
                              "take care of it right now. Which would you prefer?")},
                ],
            },
            {
                "caller": "Ugh, fine — can I just pay the whole $130 now and be done with it?",
                "sentiment_watch": True,
                "cards": [
                    {"kind": "next_step", "title": "Recommended next step",
                     "body": ("Offer a **secure payment link by text**. Never read card numbers "
                              "aloud on a recorded line. If they insist on paying by phone, "
                              "warm-transfer to billing.")},
                    {"kind": "suggestion", "title": "Suggested reply",
                     "body": ("Absolutely — I can text you a secure payment link right now and "
                              "you'll be all set. What's the best mobile number to send it to?")},
                ],
            },
        ],
        "wrap_up": ("**Caller:** Sarah\n\n**What happened:** balance inquiry ($85 current + "
                    "$45 past due). She disputed a September payment — no record found; offered "
                    "a billing review. Agreed to pay the full $130 via secure text link.\n\n"
                    "**Follow-up:** confirm payment posted."),
        "policies": ("Refunds: duplicate charges are refunded within 5 business days after "
                     "manager approval. Billing disputes: offer a billing-team review — "
                     "never promise an adjustment yourself."),
    },

    "heated": {
        "label": "Heated caller",
        "caller_name": "Mike",
        "steps": [
            {
                "caller": ("Yeah, hi — I was in last Tuesday and I just checked my statement. "
                           "You guys charged me twice for the same visit. This is ridiculous."),
                "sentiment_watch": True,
                "cards": [
                    {"kind": "record", "title": "Visit record",
                     "body": ("**Mike** — visit Tue Oct 6. Two charges of **$75.00** "
                              "(\"Service visit\"). Possible duplicate.")},
                    {"kind": "suggestion", "title": "Suggested reply",
                     "body": ("Mike, I hear you — getting charged twice is frustrating, and I'm "
                              "sorry. Let me pull up that visit right now and we'll sort it out "
                              "together.")},
                ],
            },
            {
                "caller": "I don't want apologies, I want my money back. Today.",
                "sentiment_watch": True,
                "cards": [
                    {"kind": "note", "title": "Coaching",
                     "body": ("Don't promise an amount or a timeline — refunds need **manager "
                              "approval**. Offer the process, not the outcome.")},
                    {"kind": "note", "title": "Policy",
                     "body": ("Duplicate-charge refunds are issued within **5 business days** "
                              "after manager approval.")},
                    {"kind": "suggestion", "title": "Suggested reply",
                     "body": ("You're absolutely right to want this fixed fast. Here's what I'll "
                              "do: I'm flagging this as a duplicate charge for my manager's "
                              "approval right now — that's the fastest way to get your $75 back.")},
                ],
            },
            {
                "caller": "Fine. Then get your manager on the line.",
                "sentiment_watch": True,
                "cards": [
                    {"kind": "brief", "title": "Warm handoff brief — for the manager",
                     "body": ("WARM HANDOFF BRIEF\n"
                              "Caller: Mike\n"
                              "Issue: possible duplicate charge — 2 × $75.00 for the Oct 6 visit\n"
                              "Sentiment: high frustration throughout; acknowledged, did not debate the charge\n"
                              "Offered: duplicate-charge flag for manager approval (refund within 5 business days)\n"
                              "Recommendation: approve/deny the refund — caller expects an answer, don't re-ask what happened.")},
                    {"kind": "suggestion", "title": "Suggested reply",
                     "body": ("Of course — my manager's right here, and I've already briefed them "
                              "on everything, so you won't have to repeat a word.")},
                ],
            },
        ],
        "wrap_up": ("**Caller:** Mike\n\n**What happened:** disputed a duplicate $75 charge from "
                    "the Oct 6 visit. High frustration throughout — acknowledged, avoided "
                    "debating. Warm-transferred to the manager with a full brief.\n\n"
                    "**Follow-up:** confirm refund issued within 5 business days."),
        "policies": ("Refunds need manager approval — never promise an amount or a timeline. "
                     "Duplicate-charge refunds are issued within 5 business days after approval."),
    },
}


# ------------------------------------------------------------------
# Live AI: whisper answers + feedback on Nikki's replies
# ------------------------------------------------------------------
def _payment_history_body(customer_name):
    c = find_customer(customer_name)
    pays = c.get("payments", []) if c else []
    if not c:
        return f"No account found for {customer_name}."
    if not pays:
        return f"No payments on record for {c['name']}."
    lines = [f"**{c['name']}** — payment history:"]
    for p in pays:
        lines.append(f"- {p['date']}: **${p['amount']:.2f}** ({p['method']})")
    return "\n".join(lines)


def _draft_brief(ctx):
    lines = [f"Caller: {ctx['caller_name']}", "Discussed so far:"]
    for role, text in ctx["transcript"]:
        who = "Caller" if role == "caller" else "Nikki"
        snippet = text if len(text) <= 90 else text[:87] + "..."
        lines.append(f"- {who}: '{snippet}'")
    lines.append("Status: draft — review before handing off.")
    return "\n".join(lines)


def handle_whisper(text, ctx):
    """Answer Nikki's private question live.

    ctx keys: scenario, caller_name, step_idx, transcript.
    Returns (kind, title, body) for an AI panel card.
    """
    t = text.lower()
    sc = ctx["scenario"]
    caller = ctx["caller_name"]

    def current_suggestion():
        step = sc["steps"][min(ctx["step_idx"], len(sc["steps"]) - 1)]
        for c in step.get("cards", []):
            if c.get("kind") == "suggestion":
                return c["body"]
        return None

    # balance / invoice
    if re.search(r"\b(balance|bill|invoice|owe|owed|owing)\b", t):
        return ("invoice", "Account lookup", _invoice_body(caller))

    # payment history
    if re.search(r"\b(paid|payment|payments|history)\b", t):
        return ("note", "Payment history", _payment_history_body(caller))

    # handoff brief draft
    if re.search(r"\b(brief|handoff|hand off|transfer|escalate|manager)\b", t):
        return ("brief", "Draft handoff brief", _draft_brief(ctx))

    # suggested reply for this moment
    if re.search(r"\b(suggest|draft|what should i say|what do i say|help me (respond|reply|answer))\b", t):
        s = current_suggestion()
        return ("suggestion", "Suggested reply",
                s or "No draft for this exact moment — speak naturally, I've got the records if you need them.")

    # policy
    if re.search(r"\b(policy|policies|refund|allowed|permitted)\b", t):
        return ("note", "Policy", sc.get("policies", "No policy notes for this scenario."))

    return ("note", "AI assistant",
            "I can pull **balances/invoices**, **payment history**, **policy** notes, "
            "**suggest a reply**, or **draft a handoff brief**. What do you need?")


RISKY_PROMISE = re.compile(
    r"\b(will|'ll|gonna|going to)\s+(refund|give\s+you\s+back)\b"
    r"|\brefund\s+(you|your|it|that|the)\b"
    r"|\b(full|complete|entire)\s+refund\b"
    r"|\bmoney\s+back\b"
    r"|\b(issue|process)\s+(a\s+|the\s+)?refund\b",
    re.IGNORECASE,
)
TIMELINE_WORDS = re.compile(r"\b(today|now|immediately|right away|guarantee|promise)\b",
                            re.IGNORECASE)
EMPATHY_WORDS = re.compile(
    r"\b(sorry|apolog|understand|hear you|frustrat|i see|makes sense)\b",
    re.IGNORECASE)


def feedback_on_reply(nikki_text, caller_text):
    """Live private feedback on Nikki's reply. Returns list of (kind, title, body)."""
    out = []
    t = nikki_text.lower()
    # any committed refund phrasing — unless she's already deferring to approval
    m = RISKY_PROMISE.search(nikki_text)
    if m and "approv" not in t:
        extra = ""
        if TIMELINE_WORDS.search(nikki_text):
            extra = (" Committing to a timeline makes it worse — "
                     "the manager hasn't approved anything yet.")
        out.append(("alert", "Policy flag",
                    f"You just committed to a refund (\"{m.group(0).strip()}\") — refunds need "
                    "**manager approval** first. Offer the approval process "
                    "(\"I'll flag this for approval right now\"), not the outcome." + extra))
    if frustration_level(caller_text) in ("MEDIUM", "HIGH"):
        if EMPATHY_WORDS.search(t):
            out.append(("note", "Nice",
                        "✅ Good — you acknowledged the frustration before moving to facts."))
        else:
            out.append(("note", "Coaching",
                        "The caller is frustrated and your reply doesn't acknowledge it — "
                        "leading with empathy (\"I hear you…\") usually de-escalates faster."))
    return out
