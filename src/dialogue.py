"""Dialogue manager: DistilBERT intent + entities -> rule-based next step.

The intent model loads from HuggingFace Hub (too big for GitHub).
Returns (response_text, escalated, brief).

Key design point: the manager has short-term memory (slots). When it asks a
follow-up question ("what day and time?"), the next message is interpreted
as the answer — NOT re-classified from scratch. That's slot filling.

Deterministic fast paths (no classifier involved):
  - explicit "talk to a human" -> warm handoff
  - closing phrases ("that's all", "bye") -> graceful end of call
"""
import json
import re

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import hf_hub_download

from .entities import extract_entities
from .handoff import summarize_handoff

# ------------------------------------------------------------------
# 👉 CUSTOMIZE PER BUSINESS — this is the per-client seam.
# ------------------------------------------------------------------
HOURS_TEXT = "We're open 9-5 weekdays, 9-1 Saturdays."
PRICING_TEXT = "Pricing depends on the service — consultations are free."
CONFIDENCE_THRESHOLD = 0.6

# Explicit "let me talk to a human" requests bypass the classifier entirely.
# Deterministic — no ML uncertainty on the clearest escalation signal there is.
HUMAN_REQUEST = re.compile(
    r"\b(speak|talk)\s+to\s+(someone|a human|a person|a manager|an agent)\b"
    r"|\bget\s+me\s+(a )?(human|person|manager|someone)\b"
    r"|\bbring\s+me\s+(a )?(human|person|manager|someone)\b"
    r"|\bi\s+want\s+(a )?(human|manager)\b"
    r"|\breal\s+person\b",
    re.IGNORECASE,
)

# Closing phrases — end the call gracefully, never escalate.
GOODBYE = re.compile(
    r"("
    r"\bgood-?bye\b|\bbye\b|"
    r"\bthat'?s (all|it|everything)\b|"
    r"\bthat (is|will be) all\b|"
    r"\bno,? that('ll| will be|s)? all\b|"
    r"\bi'?m (all )?done\b|"
    r"\b(thanks|thank you)(,? bye)?\s*$"
    r")",
    re.IGNORECASE,
)

CANCEL_WORDS = re.compile(r"\b(never mind|nevermind|cancel|no thanks|forget it)\b",
                          re.IGNORECASE)

# Caller-facing handoff line. Internal reasons (e.g. confidence scores) stay
# in history for debugging — the caller never sees them.
HANDOFF_LINE = "Let me bring in someone to help."


class DialogueManager:
    def __init__(self, model_id, hf_token=None):
        self.tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id, token=hf_token)
        self.model.eval()
        label_path = hf_hub_download(repo_id=model_id, filename="label_map.json", token=hf_token)
        with open(label_path) as f:
            labels = json.load(f)
        self.labels = {int(k): v for k, v in labels.items()}
        self.history = []
        self.slots = {}

    def reset(self):
        self.history = []
        self.slots = {}

    def is_conversation_end(self, text):
        return bool(GOODBYE.search(text))

    def predict_intent(self, text):
        enc = self.tok(text, truncation=True, padding=True,
                       return_tensors="pt", max_length=64)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
            return self.labels[idx], float(probs[idx])

    def _log(self, text, intent, conf, ents):
        self.history.append({"user": text, "intent": intent,
                             "conf": conf, "entities": ents})

    def handle(self, text):
        """Returns (response_text, escalated: bool, brief: str | None)."""
        ents = extract_entities(text)

        # 0. explicit human request — skip everything, escalate confidently
        if HUMAN_REQUEST.search(text):
            self._log(text, "handoff_request", 1.0, ents)
            self.slots.pop("awaiting", None)
            return HANDOFF_LINE, True, summarize_handoff(self.history)

        # 0b. closing phrase — end the call gracefully, never escalate
        if self.is_conversation_end(text):
            self._log(text, "goodbye", 1.0, ents)
            self.slots.pop("awaiting", None)
            nm = self.slots.get("name")
            return (f"You're welcome{', ' + nm if nm else ''}! Have a great day.",
                    False, None)

        # 1. slot filling — the caller is answering our follow-up question,
        #    so don't re-classify; just fill the slot.
        awaiting = self.slots.get("awaiting")

        if awaiting == "datetime":
            dts = ents["datetimes"]
            if not dts:
                return ("I didn't catch a day or time — what day and time works for you?",
                        False, None)
            self._log(text, "booking", 1.0, ents)
            self.slots["pending_booking"] = dts[0]
            nm = ents["name"]
            if nm:  # gave name in the same message — skip asking
                self.slots["name"] = nm
                self.slots.pop("awaiting", None)
                self.slots.pop("pending_booking", None)
                return (f"Perfect, {nm} — you're booked for {dts[0]['text']}. "
                        "Anything else?", False, None)
            self.slots["awaiting"] = "name"
            return (f"Got it, {dts[0]['text']} — what's your name?", False, None)

        if awaiting == "name":
            if CANCEL_WORDS.search(text):
                self.slots.pop("awaiting", None)
                self.slots.pop("pending_booking", None)
                return ("No problem — we'll skip the booking. Anything else?",
                        False, None)
            nm = ents["name"] or text.strip()
            ents["name"] = nm  # so the handoff brief picks it up
            pending = self.slots.pop("pending_booking", {})
            self.slots.pop("awaiting", None)
            self.slots["name"] = nm
            self._log(text, "booking", 1.0, ents)
            return (f"Perfect, {nm} — you're booked for {pending.get('text', 'that time')}. "
                    "Anything else?", False, None)

        if awaiting == "reschedule_datetime":
            dts = ents["datetimes"]
            if not dts:
                return ("I didn't catch a day or time — what new day/time works for you?",
                        False, None)
            self._log(text, "reschedule", 1.0, ents)
            self.slots.pop("awaiting", None)
            return (f"Got it, moving your appointment to {dts[0]['text']}. "
                    "Anything else?", False, None)

        # 2. fresh turn — classify intent as usual
        intent, conf = self.predict_intent(text)
        self._log(text, intent, conf, ents)

        def escalate():
            self.slots.pop("awaiting", None)
            return HANDOFF_LINE, True, summarize_handoff(self.history)

        if conf < CONFIDENCE_THRESHOLD:
            return escalate()

        if intent == "booking":
            dts = ents["datetimes"]
            name = ents["name"]
            if dts:
                self.slots["pending_booking"] = dts[0]
            if name:
                self.slots["name"] = name
            pending = self.slots.get("pending_booking")
            nm = self.slots.get("name")
            if not pending:
                self.slots["awaiting"] = "datetime"
                return ("Sure, I can book that. What day and time works for you?",
                        False, None)
            if not nm:
                self.slots["awaiting"] = "name"
                return (f"Got it, {pending['text']} — what's your name?",
                        False, None)
            self.slots.pop("pending_booking", None)
            self.slots.pop("awaiting", None)
            return (f"Perfect, {nm} — you're booked for {pending['text']}. "
                    "Anything else?", False, None)

        if intent == "reschedule":
            dts = ents["datetimes"]
            if not dts:
                self.slots["awaiting"] = "reschedule_datetime"
                return ("I can help move that. What new day/time works for you?",
                        False, None)
            return (f"Got it, moving your appointment to {dts[0]['text']}. "
                    "Anything else?", False, None)

        if intent == "hours":
            return (HOURS_TEXT + " Anything else?", False, None)

        if intent == "pricing":
            return (PRICING_TEXT + " What did you need done?", False, None)

        if intent == "complaint":
            return escalate()

        return ("I didn't quite catch that — can you say it another way? "
                "Or I can bring in someone to help.", False, None)
