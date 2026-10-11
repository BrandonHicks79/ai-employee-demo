"""Dialogue manager: DistilBERT intent + entities -> rule-based next step.

The intent model loads from HuggingFace Hub (too big for GitHub).
Same rules as the Colab build. Returns (response_text, escalated, brief).
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

    def predict_intent(self, text):
        enc = self.tok(text, truncation=True, padding=True,
                       return_tensors="pt", max_length=64)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[0]
            idx = int(probs.argmax())
            return self.labels[idx], float(probs[idx])

    def handle(self, text):
        """Returns (response_text, escalated: bool, brief: str | None)."""
        # 0. explicit human request — skip the classifier, escalate confidently
        if HUMAN_REQUEST.search(text):
            ents = extract_entities(text)
            self.history.append({"user": text, "intent": "handoff_request",
                                 "conf": 1.0, "entities": ents})
            return HANDOFF_LINE, True, summarize_handoff(self.history)

        intent, conf = self.predict_intent(text)
        ents = extract_entities(text)
        self.history.append({"user": text, "intent": intent,
                             "conf": conf, "entities": ents})

        def escalate():
            return HANDOFF_LINE, True, summarize_handoff(self.history)

        if conf < CONFIDENCE_THRESHOLD:
            return escalate()

        if intent == "booking":
            dts = ents["datetimes"]
            name = ents["name"]
            if not dts:
                return ("Sure, I can book that. What day and time works for you?",
                        False, None)
            if not name and not self.slots.get("name"):
                self.slots["pending_booking"] = dts[0]
                return (f"Got it, {dts[0]['text']} — what's your name?",
                        False, None)
            nm = name or self.slots.get("name")
            if name:
                self.slots["name"] = name
            dt_txt = (dts[0]["text"] if dts
                      else self.slots.get("pending_booking", {}).get("text", "that time"))
            self.slots.pop("pending_booking", None)
            return (f"Perfect{', ' + nm if nm else ''} — you're booked for {dt_txt}. "
                    "Anything else?", False, None)

        if intent == "reschedule":
            dts = ents["datetimes"]
            if not dts:
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
