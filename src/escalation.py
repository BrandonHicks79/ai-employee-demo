"""Escalation detector: TF-IDF + logistic regression, binary classifier.

Loads the pickles committed in models/escalation/ (tiny files, safe for GitHub).
"""
import os
import pickle

_BASE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "escalation")
_vec = None
_clf = None


def _load():
    global _vec, _clf
    if _vec is None:
        with open(os.path.join(_BASE, "vectorizer.pkl"), "rb") as f:
            _vec = pickle.load(f)
        with open(os.path.join(_BASE, "model.pkl"), "rb") as f:
            _clf = pickle.load(f)


def should_escalate(text, threshold=0.5):
    """Returns (escalate: bool, probability: float)."""
    _load()
    prob = _clf.predict_proba(_vec.transform([text]))[0][1]
    return prob >= threshold, float(prob)
