"""AI Employee — interactive text demo (Streamlit).

Overflow / After-hours: you play the caller, the real pipeline answers
(DistilBERT intent from HuggingFace Hub -> entities -> dialogue manager ->
escalation detector -> warm-handoff brief).

Assist: you play Nikki (office staff). The app plays a scripted caller and
the AI assists you in a private panel — account lookups, sentiment alerts,
suggested replies, call wrap-up notes.
"""
import io

import streamlit as st
from gtts import gTTS

from src.assist import (ASSIST_SCENARIOS, build_cards, CARD_ICONS,
                      handle_whisper, feedback_on_reply)
from src.dialogue import DialogueManager
from src.escalation import should_escalate

# ------------------------------------------------------------------
# 👉 EDIT THIS after uploading the model to HuggingFace Hub (see README).
# ------------------------------------------------------------------
HF_MODEL_ID = "brandonhicks79/ai-employee-intent"

st.set_page_config(page_title="AI Employee — Live Demo", page_icon="📞", layout="centered")

GREETINGS = {
    "Overflow": ("Thanks for calling — all of our staff are on other lines right now, "
                 "so you're with me, the AI assistant. How can I help?"),
    "After-hours": ("You've reached us after hours — I'm the AI front desk tonight. "
                    "I can book, reschedule, or take a message. What do you need?"),
    "Assist": ("You're Nikki, the office staff. The caller is on the line — "
               "the AI assists you privately in the panel on the right."),
}

# ---------------------------------------------------------------- state
if "dm" not in st.session_state:
    st.session_state.dm = None
if "turns" not in st.session_state:
    st.session_state.turns = []          # (role, text) — role: caller | ai | brief
if "mode" not in st.session_state:
    st.session_state.mode = "Overflow"

ASSIST_KEYS = ("assist_scenario", "assist_step", "assist_transcript",
               "assist_cards", "assist_over")


def reset_call():
    # every fresh call opens with the current mode's greeting
    st.session_state.turns = [("ai", GREETINGS[st.session_state.mode])]
    if st.session_state.dm is not None:
        st.session_state.dm.reset()
    for k in ASSIST_KEYS:
        st.session_state.pop(k, None)


# ---------------------------------------------------------------- assist mode
def start_assist_call(scenario_id):
    sc = ASSIST_SCENARIOS[scenario_id]
    step0 = sc["steps"][0]
    st.session_state.assist_scenario = scenario_id
    st.session_state.assist_step = 0
    st.session_state.assist_transcript = [("caller", step0["caller"])]
    st.session_state.assist_cards = build_cards(step0, step0["caller"])
    st.session_state.assist_over = False


def render_assist_card(card):
    icon = CARD_ICONS.get(card["kind"], "•")
    with st.container(border=True):
        st.markdown(f"### {icon} {card['title']}")
        if card["kind"] == "brief":
            st.code(card["body"], language=None)
        else:
            st.markdown(card["body"])


def render_assist_mode():
    st.caption("You are **Nikki**, the office staff. The AI panel on the right is "
               "private — the caller can't see it.")

    ids = list(ASSIST_SCENARIOS.keys())
    labels = [ASSIST_SCENARIOS[i]["label"] for i in ids]
    sel = st.radio("Scenario", labels, horizontal=True)
    sid = ids[labels.index(sel)]
    if st.session_state.get("assist_scenario") != sid:
        start_assist_call(sid)

    sc = ASSIST_SCENARIOS[st.session_state.assist_scenario]
    col_call, col_ai = st.columns([3, 2])

    with col_call:
        st.subheader("📞 Live call")
        for role, text in st.session_state.assist_transcript:
            who = (f"**Caller — {sc['caller_name']}:**" if role == "caller"
                   else "**Nikki (you):**")
            with st.chat_message("user" if role == "caller" else "assistant"):
                st.markdown(f"{who} {text}")
        if not st.session_state.get("assist_over"):
            nikki_text = st.chat_input("Reply as Nikki…", key="nikki_input")
            if nikki_text:
                step = sc["steps"][st.session_state.assist_step]
                st.session_state.assist_transcript.append(("nikki", nikki_text))
                # live AI feedback on Nikki's actual reply
                for kind, title, body in feedback_on_reply(nikki_text, step["caller"]):
                    st.session_state.assist_cards.append(
                        {"kind": kind, "title": title, "body": body})
                nxt = st.session_state.assist_step + 1
                if nxt < len(sc["steps"]):
                    st.session_state.assist_step = nxt
                    step = sc["steps"][nxt]
                    st.session_state.assist_transcript.append(("caller", step["caller"]))
                    st.session_state.assist_cards.extend(build_cards(step, step["caller"]))
                else:
                    st.session_state.assist_over = True
                    st.session_state.assist_cards.append(
                        {"kind": "wrapup", "title": "Call wrap-up — AI-generated notes",
                         "body": sc["wrap_up"]})
                st.rerun()
        else:
            st.info("Call ended — pick another scenario above to run it again.")

    with col_ai:
        st.subheader("🤖 AI assistant")
        st.caption("Private — the caller can't see this.")
        for card in st.session_state.assist_cards:
            render_assist_card(card)
        st.divider()
        with st.form("whisper_form", clear_on_submit=True):
            whisper = st.text_input("Ask the AI privately",
                                    placeholder="e.g. what's her balance?")
            submitted = st.form_submit_button("Ask")
        if submitted and whisper:
            ctx = {
                "scenario": sc,
                "caller_name": sc["caller_name"],
                "step_idx": st.session_state.assist_step,
                "transcript": list(st.session_state.assist_transcript),
            }
            kind, title, body = handle_whisper(whisper, ctx)
            st.session_state.assist_cards.append(
                {"kind": kind, "title": title, "body": body})
            st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Demo controls")
    mode = st.radio("Mode", list(GREETINGS),
                    index=list(GREETINGS).index(st.session_state.mode))
    if st.session_state.mode == "Assist":
        voice = False  # AI whispers to Nikki here — nothing to speak aloud
    else:
        voice = st.checkbox("🔊 Speak AI responses", value=True)
    if st.button("↺ Restart call"):
        reset_call()
        st.rerun()
    if mode != st.session_state.mode:
        st.session_state.mode = mode
        reset_call()
        st.rerun()
    with st.expander("How this works"):
        st.markdown(
            "1. **Intent** — fine-tuned DistilBERT classifies what the caller wants\n"
            "2. **Entities** — dates, times, names pulled out (parsed, not trained)\n"
            "3. **Dialogue** — rule-based manager decides the next step\n"
            "4. **Escalation** — binary classifier watches for handoff triggers\n"
            "5. **Handoff** — warm brief for the human, no re-asking\n"
            "6. **Account lookup** — demo customer data (stand-in for a real CRM/billing integration)"
        )

# ---------------------------------------------------------------- header
st.title("📞 AI Employee — live demo")
if st.session_state.mode == "Assist":
    st.caption("SIMULATED DEMO · assist mode · you play Nikki (office staff)")
else:
    st.caption("SIMULATED DEMO · generic front-desk build · you play the caller")
st.divider()

# ---------------------------------------------------------------- model load (once)
if st.session_state.dm is None:
    with st.spinner("Loading AI models — first visit takes a minute or two…"):
        try:
            token = st.secrets["HF_TOKEN"]
        except Exception:
            token = None
        st.session_state.dm = DialogueManager(HF_MODEL_ID, hf_token=token)
        reset_call()

dm = st.session_state.dm


def speak(text):
    tts = gTTS(text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    st.audio(buf.getvalue(), format="audio/mp3")


# ---------------------------------------------------------------- main
if st.session_state.mode == "Assist":
    render_assist_mode()
else:
    # transcript
    for role, text in st.session_state.turns:
        if role == "brief":
            with st.chat_message("assistant"):
                with st.container(border=True):
                    st.markdown("#### 🔥 Warm handoff brief")
                    st.code(text, language=None)
        else:
            with st.chat_message("user" if role == "caller" else "assistant"):
                st.markdown(text)

    # caller input
    caller_text = st.chat_input("Type as the caller…")
    if caller_text:
        st.session_state.turns.append(("caller", caller_text))
        with st.chat_message("user"):
            st.markdown(caller_text)

        # while slot-filling, saying goodbye, or asking about billing, the
        # rule-based flow owns the turn — skip the ML escalation check so
        # fragments like "Tuesday at 3pm" don't misfire it
        if (dm.slots.get("awaiting") or dm.is_conversation_end(caller_text)
                or dm.is_billing_request(caller_text)):
            esc_ml = False
        else:
            esc_ml, _prob = should_escalate(caller_text)
        reply, escalated_dm, brief = dm.handle(caller_text)
        escalated = esc_ml or escalated_dm

        with st.chat_message("assistant"):
            if escalated:
                st.markdown(reply)
                with st.container(border=True):
                    st.markdown("#### 🔥 Warm handoff brief")
                    st.code(brief, language=None)
                st.session_state.turns.append(("ai", reply))
                st.session_state.turns.append(("brief", brief))
                if voice:
                    speak(reply)
            else:
                st.markdown(reply)
                st.session_state.turns.append(("ai", reply))
                if voice:
                    speak(reply)
