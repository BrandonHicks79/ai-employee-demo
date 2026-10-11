"""AI Employee — interactive text demo (Streamlit).

You play the caller. The real pipeline answers:
DistilBERT intent (HuggingFace Hub) -> entities -> dialogue manager ->
escalation detector -> warm-handoff brief.
"""
import io

import streamlit as st
from gtts import gTTS

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
    "Assist": ("I'm assisting alongside our team today — I'll pull up info and draft "
               "responses live. Go ahead, talk to me like a caller."),
}

# ---------------------------------------------------------------- state
if "dm" not in st.session_state:
    st.session_state.dm = None
if "turns" not in st.session_state:
    st.session_state.turns = []          # (role, text) — role: caller | ai | brief
if "mode" not in st.session_state:
    st.session_state.mode = "Overflow"


def reset_call():
    # every fresh call opens with the current mode's greeting
    st.session_state.turns = [("ai", GREETINGS[st.session_state.mode])]
    if st.session_state.dm is not None:
        st.session_state.dm.reset()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Demo controls")
    mode = st.radio("Mode", list(GREETINGS),
                    index=list(GREETINGS).index(st.session_state.mode))
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
            "5. **Handoff** — warm brief for the human, no re-asking"
        )

# ---------------------------------------------------------------- header
st.title("📞 AI Employee — live demo")
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


# ---------------------------------------------------------------- transcript
for role, text in st.session_state.turns:
    if role == "brief":
        with st.chat_message("assistant"):
            with st.container(border=True):
                st.markdown("#### 🔥 Warm handoff brief")
                st.code(text, language=None)
    else:
        with st.chat_message("user" if role == "caller" else "assistant"):
            st.markdown(text)

# ---------------------------------------------------------------- caller input
caller_text = st.chat_input("Type as the caller…")
if caller_text:
    st.session_state.turns.append(("caller", caller_text))
    with st.chat_message("user"):
        st.markdown(caller_text)

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
