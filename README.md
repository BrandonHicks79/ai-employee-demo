# AI Employee — interactive Streamlit demo

You play the caller in a text chat. The app runs the real pipeline from the
Colab build: DistilBERT intent classifier → entity extraction → rule-based
dialogue manager → escalation detector → warm-handoff brief.

## One-time setup

### 1. Upload the intent model to HuggingFace Hub (from Colab)

The model folder is ~250MB — too big for GitHub. In your `AIEmployee`
notebook, run:

```python
!pip install -q huggingface_hub
from huggingface_hub import HfApi
from getpass import getpass

token = getpass("Paste your HF token (hidden, I never see it): ")
api = HfApi()
api.create_repo("ai-employee-intent", token=token, exist_ok=True)
api.upload_folder(
    folder_path="/content/ai-employee/models/intent_classifier",
    repo_id="brandonhicks79/ai-employee-intent",
    token=token,
)
```

Get a token at huggingface.co/settings/tokens (needs write permission).
`getpass` keeps it out of the notebook — never paste secrets in chat.

### 2. Copy the escalation models into this repo

From Google Drive (`ai-employee/models/escalation/`), download:
- `vectorizer.pkl`
- `model.pkl`

Put them in `models/escalation/` in this repo (create the folder).
They're tiny — safe to commit to GitHub.

### 3. Push to GitHub

```bash
git init
git add .
git commit -m "AI Employee interactive demo"
git branch -M main
git remote add origin https://github.com/brandonhicks79/ai-employee-demo.git
git push -u origin main
```

(Your remote URL may differ — use whatever GitHub gives you.)

### 4. Deploy on Streamlit Community Cloud

1. Go to share.streamlit.io → **New app**
2. Pick the repo, branch `main`, main file `app.py` → **Deploy**
3. First load takes a couple of minutes (model downloads from HF Hub).
4. If your HuggingFace repo is **private**: app dashboard → Settings →
   Secrets, add `HF_TOKEN = "hf_..."`.

### 5. If your HF repo name differs

Edit the `HF_MODEL_ID` constant at the top of `app.py`.

## Notes

- Free Streamlit tier sleeps when idle — the first visitor after a nap
  waits while models reload. Normal.
- Text chat only in v1 — Whisper on free-tier CPU is too slow for voice input.
- The `🔊 Speak AI responses` toggle uses gTTS (needs internet, which
  Streamlit Cloud has).
- To customize for a real business, edit the `HOURS_TEXT` / `PRICING_TEXT`
  constants at the top of `src/dialogue.py` — that's the per-client seam.
