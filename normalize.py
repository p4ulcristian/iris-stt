"""Qwen text normalization for TTS - conversational style."""

import re
import requests

LLM_URL = "http://localhost:11434/v1/chat/completions"
LLM_MODEL = "qwen"
LLM_TIMEOUT = 5

# Strip emojis with regex
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)

SYSTEM_PROMPT = """You convert text into natural spoken conversation. Keep ALL information but make it sound like you're casually talking to a friend.

RULES:
- Add natural connectors: "and", "so", "also", "basically", "which is"
- Add verbs to make it flow: "has", "is", "there's", "we've got"
- Break long lists into natural sentences
- kebab-case/camelCase → spoken words (flex-workspace → flex workspace, getUserById → get user by ID)
- Math → words: 54 + 67 = 121 → 54 plus 67 equals 121
- ~ → about, × → times
- Remove markdown formatting (**, *, `, #, >, |) but keep the words
- NEVER translate - if text is in Hungarian/Spanish/etc, keep it in that language
- Sound natural and friendly, not robotic

EXAMPLES:

IN: **`suite.new`** (3 workflows):
- `flex.workspace-creation`
- `flex.reservation-approval-flow`
- `flex.process-crud`
OUT: So suite.new has 3 workflows. There's flex workspace creation, reservation approval flow, and process crud.

IN: Completed in ~2 seconds, 1 turn. Cost was ~$0.016 vs gold's ~$0.040.
OUT: Completed in about 2 seconds, just one turn. Cost was about 1.6 cents compared to gold's 4 cents.

IN: | Name | Status |
|------|--------|
| Gold | Done |
| Jade | Pending |
OUT: So we've got Gold which is done, and Jade which is still pending.

IN: Úgy tűnik, hogy véletlenül küldted el ezt.
OUT: Úgy tűnik, hogy véletlenül küldted el ezt.

Output ONLY the conversational version, nothing else."""


def normalize_for_tts(text: str) -> str:
    """Normalize text for TTS - conversational style."""
    if not text or not text.strip():
        return text

    # Pre-strip emojis
    clean = EMOJI_PATTERN.sub("", text).strip()

    # Skip very short simple text
    if len(clean) < 15 and not re.search(r'[*`#|~×←→\-]', clean):
        return clean

    try:
        resp = requests.post(
            LLM_URL,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": clean},
                ],
                "temperature": 0.1,  # Tiny bit of variation for natural speech
                "max_tokens": 2048,
            },
            timeout=LLM_TIMEOUT,
        )

        if resp.status_code == 200:
            data = resp.json()
            result = data["choices"][0]["message"]["content"].strip()

            if result and len(result) > 5:
                print(f"[Normalize] {len(text)}→{len(result)} chars", flush=True)
                print(f"[Normalize] IN:  {text[:80]!r}{'...' if len(text) > 80 else ''}", flush=True)
                print(f"[Normalize] OUT: {result[:80]!r}{'...' if len(result) > 80 else ''}", flush=True)
                return result

    except requests.exceptions.Timeout:
        print(f"[Normalize] Timeout, using original", flush=True)
    except Exception as e:
        print(f"[Normalize] Error: {e}", flush=True)

    # Fallback: basic cleanup
    fallback = clean
    fallback = re.sub(r'\*\*([^*]+)\*\*', r'\1', fallback)
    fallback = re.sub(r'`([^`]+)`', r'\1', fallback)
    fallback = re.sub(r'\n{2,}', '. ', fallback)
    return fallback
