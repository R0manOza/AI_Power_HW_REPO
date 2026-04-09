# HW2 Data Governance Reflection — Audio Pipeline

**Course:** CS-AI-2025 — Building AI-Powered Applications | Spring 2026  
**Assignment:** HW2 Individual Audio Pipeline  
**Pipeline:** `hw2/hw2-audio-pipeline.py` (text → TTS → `audio-output/voice_nova_sample.mp3` / `voice_alloy_sample.mp3` → Whisper STT → transcript comparison)

This reflection references the default passage used by the script (224 characters). At the stated HW2 rates, each `tts-1` call costs about \((224 / 1000) \times \$0.015 \approx \$0.00336\). A full run performs two TTS calls plus at least one STT call on the generated MP3; Whisper billing uses audio duration from the API’s `verbose_json` response (typically on the order of a few to tens of seconds for short passages), at \(\$0.006\) per minute. File names above are the ones the grader expects in the submission tree.

## 1. Consent (real user audio instead of synthetic TTS output)

If this pipeline processed **real user audio** (for example, a voice note describing car symptoms), consent could not be a buried checkbox in settings. A practical approach is a **just-in-time screen** that appears the first time the user taps “Record” or “Transcribe,” with plain-language text such as: “We will send your audio to our transcription provider to convert speech to text. Audio is processed to produce a transcript; we do not use it for unrelated purposes. You can stop recording at any time.” The screen should name the third party (e.g., OpenAI via OpenRouter), link to the privacy policy, and offer **Decline** (disabling the feature but keeping the rest of the app usable, per the checklist in `Lab-4/templates/data-governance-checklist.md`). **Revocation** should allow the user to delete stored audio and transcripts from account settings and to withdraw consent going forward, with deletion propagated to any retained copies where technically feasible.

## 2. Retention (three scenarios)

Retention should match **purpose** and **risk**:

**(a) Study app (audio lessons):** If the audio is **generated TTS** and not personalized, retention can be short—cache for playback performance (hours to days) then delete, unless the user explicitly saves a lesson. Minimizing stored audio reduces cost and breach impact.

**(b) Customer service transcription:** Retain **raw audio** only as long as needed for QA or disputes (often **30–90 days** with policy), with **transcripts** kept longer if tied to tickets. Delete audio early when the transcript is authoritative and disputes are unlikely.

**(c) Medical intake:** Assume **high sensitivity**. Default to **no retention of raw audio** after transcription unless a clinician workflow and legal basis require it; if retained, use **strict encryption, access control, and short retention** with documented exceptions. Transcripts may still be sensitive but are easier to redact than voiceprints.

## 3. PII and risks in audio beyond the words

Audio carries **voice biometrics** (speaker identification), **accent and language markers**, and **paralinguistics** (stress, pace) that can imply emotional or health-related states. **Background sound** may capture other people or locations without their consent. **File metadata** (timestamps, device IDs, geotags) can identify individuals even when spoken content is mundane. Unlike plain text, raw audio is harder to redact and easier to misuse for profiling; governance should prefer **deleting raw audio** once a transcript exists, strip metadata before storage, and avoid secondary uses (e.g., emotion scoring) without explicit opt-in.

## 4. Capstone (Pocket Mechanics / car helper) and governance

Our team treats audio as **enhancement**, not core: users primarily photograph the engine bay and read guidance, but **hands-free** questions (“Where is the oil cap?”) could help while working. Governance implications: **consent before microphone use**, **warning about background conversations** in a garage, and **clear disclosure** that audio is sent to a third-party API. If we **never ship audio**, governance is simpler (no voiceprints in scope); if we add it later, we would add **retention limits**, **deletion on request**, and **minimization** (short clips, no continuous streaming without strong justification). The HW2 pipeline’s logs (timestamps, model, latency, cost) are useful for engineering but must not accidentally store **secrets** or **raw user audio** in shared logs in production.

---


