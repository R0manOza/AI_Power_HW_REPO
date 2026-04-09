# HW2 — Individual Audio Pipeline

One-sentence summary: This folder contains a single Python script that turns text into speech with two different voices, transcribes the audio back with Whisper, compares the transcript to the original text, and prints cost and latency estimates for each API call.

## Setup

1. **Python 3.10+**
2. Install dependencies:

```bash
pip install -r hw2/requirements.txt
```

3. **API key** (pick one — **OpenAI is checked first**)

**A — OpenAI (recommended if you have API billing / free trial credits)**  
Add to the **repository root** `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
```

The script then calls **`tts-1`** and **`whisper-1`** on `https://api.openai.com/v1`, same idea as `LAB4/Lab-4/examples/starter-code/`.

**B — OpenRouter only** (used **only when `OPENAI_API_KEY` is not set**)  
```env
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

That path uses OpenRouter’s chat + audio modalities (not the raw `/v1/audio/speech` URL). See `hw2/.env.example` for `OPENROUTER_AUDIO_MODEL`.

`GEMINI_API_KEY` is **not** used by this homework script. See `hw2/.env.example`.

## Run

From the repository root:

```bash
python hw2/hw2-audio-pipeline.py
```

Optional custom text:

```bash
python hw2/hw2-audio-pipeline.py --text "Your shorter passage may cost less TTS."
```

## Expected output (shape)

- Header `=== HW2 Audio Pipeline ===`
- Steps `[1/5]` … `[5/5]` for two TTS voices (`nova`, `alloy`), STT on `audio-output/voice_nova_sample.mp3`, optional WAV demo tone transcription, then side-by-side original vs transcript and **word overlap %**
- Per-call log lines: UTC timestamp, model, latency, input size, estimated USD
- Final **Cost and Latency Summary** for TTS vs STT totals and averages

Generated files (after a successful run):

- `hw2/audio-output/voice_nova_sample.mp3`
- `hw2/audio-output/voice_alloy_sample.mp3`
- `hw2/audio-output/hw2_demo_tone.wav` (short synthetic tone for `.wav` handling)

## Submission checklist

- [ ] `reflection.md` (300+ words) — included
- [ ] `.env.example` — included (no real keys)
- [ ] Do **not** commit `.env`

## References

- Full brief: `LAB4/Lab-4/homework/hw2-individual-audio.md`
- API patterns: `LAB4/Lab-4/guides/audio-api-guide.md`
