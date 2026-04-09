"""
HW2: Individual Audio Pipeline (Lab 4)
Text -> TTS (MP3) -> STT -> compare + cost/latency logging.

API routing (OPENAI_API_KEY wins if both are set):
  - If OPENAI_API_KEY is set: uses OpenAI's native /v1/audio/speech and /v1/audio/transcriptions
    (matches the course starter scripts exactly).
  - Else uses OPENROUTER_API_KEY: OpenRouter does NOT expose those audio paths; we use their
    documented chat/completions flow (streaming audio output + input_audio for transcription).
    See: https://openrouter.ai/docs/guides/overview/multimodal/audio

Run: python hw2/hw2-audio-pipeline.py [--text "Your passage here"]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI

# ── Paths ────────────────────────────────────────────────────────────────────
HW2_DIR = Path(__file__).resolve().parent
REPO_ROOT = HW2_DIR.parent
OUTPUT_DIR = HW2_DIR / "audio-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STT_SUPPORTED = {".mp3", ".mp4", ".wav", ".webm", ".m4a", ".mpeg", ".mpga"}
STT_FORMAT_MAP = {
    ".mp3": "mp3",
    ".mp4": "mp4",
    ".wav": "wav",
    ".webm": "webm",
    ".m4a": "m4a",
    ".mpeg": "mpeg",
    ".mpga": "mpga",
}
MAX_AUDIO_MB = 25

TTS_COST_PER_1K_CHARS = 0.015
STT_COST_PER_MINUTE = 0.006

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# Must support output_modalities including audio on OpenRouter (gpt-4o-mini does not).
# See https://openrouter.ai/models?q=audio
DEFAULT_OPENROUTER_AUDIO_MODEL = "openai/gpt-audio-mini"

DEFAULT_TEXT = (
    "Machine learning models learn patterns from data. "
    "They generalize from training examples to make predictions "
    "on new, unseen inputs. The quality of the training data "
    "directly determines the quality of the model's predictions."
)


def load_env() -> None:
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(HW2_DIR / ".env")


def openrouter_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/",
        "X-Title": "HW2 Audio Pipeline",
    }


def resolve_backend() -> tuple[str, OpenAI | None, str]:
    """
    Returns (mode, openai_client_or_none, openrouter_model_id).
    mode is 'openai_native' or 'openrouter_chat'.
    """
    oa = (os.getenv("OPENAI_API_KEY") or "").strip()
    if oa:
        return "openai_native", OpenAI(api_key=oa, base_url="https://api.openai.com/v1"), ""

    or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if not or_key:
        print(
            "ERROR: No API key found for audio.\n"
            f"Add to repo root .env: {REPO_ROOT / '.env'}\n\n"
            "Recommended (matches Lab 4 starters — uses your OpenAI billing):\n"
            "  OPENAI_API_KEY=sk-...\n\n"
            "Alternative (OpenRouter credits / course org key):\n"
            "  OPENROUTER_API_KEY=sk-or-v1-...\n\n"
            "See hw2/.env.example"
        )
        sys.exit(1)
    model = (os.getenv("OPENROUTER_AUDIO_MODEL") or DEFAULT_OPENROUTER_AUDIO_MODEL).strip()
    return "openrouter_chat", None, model


def log_call(
    logs: list[dict[str, Any]],
    *,
    call_type: str,
    model: str,
    latency_s: float,
    input_size: str,
    cost_usd: float,
    extra: dict[str, Any] | None = None,
) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "call_type": call_type,
        "model": model,
        "latency_seconds": round(latency_s, 3),
        "input_size": input_size,
        "estimated_cost_usd": round(cost_usd, 6),
        **(extra or {}),
    }
    logs.append(entry)
    print(
        f"  [log] {entry['timestamp']} | {call_type} | {model} | "
        f"latency={entry['latency_seconds']}s | input={input_size} | cost≈${entry['estimated_cost_usd']:.6f}"
    )


def api_call_with_retry(fn, *, retries: int = 2, pause_s: float = 2.0):
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return fn()
        except (APIConnectionError, APITimeoutError, TimeoutError, requests.RequestException) as e:
            last_err = e
            if attempt < retries - 1:
                print(
                    f"  Connection/timeout error (attempt {attempt + 1}/{retries}): {e}\n"
                    f"  Retrying in {pause_s:.0f}s..."
                )
                time.sleep(pause_s)
            else:
                print(f"  ERROR: API call failed after {retries} attempts: {e}")
                raise
    assert last_err is not None
    raise last_err


# ── OpenAI native (course / Lab starter pattern) ─────────────────────────────


def tts_openai_native(
    client: OpenAI, text: str, voice: str, out_path: Path, logs: list[dict[str, Any]]
) -> dict[str, Any]:
    print(f"\n[ TTS / OpenAI ] voice={voice} -> {out_path.name}")

    def _do():
        start = time.perf_counter()
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format="mp3",
        )
        response.stream_to_file(str(out_path))
        return time.perf_counter() - start

    elapsed = api_call_with_retry(_do)
    size_b = out_path.stat().st_size
    cost = (len(text) / 1000.0) * TTS_COST_PER_1K_CHARS
    log_call(
        logs,
        call_type="tts",
        model="tts-1",
        latency_s=elapsed,
        input_size=f"{len(text)} chars",
        cost_usd=cost,
        extra={"voice": voice, "output_file": out_path.name, "file_size_bytes": size_b},
    )
    print(f"  Generated in {elapsed:.2f}s | {size_b / 1024:.1f} KB | cost ~${cost:.4f}")
    return {"path": out_path, "latency_s": elapsed, "cost": cost, "size_bytes": size_b}


def stt_openai_native(
    client: OpenAI, audio_path: Path, logs: list[dict[str, Any]]
) -> dict[str, Any]:
    validate_audio_path(audio_path)
    print(f"\n[ STT / OpenAI ] {audio_path.name}")

    def _do():
        start = time.perf_counter()
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                language="en",
            )
        return transcript, time.perf_counter() - start

    transcript, elapsed = api_call_with_retry(_do)
    text = transcript.text
    duration = float(getattr(transcript, "duration", None) or 0.0)
    language = getattr(transcript, "language", "unknown")
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    cost = (duration / 60.0) * STT_COST_PER_MINUTE if duration else 0.0
    log_call(
        logs,
        call_type="stt",
        model="whisper-1",
        latency_s=elapsed,
        input_size=f"{size_mb:.3f} MB, {duration:.1f}s",
        cost_usd=cost,
        extra={"file": audio_path.name},
    )
    print(f"  Transcript: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"  {elapsed:.2f}s | lang={language} | ~${cost:.4f}")
    return {"text": text, "latency_s": elapsed, "cost": cost, "duration_s": duration, "language": language}


# ── OpenRouter (chat completions — audio modalities) ─────────────────────────


def openrouter_tts_stream_to_mp3(
    api_key: str, model: str, text: str, voice: str, out_path: Path
) -> float:
    """Stream TTS via OpenRouter; returns elapsed seconds."""
    prompt = (
        "Read the following text aloud exactly as written. "
        "Do not add introduction or closing remarks.\n\n"
        f"{text}"
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "modalities": ["text", "audio"],
        "audio": {"voice": voice, "format": "mp3"},
        "stream": True,
    }
    start = time.perf_counter()
    audio_b64_parts: list[str] = []

    with requests.post(
        OPENROUTER_URL,
        headers=openrouter_headers(api_key),
        json=payload,
        stream=True,
        timeout=180,
    ) as resp:
        if resp.status_code >= 400:
            body = resp.text[:500]
            raise RuntimeError(f"OpenRouter TTS HTTP {resp.status_code}: {body}")

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            err = chunk.get("error")
            if err:
                raise RuntimeError(f"OpenRouter error: {err}")
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                audio = delta.get("audio") or {}
                if audio.get("data"):
                    audio_b64_parts.append(audio["data"])

    elapsed = time.perf_counter() - start
    if not audio_b64_parts:
        raise RuntimeError(
            "OpenRouter returned no audio chunks. Try OPENROUTER_AUDIO_MODEL=openai/gpt-audio-mini "
            "or set OPENAI_API_KEY for native TTS."
        )
    raw = base64.b64decode("".join(audio_b64_parts))
    out_path.write_bytes(raw)
    return elapsed


def tts_openrouter(
    api_key: str, model: str, text: str, voice: str, out_path: Path, logs: list[dict[str, Any]]
) -> dict[str, Any]:
    print(f"\n[ TTS / OpenRouter ] model={model} voice={voice} -> {out_path.name}")

    def _do():
        return openrouter_tts_stream_to_mp3(api_key, model, text, voice, out_path)

    elapsed = api_call_with_retry(_do)
    size_b = out_path.stat().st_size
    cost = (len(text) / 1000.0) * TTS_COST_PER_1K_CHARS
    log_call(
        logs,
        call_type="tts",
        model=model,
        latency_s=elapsed,
        input_size=f"{len(text)} chars",
        cost_usd=cost,
        extra={"voice": voice, "output_file": out_path.name, "file_size_bytes": size_b},
    )
    print(f"  Generated in {elapsed:.2f}s | {size_b / 1024:.1f} KB | cost ~${cost:.4f} (char-based est.)")
    return {"path": out_path, "latency_s": elapsed, "cost": cost, "size_bytes": size_b}


def message_text_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text") or "")
        return "".join(parts).strip()
    return ""


def openrouter_stt_chat(api_key: str, model: str, audio_path: Path) -> tuple[str, float]:
    validate_audio_path(audio_path)
    fmt = STT_FORMAT_MAP.get(audio_path.suffix.lower())
    if not fmt:
        raise ValueError(f"Unsupported format for OpenRouter input_audio: {audio_path.suffix}")
    b64 = base64.b64encode(audio_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Transcribe the audio verbatim in English. "
                            "Output only the spoken words, no quotes or labels."
                        ),
                    },
                    {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
                ],
            }
        ],
    }
    start = time.perf_counter()
    r = requests.post(
        OPENROUTER_URL,
        headers=openrouter_headers(api_key),
        json=payload,
        timeout=180,
    )
    elapsed = time.perf_counter() - start
    if r.status_code >= 400:
        raise RuntimeError(f"OpenRouter STT HTTP {r.status_code}: {r.text[:500]}")
    data = r.json()
    err = data.get("error")
    if err:
        raise RuntimeError(f"OpenRouter STT error: {err}")
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    text = message_text_content(msg)
    return text, elapsed


def estimate_duration_for_cost(path: Path, reference_text: str | None) -> float:
    """Seconds (for $/min estimate) when Whisper verbose_json is unavailable."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "r") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except wave.Error:
            pass
    if reference_text:
        w = len(reference_text.split())
        return max(w / 2.3, 0.8)
    # MP3 fallback: assume ~128 kbps
    bits = path.stat().st_size * 8
    return max(bits / 128_000.0, 0.5)


def stt_openrouter(
    api_key: str, model: str, audio_path: Path, logs: list[dict[str, Any]], ref_text: str | None
) -> dict[str, Any]:
    print(f"\n[ STT / OpenRouter ] {audio_path.name}")

    def _do():
        return openrouter_stt_chat(api_key, model, audio_path)

    text, elapsed = api_call_with_retry(_do)
    duration = estimate_duration_for_cost(audio_path, ref_text)
    cost = (duration / 60.0) * STT_COST_PER_MINUTE
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    log_call(
        logs,
        call_type="stt",
        model=model,
        latency_s=elapsed,
        input_size=f"{size_mb:.3f} MB, ~{duration:.1f}s est.",
        cost_usd=cost,
        extra={"file": audio_path.name},
    )
    print(f"  Transcript: {text[:100]}{'...' if len(text) > 100 else ''}")
    print(f"  {elapsed:.2f}s | duration est. {duration:.1f}s for cost | ~${cost:.4f}")
    return {
        "text": text,
        "latency_s": elapsed,
        "cost": cost,
        "duration_s": duration,
        "language": "en (assumed)",
    }


def validate_audio_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")
    ext = path.suffix.lower()
    if ext not in STT_SUPPORTED:
        raise ValueError(
            f"Unsupported audio format '{ext}'. Supported: {', '.join(sorted(STT_SUPPORTED))}"
        )
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_AUDIO_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). Maximum is {MAX_AUDIO_MB} MB."
        )


def write_demo_wav(path: Path, duration_s: float = 1.5, freq_hz: float = 440.0) -> None:
    sample_rate = 16_000
    n_samples = int(sample_rate * duration_s)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            v = int(32767 * 0.2 * math.sin(2 * math.pi * freq_hz * i / sample_rate))
            wf.writeframes(struct.pack("<h", v))


def word_overlap_accuracy(original: str, transcribed: str) -> float:
    o_words = original.lower().split()
    t_set = set(transcribed.lower().split())
    if not o_words:
        return 100.0
    o_set = set(o_words)
    overlap = o_set & t_set
    return round(100.0 * len(overlap) / len(o_set), 1)


def print_side_by_side(a: str, b: str, width: int = 72) -> None:
    def chunk(s: str):
        s = s.replace("\n", " ")
        return [s[i : i + width] for i in range(0, len(s), width)] or [""]

    a_lines, b_lines = chunk(a), chunk(b)
    n = max(len(a_lines), len(b_lines))
    print("\n  --- Side by side (wrapped) ---")
    for i in range(n):
        al = a_lines[i] if i < len(a_lines) else ""
        bl = b_lines[i] if i < len(b_lines) else ""
        print(f"  Original:    {al}")
        print(f"  Transcribed: {bl}")
        if i < n - 1:
            print()


def summarize(logs: list[dict[str, Any]]) -> None:
    tts = [x for x in logs if x["call_type"] == "tts"]
    stt = [x for x in logs if x["call_type"] == "stt"]
    tts_cost = sum(x["estimated_cost_usd"] for x in tts)
    stt_cost = sum(x["estimated_cost_usd"] for x in stt)
    tts_lat = [x["latency_seconds"] for x in tts]
    stt_lat = [x["latency_seconds"] for x in stt]
    avg_tts = sum(tts_lat) / len(tts_lat) if tts_lat else 0.0
    avg_stt = sum(stt_lat) / len(stt_lat) if stt_lat else 0.0

    print("\n=== Cost and Latency Summary ===")
    print(
        f"  TTS calls:  {len(tts)} | Total cost: ${tts_cost:.4f} | "
        f"Avg latency: {avg_tts:.2f}s"
    )
    print(
        f"  STT calls:  {len(stt)} | Total cost: ${stt_cost:.4f} | "
        f"Avg latency: {avg_stt:.2f}s"
    )
    print(f"  Pipeline total (est.): ${tts_cost + stt_cost:.4f}")


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="HW2 audio pipeline: TTS + STT round trip.")
    parser.add_argument("--text", type=str, default=None, help="Text to synthesize.")
    args = parser.parse_args()
    text = (args.text or DEFAULT_TEXT).strip()
    if not text:
        print("ERROR: Empty text.")
        sys.exit(1)

    mode, oa_client, or_model = resolve_backend()
    or_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()

    print("\n=== HW2 Audio Pipeline ===")
    print(f"  Backend: {mode}")
    if mode == "openrouter_chat":
        print(f"  OpenRouter audio model: {or_model} (override with OPENROUTER_AUDIO_MODEL)")
    print()

    logs: list[dict[str, Any]] = []
    path_nova = OUTPUT_DIR / "voice_nova_sample.mp3"
    path_alloy = OUTPUT_DIR / "voice_alloy_sample.mp3"

    try:
        print("[1/5] Generating speech with voice: nova")
        if mode == "openai_native":
            assert oa_client is not None
            tts_openai_native(oa_client, text, "nova", path_nova, logs)
        else:
            tts_openrouter(or_key, or_model, text, "nova", path_nova, logs)

        print("\n[2/5] Generating speech with voice: alloy")
        if mode == "openai_native":
            assert oa_client is not None
            tts_openai_native(oa_client, text, "alloy", path_alloy, logs)
        else:
            tts_openrouter(or_key, or_model, text, "alloy", path_alloy, logs)

        print("\n[3/5] Transcribing generated MP3 (round trip)")
        if mode == "openai_native":
            assert oa_client is not None
            stt_main = stt_openai_native(oa_client, path_nova, logs)
        else:
            stt_main = stt_openrouter(or_key, or_model, path_nova, logs, ref_text=text)

        print("\n[4/5] Demonstrating WAV support (short synthetic tone)")
        wav_path = OUTPUT_DIR / "hw2_demo_tone.wav"
        try:
            write_demo_wav(wav_path)
            if mode == "openai_native":
                assert oa_client is not None
                stt_openai_native(oa_client, wav_path, logs)
            else:
                stt_openrouter(or_key, or_model, wav_path, logs, ref_text=None)
        except Exception as e:
            print(f"  NOTE: WAV demo skipped: {e}")

        print("\n[5/5] Comparing original vs transcribed (nova round trip)")
        acc = word_overlap_accuracy(text, stt_main["text"])
        print_side_by_side(text, stt_main["text"])
        print(f"\n  Word overlap accuracy: {acc}%")

        summarize(logs)
        print("\n=== Pipeline complete ===\n")

    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
