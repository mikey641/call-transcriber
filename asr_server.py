"""Speakr-compatible ASR service for Apple Silicon.

Implements the `/asr` contract of whisper-asr-webservice / WhisperX that
Speakr's `asr_endpoint` connector speaks, but runs natively on the Mac GPU:

* transcription: whisper.cpp (`whisper-cli`, Metal) with any ggml model, e.g.
  the Hebrew-tuned ivrit.ai large-v3, plus Silero VAD so silence does not
  turn into hallucinated text;
* diarization: pyannote speaker-diarization-community-1 on MPS, which also
  yields the 256-dim per-speaker embeddings Speakr uses for voice profiles.

Run:  uv run uvicorn asr_server:app --host 127.0.0.1 --port 9000
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

log = logging.getLogger("asr")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

MODELS_DIR = Path(os.environ.get("ASR_MODELS_DIR", Path.home() / "whisper-models"))
WHISPER_MODEL = Path(os.environ.get("WHISPER_MODEL", MODELS_DIR / "ivrit-large-v3.bin"))
VAD_MODEL = Path(os.environ.get("VAD_MODEL", MODELS_DIR / "ggml-silero-v6.2.0.bin"))
WHISPER_CLI = os.environ.get("WHISPER_CLI", shutil.which("whisper-cli") or "/opt/homebrew/bin/whisper-cli")
FFMPEG = os.environ.get("FFMPEG", shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
DEFAULT_LANGUAGE = os.environ.get("ASR_DEFAULT_LANGUAGE", "he")
DIARIZATION_MODEL = os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1")
# Short segments make speaker attribution finer: whisper segments are split
# on word boundaries at this many characters.
MAX_SEGMENT_CHARS = int(os.environ.get("ASR_MAX_SEGMENT_CHARS", "80"))

app = FastAPI(title="call-transcriber ASR")
_gpu_lock = threading.Lock()  # one GPU job at a time
_pipeline = None


def _diarization_pipeline():
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        log.info("loading %s", DIARIZATION_MODEL)
        _pipeline = Pipeline.from_pretrained(DIARIZATION_MODEL)
        if torch.backends.mps.is_available():
            _pipeline.to(torch.device("mps"))
    return _pipeline


def _to_wav(src: Path, dst: Path) -> float:
    subprocess.run(
        [FFMPEG, "-nostdin", "-y", "-loglevel", "error", "-i", str(src), "-ar", "16000", "-ac", "1", str(dst)],
        check=True,
    )
    return dst.stat().st_size / 32000  # 16 kHz * 16-bit mono


def _transcribe(wav: Path, language: str, prompt: str | None, workdir: Path) -> list[dict]:
    out = workdir / "whisper"
    cmd = [
        WHISPER_CLI, "-m", str(WHISPER_MODEL), "-f", str(wav), "-l", language,
        "-sns", "-oj", "-of", str(out), "-ml", str(MAX_SEGMENT_CHARS), "-sow",
    ]
    if VAD_MODEL.exists():
        cmd += ["--vad", "-vm", str(VAD_MODEL)]
    if prompt:
        cmd += ["--prompt", prompt]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"whisper-cli failed: {proc.stderr[-800:]}")
    data = json.loads(out.with_suffix(".json").read_text(encoding="utf-8", errors="replace"))
    segments = []
    for item in data.get("transcription", []):
        text = item.get("text", "").strip()
        if not text:
            continue
        segments.append({
            "start": item["offsets"]["from"] / 1000,
            "end": item["offsets"]["to"] / 1000,
            "text": text,
        })
    return segments


def _diarize(wav: Path, min_speakers: int | None, max_speakers: int | None):
    """Return (turns, embeddings): turns = [(start, end, label)], embeddings = {label: [256 floats]}."""
    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers
    output = _diarization_pipeline()(str(wav), **kwargs)
    annotation = getattr(output, "speaker_diarization", output)
    # community-1's exclusive diarization (one speaker at a time) is made for
    # reconciling with transcription timestamps; embeddings follow the
    # regular annotation's label order.
    exclusive = getattr(output, "exclusive_speaker_diarization", None) or annotation
    turns = [(t.start, t.end, label) for t, _, label in exclusive.itertracks(yield_label=True)]
    embeddings = {}
    raw = getattr(output, "speaker_embeddings", None)
    if raw is not None:
        for label, vector in zip(annotation.labels(), raw):
            values = [float(x) for x in vector]
            # a speaker with too little audio gets a NaN embedding, which
            # is useless for voice matching and breaks JSON encoding
            if all(math.isfinite(x) for x in values):
                embeddings[label] = values
    return turns, embeddings


def _assign_speakers(segments: list[dict], turns: list[tuple]) -> None:
    """Give each segment the speaker it overlaps most, else the nearest turn."""
    for seg in segments:
        best, best_overlap = None, 0.0
        for start, end, label in turns:
            overlap = min(seg["end"], end) - max(seg["start"], start)
            if overlap > best_overlap:
                best, best_overlap = label, overlap
        if best is None and turns:
            mid = (seg["start"] + seg["end"]) / 2
            best = min(turns, key=lambda t: min(abs(t[0] - mid), abs(t[1] - mid)))[2]
        seg["speaker"] = best or "SPEAKER_00"


@app.get("/")
@app.get("/health")
def health():
    return {"ok": True, "whisper_model": WHISPER_MODEL.name, "vad": VAD_MODEL.exists(),
            "diarization_model": DIARIZATION_MODEL}


@app.get("/v1/models")
def models():
    return {"data": [{"id": WHISPER_MODEL.stem, "owned_by": "local"}]}


@app.post("/asr")
def asr(
    audio_file: UploadFile = File(...),
    language: str | None = Query(None),
    diarize: bool = Query(True),
    enable_diarization: bool | None = Query(None),
    min_speakers: int | None = Query(None),
    max_speakers: int | None = Query(None),
    return_speaker_embeddings: bool = Query(False),
    initial_prompt: str | None = Query(None),
    output: str = Query("json"),
    # accepted for compatibility, ignored:
    encode: bool = Query(True), task: str = Query("transcribe"),
    hotwords: str | None = Query(None), model: str | None = Query(None),
):
    do_diarize = diarize if enable_diarization is None else enable_diarization
    lang = (language or DEFAULT_LANGUAGE).strip() or DEFAULT_LANGUAGE
    with tempfile.TemporaryDirectory(prefix="asr-") as tmp:
        workdir = Path(tmp)
        src = workdir / ("input" + Path(audio_file.filename or "a.bin").suffix)
        with src.open("wb") as f:
            shutil.copyfileobj(audio_file.file, f)
        wav = workdir / "audio.wav"
        with _gpu_lock:
            t0 = time.time()
            try:
                duration = _to_wav(src, wav)
                segments = _transcribe(wav, lang, initial_prompt, workdir)
                t1 = time.time()
                embeddings = {}
                if do_diarize and segments:
                    turns, embeddings = _diarize(wav, min_speakers, max_speakers)
                    _assign_speakers(segments, turns)
                else:
                    for seg in segments:
                        seg["speaker"] = "SPEAKER_00"
            except subprocess.CalledProcessError as e:
                raise HTTPException(400, f"could not decode audio: {e}") from e
            except Exception as e:  # surface the real error to Speakr
                log.exception("asr failed")
                raise HTTPException(500, str(e)) from e
        log.info("%s: %.0fs audio, transcribe %.1fs, diarize %.1fs, %d segments, %d speakers",
                 audio_file.filename, duration, t1 - t0, time.time() - t1, len(segments),
                 len({s["speaker"] for s in segments}))

    result = {
        "text": "\n".join(s["text"] for s in segments),
        "language": lang,
        "segments": segments,
    }
    if return_speaker_embeddings and embeddings:
        result["speaker_embeddings"] = {k: v for k, v in embeddings.items()
                                        if any(s["speaker"] == k for s in segments)}
    return result
