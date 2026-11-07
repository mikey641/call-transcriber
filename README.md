# call-transcriber

Self-hosted phone call transcription for Apple Silicon Macs, with speaker
separation, voice profiles, summaries and search, tuned for Hebrew.

It combines:

- **[Speakr](https://github.com/murtaza-nasir/speakr)** for the features: web UI,
  speaker labels, voice profiles that recognize the same person across calls,
  AI summaries, chat with a recording, semantic search across all calls, tags
  and an API;
- **a native ASR service** (`asr_server.py`) that replaces Speakr's usual
  WhisperX/CUDA backend with what runs well on a Mac:
  - [whisper.cpp](https://github.com/ggml-org/whisper.cpp) on the Metal GPU
    with the Hebrew-tuned [ivrit.ai large-v3](https://huggingface.co/ivrit-ai/whisper-large-v3-ggml)
    model (any ggml Whisper model works);
  - Silero VAD, so silent or ringing-only recordings don't come out as
    hallucinated text ("תודה", "Thank you.");
  - [pyannote community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
    diarization on MPS, which also provides the 256-dim speaker embeddings
    Speakr's voice profiles use;
- **a folder bridge** (`bridge.py`) that watches the folder your phone's call
  recorder syncs to, uploads each new call to Speakr (titled with the contact
  name and call time), and writes a `.txt` next to the recording with the
  summary and the full speaker-labelled transcript, so you can read it on the
  phone. Speakr never touches the recordings folder.

```
phone call recorder (e.g. Cube ACR)
  └─ Autosync for Google Drive ─► Drive folder ─► Google Drive for desktop (Mac)
                                                     │
                                       bridge.py ◄───┘  writes <call>.txt back
                                           │ upload / poll
                                           ▼
                              Speakr (Docker) ──/asr──► asr_server.py (native)
                                   │                     whisper.cpp + ivrit (Metal)
                                   │                     pyannote (MPS)
                                   └── summaries/chat ──► Ollama (local) or any OpenAI-compatible API
```

Everything runs locally by default; no audio or text leaves the Mac.

## Requirements

- Apple Silicon Mac (tested on M1 Max, 32 GB)
- [Homebrew](https://brew.sh), Docker ([Colima](https://github.com/abiosoft/colima),
  OrbStack or Docker Desktop) with Compose
- A [Hugging Face](https://huggingface.co) token (`uvx hf auth login`) whose
  account has accepted the terms of
  [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
- [Ollama](https://ollama.com) with a model for summaries (default `gpt-oss:20b`),
  or an OpenAI-compatible API set in `speakr/.env`
- About 15 GB of disk: Whisper model 3 GB, Speakr image 10 GB

## Install

```sh
git clone https://github.com/<you>/call-transcriber.git && cd call-transcriber
./install.sh "$HOME/Library/CloudStorage/GoogleDrive-you@gmail.com/My Drive/cube ACR"
```

The installer:

1. installs `whisper-cpp`, `ffmpeg` and `uv`, and downloads the ivrit.ai and Silero VAD models into `~/whisper-models`;
2. starts the ASR service as the LaunchAgent `dev.call-transcriber.asr` on `127.0.0.1:9000`;
3. starts Speakr on `http://127.0.0.1:8899`, generating an admin password and secret key in `speakr/.env`, and prints the login;
4. creates a Speakr API token and writes `~/.config/call-transcriber/config.env`;
5. starts the bridge as the LaunchAgent `dev.call-transcriber.bridge`.

macOS doesn't let plain background jobs read `~/Library/CloudStorage`, so
the bridge runs inside a tiny wrapper app (`~/Applications/CallTranscriberBridge.app`).
The first time it runs, macOS asks whether it may access your Google Drive
files. Click **Allow**.

For Hebrew phone calls, also set short Hebrew summaries and diarization for your Speakr user:

```sh
docker exec -i speakr python - admin he Hebrew < scripts/configure_speakr_user.py
```

Re-running `./install.sh` is safe; it keeps existing secrets, tokens and models.

## Configuration

`~/.config/call-transcriber/config.env` (see `config.env.example`):

| Key | Meaning |
|---|---|
| `RECORDINGS_DIR` | folder the phone syncs recordings to |
| `SPEAKR_URL`, `SPEAKR_TOKEN` | Speakr address and API token (set by the installer) |
| `LANGUAGE` | transcription language (`he`, `en`, …, or `auto`) |
| `MIN_SECONDS` | shorter recordings get `NO_SPEECH_TEXT` without being uploaded |
| `CONTACTS_JSON` | optional `{"phone": "name"}` file, so calls are titled by contact name |

ASR service environment (set in `launchd/dev.call-transcriber.asr.plist`):
`WHISPER_MODEL`, `VAD_MODEL`, `ASR_DEFAULT_LANGUAGE`, `DIARIZATION_MODEL`,
`ASR_MAX_SEGMENT_CHARS`.

Speakr settings live in `speakr/.env` (see `speakr/env.example` and the
[Speakr docs](https://murtaza-nasir.github.io/speakr/)).

## Using it

- **Name speakers once:** open a call in Speakr and give each `SPEAKER_xx` a
  name. With voice profiles, later calls with the same person are labelled
  automatically.
- **Ask across calls:** Speakr's Inquire mode searches and answers questions
  over the whole call history.
- **On the phone:** each `<recording>.txt` in the Drive folder has the summary
  first, then the full transcript.

## Operations

```sh
tail -f ~/Library/Logs/call-transcriber-bridge.log ~/Library/Logs/call-transcriber-asr.log
launchctl kickstart -k gui/$(id -u)/dev.call-transcriber.asr     # restart ASR
docker restart speakr
```

Bridge state, recording by recording, is kept in `~/.local/state/call-transcriber/recordings.json`.
To re-send a recording, delete its entry and restart the bridge.

The ASR service also works on its own, with any client of the
whisper-asr-webservice API:

```sh
curl -F audio_file=@call.m4a "http://127.0.0.1:9000/asr?language=he&diarize=true&return_speaker_embeddings=true"
```

## Performance

On an M1 Max with ivrit.ai large-v3, a 5.4-minute call took about 50 s to transcribe and 30 s to diarize.

## License

MIT for this repository. Speakr runs from its own Docker image and is
[AGPL-3.0](https://github.com/murtaza-nasir/speakr/blob/master/LICENSE). The
models have their own licenses; see the ivrit.ai and pyannote community-1 (CC-BY-4.0) model cards.
