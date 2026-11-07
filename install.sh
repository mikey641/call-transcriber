#!/bin/zsh
# Installs call-transcriber on macOS (Apple Silicon):
#   - native ASR service (whisper.cpp Metal + pyannote MPS) as a LaunchAgent
#   - Speakr in Docker, wired to it and to local Ollama
#   - the folder bridge as a LaunchAgent, wrapped in a tiny app so macOS can
#     grant it access to Google Drive / CloudStorage folders
#
# Usage: ./install.sh "<recordings folder>"
# Re-running is safe: existing secrets, tokens and models are kept.
set -eu
setopt pipefail

ROOT="${0:A:h}"
RECORDINGS_DIR="${1:-}"
MODELS_DIR="${ASR_MODELS_DIR:-$HOME/whisper-models}"
CONFIG_DIR="$HOME/.config/call-transcriber"
CONFIG="$CONFIG_DIR/config.env"
AGENTS="$HOME/Library/LaunchAgents"
LOGS="$HOME/Library/Logs"
APP="$HOME/Applications/CallTranscriberBridge.app"
UID_DOMAIN="gui/$(id -u)"

say_step() { print -P "%F{cyan}==>%f $*"; }
die() { print -P "%F{red}error:%f $*" >&2; exit 1; }

[[ "$(uname -m)" == arm64 ]] || die "Apple Silicon Mac required"
if [[ -z "$RECORDINGS_DIR" && -f "$CONFIG" ]]; then
  RECORDINGS_DIR=$(sed -n 's/^RECORDINGS_DIR=//p' "$CONFIG" | tr -d '"')
fi
[[ -n "$RECORDINGS_DIR" ]] || die "usage: ./install.sh \"<folder your call recorder syncs to>\""

say_step "Homebrew packages"
for pkg in whisper-cpp ffmpeg uv; do
  brew list --versions "$pkg" >/dev/null 2>&1 || brew install "$pkg"
done
command -v docker >/dev/null || die "install Docker (Colima, OrbStack or Docker Desktop) first"
command -v ollama >/dev/null || print "note: Ollama not found; set TEXT_MODEL_* in speakr/.env to another LLM"

say_step "Models in $MODELS_DIR"
mkdir -p "$MODELS_DIR"
fetch() {  # url dest
  [[ -f "$2" ]] && return
  curl -L --fail -o "$2.part" "$1" && mv "$2.part" "$2"
}
fetch https://huggingface.co/ivrit-ai/whisper-large-v3-ggml/resolve/main/ggml-model.bin "$MODELS_DIR/ivrit-large-v3.bin"
fetch https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin "$MODELS_DIR/ggml-silero-v6.2.0.bin"

say_step "Python environment"
(cd "$ROOT" && uv sync -q)
HF_TOKEN_FILE="$HOME/.cache/huggingface/token"
[[ -f "$HF_TOKEN_FILE" ]] || die "no Hugging Face token: run 'uvx hf auth login' first"
code=$(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $(cat "$HF_TOKEN_FILE")" \
  https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml)
[[ "$code" == 200 || "$code" == 302 || "$code" == 307 ]] \
  || die "accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1 (HTTP $code)"
(cd "$ROOT" && uv run python -c 'from pyannote.audio import Pipeline; Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")' >/dev/null 2>&1)

render() {  # template dest  (replaces __ROOT__, __HOME__, __APP__)
  sed -e "s|__ROOT__|$ROOT|g" -e "s|__HOME__|$HOME|g" -e "s|__APP__|$APP|g" "$1" > "$2"
}
reload_agent() {  # label plist
  launchctl bootout "$UID_DOMAIN/$1" 2>/dev/null || true
  for i in {1..30}; do launchctl print "$UID_DOMAIN/$1" >/dev/null 2>&1 || break; sleep 1; done  # bootout is async
  launchctl bootstrap "$UID_DOMAIN" "$2"
}

say_step "ASR service (LaunchAgent dev.call-transcriber.asr)"
mkdir -p "$AGENTS" "$LOGS"
render "$ROOT/launchd/dev.call-transcriber.asr.plist" "$AGENTS/dev.call-transcriber.asr.plist"
reload_agent dev.call-transcriber.asr "$AGENTS/dev.call-transcriber.asr.plist"
for i in {1..60}; do curl -sf http://127.0.0.1:9000/health >/dev/null && break; sleep 2; done
curl -sf http://127.0.0.1:9000/health >/dev/null || die "ASR service did not start; see $LOGS/call-transcriber-asr.log"

say_step "Speakr (Docker)"
ENV="$ROOT/speakr/.env"
if [[ ! -f "$ENV" ]]; then
  host=host.lima.internal
  [[ "$(docker context show 2>/dev/null)" == colima* ]] || host=host.docker.internal
  # a fresh random value for each placeholder (admin password, secret key)
  perl -pe 's/__generated_by_install__/my $x = `openssl rand -hex 24`; chomp $x; $x/ge' "$ROOT/speakr/env.example" \
    | sed -e "s|^HOST_ADDRESS=.*|HOST_ADDRESS=$host|" -e "s|\${HOST_ADDRESS}|$host|g" > "$ENV"
  chmod 600 "$ENV"
fi
mkdir -p "$ROOT/speakr/data/uploads" "$ROOT/speakr/data/instance"
if docker compose version >/dev/null 2>&1; then compose=(docker compose)
elif command -v docker-compose >/dev/null; then compose=(docker-compose)
else die "install Docker Compose (brew install docker-compose)"; fi
(cd "$ROOT/speakr" && $compose up -d)
for i in {1..90}; do curl -s -o /dev/null http://127.0.0.1:8899/ && break; sleep 2; done
curl -s -o /dev/null http://127.0.0.1:8899/ || die "Speakr did not start: docker logs speakr"

say_step "Bridge config ($CONFIG)"
mkdir -p "$CONFIG_DIR"
token=""
[[ -f "$CONFIG" ]] && token=$(sed -n 's/^SPEAKR_TOKEN=//p' "$CONFIG")
if [[ -z "$token" || "$token" == __* ]]; then
  admin=$(sed -n 's/^ADMIN_USERNAME=//p' "$ENV")
  token=$(docker exec -i speakr python - "$admin" < "$ROOT/scripts/create_speakr_token.py" | tail -1)
  [[ -n "$token" ]] || die "could not create a Speakr API token"
fi
sed -e "s|^RECORDINGS_DIR=.*|RECORDINGS_DIR=\"$RECORDINGS_DIR\"|" -e "s|^SPEAKR_TOKEN=.*|SPEAKR_TOKEN=$token|" \
  "$ROOT/config.env.example" > "$CONFIG"
chmod 600 "$CONFIG"

say_step "Bridge (LaunchAgent dev.call-transcriber.bridge)"
# A plain launchd job may not read ~/Library/CloudStorage (TCC). Running the
# bridge from a small app lets macOS ask once for access to that folder.
mkdir -p "${APP:h}"
rm -rf "$APP"
osacompile -o "$APP" -e "do shell script \"export PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin; cd '$ROOT' && exec '$ROOT/.venv/bin/python' bridge.py >> '$LOGS/call-transcriber-bridge.log' 2>&1\""
/usr/libexec/PlistBuddy -c 'Add :LSUIElement bool true' "$APP/Contents/Info.plist"
/usr/libexec/PlistBuddy -c 'Set :CFBundleIdentifier dev.call-transcriber.bridge' "$APP/Contents/Info.plist" 2>/dev/null \
  || /usr/libexec/PlistBuddy -c 'Add :CFBundleIdentifier string dev.call-transcriber.bridge' "$APP/Contents/Info.plist"
codesign --force --deep -s - "$APP"
render "$ROOT/launchd/dev.call-transcriber.bridge.plist" "$AGENTS/dev.call-transcriber.bridge.plist"
pkill -f "$APP" 2>/dev/null || true
reload_agent dev.call-transcriber.bridge "$AGENTS/dev.call-transcriber.bridge.plist"

admin_pw=$(sed -n 's/^ADMIN_PASSWORD=//p' "$ENV")
print
print -P "%F{green}Done.%f"
print "  Speakr:   http://127.0.0.1:8899   (user: $(sed -n 's/^ADMIN_USERNAME=//p' "$ENV"), password: $admin_pw)"
print "  Logs:     $LOGS/call-transcriber-asr.log, $LOGS/call-transcriber-bridge.log"
print "  If macOS asks whether CallTranscriberBridge may access your files, click Allow."
