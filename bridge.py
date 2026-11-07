#!/usr/bin/env python3
"""Send phone call recordings from a synced folder to Speakr and write transcripts back.

For every recording in RECORDINGS_DIR (e.g. Cube ACR's Google Drive folder):
  1. upload it to Speakr once (title, call time and phone number parsed from the
     Cube ACR file name; an idempotency key prevents duplicates);
  2. when Speakr has finished (transcript with speakers + summary), write
     `<recording>.txt` next to it: summary first, then the speaker-labelled
     transcript, so it can be read on the phone through Drive.

Speakr never touches the recordings folder, so nothing is moved or deleted.
Standard library only; configuration comes from the environment or
~/.config/call-transcriber/config.env (see config.env.example).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
import sys
import time
import urllib.error
import shutil
import urllib.request
from datetime import datetime
from pathlib import Path

CONFIG_FILE = Path(os.environ.get("CALL_TRANSCRIBER_CONFIG", Path.home() / ".config/call-transcriber/config.env"))
AUDIO_EXT = {".m4a", ".mp3", ".amr", ".ogg", ".opus", ".aac", ".3gp", ".wav"}
STATE_FILE = Path.home() / ".local/state/call-transcriber/recordings.json"
# Cube ACR names: phone_20260101-093000_0501234567.amr, gmeet_20260101-100000_Team sync.amr
NAME_RE = re.compile(r"^(?P<kind>[a-z]+)_(?P<date>\d{8})-(?P<time>\d{6})_(?P<who>.*)$")


def load_config() -> dict:
    cfg = {}
    if CONFIG_FILE.exists():
        for line in CONFIG_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                cfg[key.strip()] = value.strip().strip('"').strip("'")
    cfg.update({k: v for k, v in os.environ.items() if k in {
        "RECORDINGS_DIR", "SPEAKR_URL", "SPEAKR_TOKEN", "LANGUAGE", "POLL_SECONDS", "MIN_SECONDS", "NO_SPEECH_TEXT", "CONTACTS_JSON"}})
    missing = [k for k in ("RECORDINGS_DIR", "SPEAKR_URL", "SPEAKR_TOKEN") if not cfg.get(k)]
    if missing:
        sys.exit(f"missing config: {', '.join(missing)} (set them in {CONFIG_FILE})")
    return cfg


CFG = load_config()
RECORDINGS_DIR = Path(os.path.expanduser(CFG["RECORDINGS_DIR"]))
SPEAKR = CFG["SPEAKR_URL"].rstrip("/")
LANGUAGE = CFG.get("LANGUAGE", "he")
POLL_SECONDS = int(CFG.get("POLL_SECONDS", "30"))
MIN_SECONDS = float(CFG.get("MIN_SECONDS", "2"))  # shorter recordings are not sent
NO_SPEECH = CFG.get("NO_SPEECH_TEXT", "[אין דיבור]")
SETTLE_SECONDS = 20
MAX_RETRIES = 2
MAX_UPLOAD_ATTEMPTS = 5
WRITE_INTERVAL = 2.5  # seconds between API writes (Speakr: 30 per minute per token)
NAMING_VERSION = 2  # bump when describe() changes to retitle existing uploads  # Speakr reprocess attempts for a failed recording
# launchd and app wrappers start with a minimal PATH
os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "/usr/bin:/bin")
FFPROBE = shutil.which("ffprobe") or "ffprobe"


def log(msg: str) -> None:
    print(f"{datetime.now():%F %T} {msg}", flush=True)


_last_write = 0.0


def pace_write() -> None:
    """Speakr allows 30 writes per minute per API token; stay under it."""
    global _last_write
    wait = _last_write + WRITE_INTERVAL - time.time()
    if wait > 0:
        time.sleep(wait)
    _last_write = time.time()


def api(method: str, path: str, data: bytes | None = None, headers: dict | None = None):
    if method != "GET":
        pace_write()
    req = urllib.request.Request(f"{SPEAKR}/api/v1{path}", method=method, data=data,
                                 headers={"Authorization": f"Bearer {CFG['SPEAKR_TOKEN']}", **(headers or {})})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read() or b"{}")


def duration(path: Path) -> float | None:
    """Seconds of audio, or None when it cannot be measured (then the file is uploaded anyway)."""
    try:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(path)], capture_output=True, text=True, timeout=60).stdout.strip()
        return float(out)
    except (ValueError, OSError, subprocess.SubprocessError):
        return None


def norm_phone(value: str) -> str:
    digits = re.sub(r"\D", "", value or "")
    if digits.startswith("00"):
        digits = digits[2:]
    if digits.startswith("972") and len(digits) >= 11:  # +972 54... -> 054...
        digits = "0" + digits[3:]
    return digits


def load_contacts() -> dict:
    """Optional {phone: name} map (CONTACTS_JSON), keys normalized like norm_phone."""
    path = CFG.get("CONTACTS_JSON")
    if not path:
        return {}
    try:
        raw = json.loads(Path(os.path.expanduser(path)).read_text())
        return {norm_phone(k): v for k, v in raw.items() if isinstance(v, str) and v.strip()}
    except (OSError, ValueError, AttributeError) as e:
        log(f"contacts not loaded from {path}: {e}")
        return {}


def describe(path: Path, contacts: dict) -> dict:
    """Title, ISO call time and participant parsed from a Cube ACR file name."""
    m = NAME_RE.match(path.stem)
    if not m:
        return {"title": path.stem}
    when = datetime.strptime(m["date"] + m["time"], "%Y%m%d%H%M%S")
    who = m["who"].strip("_ ").replace("_", " ")
    if re.fullmatch(r"[\d+]+", who):
        phone = norm_phone(who)
        who = f"{contacts[phone]} ({phone})" if phone in contacts else phone
    info = {"title": f"{who or m['kind']} · {when:%Y-%m-%d %H:%M}", "meeting_date": when.isoformat()}
    if who:
        info["participants"] = who
    return info


def upload(path: Path, contacts: dict) -> int:
    pace_write()
    meta = describe(path, contacts)
    key = hashlib.sha256(str(path.relative_to(RECORDINGS_DIR)).encode()).hexdigest()[:64]
    with tempfile.TemporaryDirectory(prefix="call-transcriber-") as tmp:
        # Copy first: an online-only Google Drive / iCloud file is materialized
        # lazily, and curl streaming it directly gets cut off after ~1 KB.
        # The copy blocks until the whole file is downloaded.
        local = Path(tmp) / ("recording" + path.suffix.lower())
        shutil.copyfile(path, local)
        if local.stat().st_size != path.stat().st_size:
            raise RuntimeError(f"incomplete copy ({local.stat().st_size}/{path.stat().st_size} bytes)")
        filename = re.sub(r'[;,"\\]', " ", path.name)  # characters curl's -F syntax would misread
        cmd = ["curl", "-sS", "--fail-with-body", "-H", f"Authorization: Bearer {CFG['SPEAKR_TOKEN']}",
               "-F", f'file=@{local};filename="{filename}"', "-F", f"language={LANGUAGE}",
               "-F", f"idempotency_key={key}", "-F", f"file_last_modified={int(path.stat().st_mtime * 1000)}"]
        for field, value in meta.items():
            cmd += ["--form-string", f"{field}={value}"]
        proc = subprocess.run(cmd + [f"{SPEAKR}/api/v1/recordings/upload", "-w", "\n%{http_code}"],
                              capture_output=True, text=True)
    body, _, code = proc.stdout.rpartition("\n")
    if code == "429":
        raise RateLimited()
    proc.stdout = body
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout or proc.stderr)[-500:])
    return int(json.loads(proc.stdout)["id"])


def render(rec_id: int) -> str:
    summary = api("GET", f"/recordings/{rec_id}/summary").get("summary") or ""
    try:
        transcript = api("GET", f"/recordings/{rec_id}/transcript?format=text").get("content") or ""
    except urllib.error.HTTPError as e:
        if e.code != 404:  # 404 = no transcription (no speech)
            raise
        transcript = ""
    if not transcript.strip():
        return NO_SPEECH + "\n"
    parts = []
    if summary.strip():
        parts += ["# סיכום", summary.strip(), ""]
    parts += ["# תמלול", transcript.strip(), "", f"Speakr: {SPEAKR} (recording #{rec_id})"]
    return "\n".join(parts) + "\n"


class RateLimited(Exception):
    pass


def statuses() -> dict[int, str]:
    """Status of every recording, one paginated listing instead of a GET per recording."""
    result, page = {}, 1
    while True:
        data = api("GET", f"/recordings?per_page=100&page={page}")
        for rec in data.get("recordings", []):
            result[int(rec["id"])] = rec.get("status")
        if not data.get("pagination", {}).get("has_next"):
            return result
        page += 1


def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    tmp.replace(STATE_FILE)


def tick(state: dict) -> None:
    now = time.time()
    contacts = None
    current = None  # recording statuses, fetched once per tick when needed
    for path in sorted(RECORDINGS_DIR.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXT or not path.is_file():
            continue
        rel = str(path.relative_to(RECORDINGS_DIR))
        entry = state.get(rel)
        if entry is not None and entry["status"] == "retry" and now >= entry.get("next", 0):
            entry = None  # upload failed earlier: try again
        if entry is None:
            if now - path.stat().st_mtime < SETTLE_SECONDS:
                continue
            seconds = duration(path)
            if seconds is not None and seconds < MIN_SECONDS:
                path.with_suffix(".txt").write_text(NO_SPEECH + "\n", encoding="utf-8")
                state[rel] = {"status": "done", "id": None}
                log(f"no speech (too short) {rel}")
            else:
                if contacts is None:
                    contacts = load_contacts()
                attempts = state.get(rel, {}).get("attempts", 0) + 1
                try:
                    state[rel] = {"status": "uploaded", "id": upload(path, contacts), "named": NAMING_VERSION}
                    log(f"uploaded {rel} -> recording {state[rel]['id']}")
                except RateLimited:
                    raise
                except Exception as e:  # e.g. Drive had not finished downloading the file
                    final = attempts >= MAX_UPLOAD_ATTEMPTS
                    state[rel] = {"status": "failed" if final else "retry", "attempts": attempts,
                                  "next": now + 300 * attempts, "error": str(e)[:300]}
                    log(f"upload {'FAILED' if final else 'failed, will retry'} {rel}: {e}")
            save_state(state)
        elif entry.get("id") and entry.get("named") != NAMING_VERSION:
            # bring older uploads' titles in line with current naming (e.g. contact names added)
            if contacts is None:
                contacts = load_contacts()
            meta = describe(path, contacts)
            try:
                api("PATCH", f"/recordings/{entry['id']}",
                    data=json.dumps({k: meta[k] for k in ("title", "participants") if k in meta}).encode(),
                    headers={"Content-Type": "application/json"})
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
            entry["named"] = NAMING_VERSION
            save_state(state)
        elif entry["status"] == "uploaded":
            if current is None:
                current = statuses()
            status = current.get(entry["id"])
            if status is None:  # deleted in Speakr: upload again next tick
                state.pop(rel)
                save_state(state)
                continue
            if status == "COMPLETED":
                path.with_suffix(".txt").write_text(render(entry["id"]), encoding="utf-8")
                entry["status"] = "done"
                log(f"transcript written {Path(rel).with_suffix('.txt')}")
                save_state(state)
            elif status == "FAILED":
                error = str(api("GET", f"/recordings/{entry['id']}").get("error_message") or "")
                if "returned no text" in error:  # Speakr's way of saying "no speech"
                    path.with_suffix(".txt").write_text(NO_SPEECH + "\n", encoding="utf-8")
                    entry["status"] = "done"
                    log(f"no speech {rel}")
                elif entry.get("retries", 0) < MAX_RETRIES:
                    entry["retries"] = entry.get("retries", 0) + 1
                    api("POST", f"/recordings/{entry['id']}/transcribe", data=b"{}",
                        headers={"Content-Type": "application/json"})
                    log(f"retrying recording {entry['id']} ({rel}), attempt {entry['retries']}")
                else:
                    entry["status"] = "failed"
                    log(f"Speakr FAILED recording {entry['id']} ({rel}): {error[:200]}")
                save_state(state)


def single_instance() -> None:
    """Block until no other bridge runs (a killed wrapper app can orphan one)."""
    import fcntl

    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock = open(STATE_FILE.with_suffix(".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("another bridge is running; waiting for it to exit")
        fcntl.flock(lock, fcntl.LOCK_EX)
    globals()["_lock"] = lock  # keep the descriptor (and the lock) alive


def allow_cloud_downloads() -> None:
    """Let this process materialize online-only (dataless) files.

    Processes started by launchd may have materialization turned off, and then
    reading a not-yet-downloaded Google Drive / iCloud file fails with
    EDEADLK ("Resource deadlock avoided"). setiopolicy_np turns it back on.
    """
    import ctypes

    IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, IOPOL_SCOPE_PROCESS, IOPOL_MATERIALIZE_DATALESS_FILES_ON = 3, 0, 2
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.setiopolicy_np(IOPOL_TYPE_VFS_MATERIALIZE_DATALESS_FILES, IOPOL_SCOPE_PROCESS,
                           IOPOL_MATERIALIZE_DATALESS_FILES_ON) != 0:
        log(f"could not enable cloud file downloads (errno {ctypes.get_errno()})")


def main() -> None:
    allow_cloud_downloads()
    single_instance()
    log(f"watching {RECORDINGS_DIR} -> {SPEAKR}")
    state = load_state()
    while True:
        try:
            tick(state)
        except RateLimited:
            log("Speakr rate limit hit; pausing a minute")
            time.sleep(60)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                log("Speakr rate limit hit; pausing a minute")
                time.sleep(60)
            else:
                log(f"Speakr error: {e}")
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            log(f"Speakr unreachable: {e}")
        except Exception as e:  # keep running; the next tick retries
            log(f"error: {e!r}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
