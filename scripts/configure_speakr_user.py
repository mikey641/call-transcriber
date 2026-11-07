# Runs inside the Speakr container: phone-call defaults for a user (Hebrew
# transcription, short Hebrew summaries, diarization on).
# Usage: docker exec -i speakr python - <username> [language-code] [output-language] < this
import sys

sys.path.insert(0, "/app")
from src.app import app, db  # noqa: E402
from src.models import User  # noqa: E402

SUMMARY_PROMPT = """This is a recorded phone call. Write a short summary in the output language.
Match the length to the call: a one-minute call gets one or two lines, never a long report.
Include only what is actually there: what the call was about, decisions or agreements,
tasks and who owns them, and any dates, amounts, addresses or names mentioned.
No tables, no "participants" or "date" sections, no interpretation of individual lines.
If the call has no real content (wrong number, voicemail, a few words), say so in one line."""

username = sys.argv[1]
language = sys.argv[2] if len(sys.argv) > 2 else "he"
output_language = sys.argv[3] if len(sys.argv) > 3 else "Hebrew"
with app.app_context():
    user = User.query.filter_by(username=username).first()
    if user is None:
        sys.exit(f"no Speakr user {username!r}")
    user.transcription_language = language
    user.output_language = output_language
    user.summary_prompt = SUMMARY_PROMPT
    user.diarize = True
    db.session.commit()
    print(f"configured {username}: {language} / {output_language}")
