# Runs inside the Speakr container: creates a full-access API token for the
# given user and prints it. Usage: docker exec -i speakr python - <username> < this
import secrets
import sys

sys.path.insert(0, "/app")
from src.app import app, db  # noqa: E402
from src.models import APIToken, User  # noqa: E402
from src.utils.token_auth import hash_token  # noqa: E402

with app.app_context():
    user = User.query.filter_by(username=sys.argv[1]).first()
    if user is None:
        sys.exit(f"no Speakr user {sys.argv[1]!r}")
    token = secrets.token_urlsafe(32)
    db.session.add(APIToken(user_id=user.id, token_hash=hash_token(token), name="call-transcriber bridge"))
    db.session.commit()
    print(token)
