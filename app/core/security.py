"""Password hashing. bcrypt with a per-password salt; hashes are not reversible."""
import bcrypt


def hash_password(password: str) -> str:
    # bcrypt caps input at 72 bytes; the 8-char minimum is enforced upstream.
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # Malformed/legacy hash string.
        return False
