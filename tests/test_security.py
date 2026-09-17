from app.core.security import hash_password, verify_password


def test_hash_is_not_plaintext_and_verifies():
    h = hash_password("correct horse")
    assert h != "correct horse"
    assert verify_password("correct horse", h)


def test_wrong_password_fails():
    assert not verify_password("wrong", hash_password("correct horse"))


def test_salts_differ_per_hash():
    assert hash_password("same") != hash_password("same")


def test_malformed_hash_returns_false():
    assert not verify_password("x", "not-a-bcrypt-hash")
