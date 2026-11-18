import pytest

from app.auth import hash_password, verify_password


def test_password_hash_is_salted_and_verifiable():
    first = hash_password("correct horse battery")
    second = hash_password("correct horse battery")
    assert first != second
    assert verify_password("correct horse battery", first)
    assert not verify_password("wrong password", first)


def test_password_policy_rejects_short_passwords():
    with pytest.raises(ValueError):
        hash_password("short")
