"""Small local authentication layer for the administrative surface."""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from .database import AuthToken, SessionLocal, UserAccount

PBKDF2_ITERATIONS = 210_000
TOKEN_TTL_HOURS = 24


def hash_password(password: str) -> str:
    if len(password) < 8:
        raise ValueError("Password must contain at least 8 characters")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS)
    return "pbkdf2_sha256${}${}${}".format(
        PBKDF2_ITERATIONS,
        base64.urlsafe_b64encode(salt).decode("ascii"),
        base64.urlsafe_b64encode(digest).decode("ascii"),
    )


def verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations, encoded_salt, encoded_digest = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        salt = base64.urlsafe_b64decode(encoded_salt.encode("ascii"))
        expected = base64.urlsafe_b64decode(encoded_digest.encode("ascii"))
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, int(iterations))
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def issue_token(db: Session, user: UserAccount) -> str:
    raw = secrets.token_urlsafe(32)
    db.add(AuthToken(user_id=user.id, token_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                     expires_at=datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS)))
    db.commit()
    return raw


def _token_hash(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def revoke_token(db: Session, raw: str) -> None:
    token = db.scalar(select(AuthToken).where(AuthToken.token_hash == _token_hash(raw)))
    if token:
        db.delete(token)
        db.commit()


def auth_dependency(authorization: str | None, db: Session) -> UserAccount:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Authentication required", headers={"WWW-Authenticate": "Bearer"})
    raw = authorization.split(" ", 1)[1].strip()
    token = db.scalar(select(AuthToken).where(AuthToken.token_hash == _token_hash(raw)))
    if not token or token.expires_at <= datetime.utcnow():
        raise HTTPException(401, "Session expired or invalid", headers={"WWW-Authenticate": "Bearer"})
    user = db.get(UserAccount, token.user_id)
    if not user:
        raise HTTPException(401, "User not found", headers={"WWW-Authenticate": "Bearer"})
    return user
