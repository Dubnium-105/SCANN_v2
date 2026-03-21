from __future__ import annotations

import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: Literal["admin", "annotator"]


class AuthUser(BaseModel):
    username: str
    role: Literal["admin", "annotator"]


class _StoredUser(BaseModel):
    username: str
    password_hash: str
    role: Literal["admin", "annotator"]


pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
bearer_scheme = HTTPBearer(auto_error=False)

JWT_SECRET = os.getenv(
    "SCANN_NATIVE_JWT_SECRET",
    "scann-native-dev-secret-2026-minimum-32-bytes",
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = int(os.getenv("SCANN_NATIVE_JWT_EXPIRE_MINUTES", "120"))


def _build_default_users() -> dict[str, _StoredUser]:
    return {
        "annotator": _StoredUser(
            username="annotator",
            password_hash=pwd_context.hash("scann123"),
            role="annotator",
        ),
        "admin": _StoredUser(
            username="admin",
            password_hash=pwd_context.hash("admin123"),
            role="admin",
        ),
    }


def _get_dataset_root() -> Path:
    return Path(os.getenv("SCANN_NATIVE_DATASET_ROOT", "dataset")).resolve()


def _get_db_path() -> Path:
    configured = os.getenv("SCANN_NATIVE_DB_PATH", "").strip()
    if configured:
        return Path(configured).resolve()
    return _get_dataset_root() / "scann_native.db"


def _get_connection() -> sqlite3.Connection:
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'annotator')),
            created_at TEXT NOT NULL
        )
        """
    )
    connection.commit()


def _ensure_default_users(connection: sqlite3.Connection) -> None:
    defaults = _build_default_users()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for item in defaults.values():
        connection.execute(
            """
            INSERT OR IGNORE INTO users (username, password_hash, role, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (item.username, item.password_hash, item.role, now),
        )
    connection.commit()


def _load_user(connection: sqlite3.Connection, username: str) -> Optional[_StoredUser]:
    row = connection.execute(
        "SELECT username, password_hash, role FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    if row is None:
        return None
    return _StoredUser(
        username=str(row["username"]),
        password_hash=str(row["password_hash"]),
        role=str(row["role"]),
    )


def authenticate_user(username: str, password: str) -> Optional[AuthUser]:
    with _get_connection() as connection:
        _ensure_schema(connection)
        _ensure_default_users(connection)
        user = _load_user(connection, username)
        if user is None:
            return None
        if not pwd_context.verify(password, user.password_hash):
            return None
        return AuthUser(username=user.username, role=user.role)


def register_user(username: str, password: str) -> AuthUser:
    normalized_username = username.strip()
    if len(normalized_username) < 3 or len(normalized_username) > 32:
        raise ValueError("Username must be 3-32 characters")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", normalized_username):
        raise ValueError("Username can only contain letters, numbers, _, -, .")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")

    with _get_connection() as connection:
        _ensure_schema(connection)
        _ensure_default_users(connection)
        existing = _load_user(connection, normalized_username)
        if existing is not None:
            raise ValueError("Username already exists")

        connection.execute(
            """
            INSERT INTO users (username, password_hash, role, created_at)
            VALUES (?, ?, 'annotator', ?)
            """,
            (
                normalized_username,
                pwd_context.hash(password),
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
            ),
        )
        connection.commit()
        return AuthUser(username=normalized_username, role="annotator")


def create_access_token(user: AuthUser) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user.username,
        "role": user.role,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def _decode_token(token: str) -> AuthUser:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token") from exc

    username = payload.get("sub")
    role = payload.get("role")
    if not isinstance(username, str) or role not in {"admin", "annotator"}:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")

    return AuthUser(username=username, role=role)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> AuthUser:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return _decode_token(credentials.credentials)
