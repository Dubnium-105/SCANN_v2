from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Literal, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel


class LoginRequest(BaseModel):
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


_USERS = _build_default_users()


def authenticate_user(username: str, password: str) -> Optional[AuthUser]:
    user = _USERS.get(username)
    if user is None:
        return None
    if not pwd_context.verify(password, user.password_hash):
        return None
    return AuthUser(username=user.username, role=user.role)


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
