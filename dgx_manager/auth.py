from __future__ import annotations

import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from .db import Database

COOKIE_NAME = "dmm2_session"
CSRF_HEADER = "x-csrf-token"
ROLE_RANK = {"viewer": 10, "operator": 20, "admin": 30}

_login_attempts: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=128))


def check_login_rate(key: str, limit: int = 8, window_seconds: int = 300) -> None:
    now = time.monotonic()
    q = _login_attempts[key]
    while q and now - q[0] > window_seconds:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")
    q.append(now)


def clear_login_rate(key: str) -> None:
    _login_attempts.pop(key, None)


def _db(request: Request) -> Database:
    return request.app.state.db


def current_user(request: Request) -> dict:
    if request.app.state.config.get("app.demo_mode", False):
        return {"id": 0, "username": "demo-admin", "display_name": "Demo Admin", "role": "admin", "is_active": 1, "auth_kind": "demo", "csrf_token": "demo"}
    db = _db(request)
    # API token auth for automation.
    auth = request.headers.get("authorization", "")
    if auth.startswith("Bearer dmm2_"):
        user = db.resolve_api_token(auth[7:])
        if user:
            user["auth_kind"] = "api_token"
            return user
    token = request.cookies.get(COOKIE_NAME, "")
    resolved = db.get_session_user(token)
    if not resolved:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    user, csrf = resolved
    user["csrf_token"] = csrf
    user["auth_kind"] = "session"
    return user


def require_role(min_role: str):
    def dep(request: Request, user: dict = Depends(current_user)) -> dict:
        if ROLE_RANK.get(user.get("role", ""), 0) < ROLE_RANK[min_role]:
            raise HTTPException(status_code=403, detail=f"{min_role.title()} role required")
        return user
    return dep


def enforce_csrf(request: Request, user: dict = Depends(current_user)) -> dict:
    if request.method in {"POST", "PUT", "PATCH", "DELETE"} and user.get("auth_kind") == "session":
        incoming = request.headers.get(CSRF_HEADER, "")
        if not incoming or not secrets.compare_digest(incoming, user.get("csrf_token", "")):
            raise HTTPException(status_code=403, detail="Invalid CSRF token")
    return user


def role_and_csrf(min_role: str):
    def dep(request: Request, user: dict = Depends(enforce_csrf)) -> dict:
        if ROLE_RANK.get(user.get("role", ""), 0) < ROLE_RANK[min_role]:
            raise HTTPException(status_code=403, detail=f"{min_role.title()} role required")
        return user
    return dep
