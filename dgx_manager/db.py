from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError
from cryptography.fernet import Fernet

from .config import Config

PH = PasswordHasher(time_cost=3, memory_cost=65536, parallelism=4, hash_len=32, salt_len=16)
ROLES = {"viewer", "operator", "admin"}
ROLE_RANK = {"viewer": 10, "operator": 20, "admin": 30}
_DUMMY_PASSWORD_HASH = PH.hash("dmm2-dummy-password-for-timing")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat()


class Database:
    def __init__(self, config: Config):
        self.config = config
        self.path = config.path_value("paths.database")
        self.key_path = config.path_value("paths.secret_key")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.key_path.parent.mkdir(parents=True, exist_ok=True)
        self._fernet = Fernet(self._load_or_create_key())
        self.init_schema()

    def _load_or_create_key(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes().strip()
        key = Fernet.generate_key()
        self.key_path.write_bytes(key + b"\n")
        os.chmod(self.key_path, 0o600)
        return key

    @contextmanager
    def conn(self) -> Iterator[sqlite3.Connection]:
        c = sqlite3.connect(self.path, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        try:
            yield c
            c.commit()
        finally:
            c.close()

    def init_schema(self) -> None:
        with self.conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('viewer','operator','admin')),
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_login TEXT
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    csrf_token TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    last_seen TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
                CREATE INDEX IF NOT EXISTS idx_sessions_exp ON sessions(expires_at);
                CREATE TABLE IF NOT EXISTS api_tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    name TEXT NOT NULL,
                    token_hash TEXT NOT NULL UNIQUE,
                    role TEXT NOT NULL CHECK(role IN ('viewer','operator','admin')),
                    created_at TEXT NOT NULL,
                    last_used TEXT
                );
                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    base_url TEXT NOT NULL,
                    token_enc TEXT,
                    verify_tls INTEGER NOT NULL DEFAULT 1,
                    tls_fingerprint TEXT NOT NULL DEFAULT '',
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_seen TEXT,
                    notes TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                    username TEXT,
                    action TEXT NOT NULL,
                    target TEXT,
                    detail TEXT,
                    source_ip TEXT
                );
                """
            )
            cols={r[1] for r in c.execute("PRAGMA table_info(nodes)").fetchall()}
            if "tls_fingerprint" not in cols:
                c.execute("ALTER TABLE nodes ADD COLUMN tls_fingerprint TEXT NOT NULL DEFAULT ''")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    # --- users ---
    def user_count(self) -> int:
        with self.conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM users").fetchone()[0])

    def active_admin_count(self) -> int:
        with self.conn() as c:
            return int(c.execute("SELECT COUNT(*) FROM users WHERE role='admin' AND is_active=1").fetchone()[0])

    @staticmethod
    def _validate_user_fields(username: str, password: str, display_name: str, role: str) -> tuple[str, str]:
        username = username.strip().lower()
        display_name = display_name.strip() or username
        if role not in ROLES:
            raise ValueError("Invalid role")
        if not (3 <= len(username) <= 64) or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for ch in username):
            raise ValueError("Username must be 3-64 characters: lowercase letters, numbers, dot, underscore, hyphen")
        if len(password) < 12:
            raise ValueError("Password must be at least 12 characters")
        return username, display_name

    def create_user(self, username: str, password: str, display_name: str, role: str) -> dict:
        username, display_name = self._validate_user_fields(username, password, display_name, role)
        pwh = PH.hash(password)
        with self.conn() as c:
            try:
                cur = c.execute(
                    "INSERT INTO users(username,display_name,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (username, display_name, pwh, role, iso()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Username already exists") from exc
            user_id = cur.lastrowid
        return self.get_user(user_id)

    def create_first_admin(self, username: str, password: str, display_name: str) -> dict:
        """Atomically create the one bootstrap administrator.

        BEGIN IMMEDIATE serializes concurrent first-run attempts so two LAN clients
        cannot both pass a separate user-count check and create administrators.
        """
        username, display_name = self._validate_user_fields(username, password, display_name, "admin")
        pwh = PH.hash(password)
        with self.conn() as c:
            c.execute("BEGIN IMMEDIATE")
            if int(c.execute("SELECT COUNT(*) FROM users").fetchone()[0]) != 0:
                raise ValueError("Bootstrap has already been completed")
            try:
                cur = c.execute(
                    "INSERT INTO users(username,display_name,password_hash,role,created_at) VALUES(?,?,?,?,?)",
                    (username, display_name, pwh, "admin", iso()),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("Bootstrap has already been completed") from exc
            user_id = cur.lastrowid
        user = self.get_user(user_id)
        if not user:
            raise ValueError("Could not create bootstrap administrator")
        return user

    def get_user(self, user_id: int) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT id,username,display_name,role,is_active,created_at,last_login FROM users WHERE id=?", (user_id,)).fetchone()
            return dict(row) if row else None

    def get_user_by_username(self, username: str, include_hash: bool = False) -> dict | None:
        cols = "id,username,display_name,role,is_active,created_at,last_login" + (",password_hash" if include_hash else "")
        with self.conn() as c:
            row = c.execute(f"SELECT {cols} FROM users WHERE username=? COLLATE NOCASE", (username.strip(),)).fetchone()
            return dict(row) if row else None

    def verify_password(self, username: str, password: str) -> dict | None:
        user = self.get_user_by_username(username, include_hash=True)
        if not user or not user["is_active"]:
            # Perform one Argon2 verification for nonexistent/disabled users so the
            # obvious timing path remains similar without generating a fresh hash on
            # every failed request.
            try:
                PH.verify(_DUMMY_PASSWORD_HASH, password)
            except Exception:
                pass
            return None
        try:
            ok = PH.verify(user["password_hash"], password)
        except (VerifyMismatchError, InvalidHashError):
            return None
        if not ok:
            return None
        if PH.check_needs_rehash(user["password_hash"]):
            with self.conn() as c:
                c.execute("UPDATE users SET password_hash=? WHERE id=?", (PH.hash(password), user["id"]))
        with self.conn() as c:
            c.execute("UPDATE users SET last_login=? WHERE id=?", (iso(), user["id"]))
        user.pop("password_hash", None)
        return user

    def list_users(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT id,username,display_name,role,is_active,created_at,last_login FROM users ORDER BY username")]

    def update_user(self, user_id: int, *, role: str | None = None, is_active: bool | None = None, password: str | None = None, display_name: str | None = None) -> dict:
        current = self.get_user(user_id)
        if not current:
            raise ValueError("User not found")
        removing_active_admin = bool(
            current.get("is_active") and current.get("role") == "admin"
            and (is_active is False or (role is not None and role != "admin"))
        )
        if removing_active_admin and self.active_admin_count() <= 1:
            raise ValueError("At least one active administrator account must remain")
        updates, args = [], []
        if role is not None:
            if role not in ROLES:
                raise ValueError("Invalid role")
            updates.append("role=?"); args.append(role)
        if is_active is not None:
            updates.append("is_active=?"); args.append(1 if is_active else 0)
        if password is not None:
            if len(password) < 12:
                raise ValueError("Password must be at least 12 characters")
            updates.append("password_hash=?"); args.append(PH.hash(password))
        if display_name is not None:
            updates.append("display_name=?"); args.append(display_name.strip())
        if updates:
            args.append(user_id)
            with self.conn() as c:
                c.execute(f"UPDATE users SET {', '.join(updates)} WHERE id=?", args)
                # Password changes invalidate browser sessions immediately.
                if password is not None:
                    c.execute("DELETE FROM sessions WHERE user_id=?", (user_id,))
                # API tokens may never retain a role higher than their owning account.
                if role is not None:
                    for row in c.execute("SELECT id,role FROM api_tokens WHERE user_id=?", (user_id,)).fetchall():
                        if ROLE_RANK.get(row["role"], 0) > ROLE_RANK[role]:
                            c.execute("UPDATE api_tokens SET role=? WHERE id=?", (role, row["id"]))
        user = self.get_user(user_id)
        if not user:
            raise ValueError("User not found")
        return user

    # --- sessions ---
    @staticmethod
    def hash_token(token: str) -> str:
        return hashlib.sha256(token.encode()).hexdigest()

    def create_session(self, user_id: int, hours: int) -> tuple[str, str]:
        token = secrets.token_urlsafe(48)
        csrf = secrets.token_urlsafe(32)
        now = utcnow(); expires = now + timedelta(hours=max(1, min(hours, 168)))
        with self.conn() as c:
            c.execute("DELETE FROM sessions WHERE expires_at < ?", (iso(now),))
            c.execute(
                "INSERT INTO sessions(token_hash,user_id,csrf_token,created_at,expires_at,last_seen) VALUES(?,?,?,?,?,?)",
                (self.hash_token(token), user_id, csrf, iso(now), iso(expires), iso(now)),
            )
        return token, csrf

    def get_session_user(self, token: str) -> tuple[dict, str] | None:
        if not token:
            return None
        now = iso()
        th = self.hash_token(token)
        with self.conn() as c:
            row = c.execute(
                """SELECT s.csrf_token,s.expires_at,u.id,u.username,u.display_name,u.role,u.is_active,u.created_at,u.last_login
                   FROM sessions s JOIN users u ON u.id=s.user_id WHERE s.token_hash=?""",
                (th,),
            ).fetchone()
            if not row:
                return None
            if row["expires_at"] < now or not row["is_active"]:
                c.execute("DELETE FROM sessions WHERE token_hash=?", (th,))
                return None
            c.execute("UPDATE sessions SET last_seen=? WHERE token_hash=?", (now, th))
            data = dict(row); csrf = data.pop("csrf_token"); data.pop("expires_at", None)
            return data, csrf

    def delete_session(self, token: str) -> None:
        if token:
            with self.conn() as c:
                c.execute("DELETE FROM sessions WHERE token_hash=?", (self.hash_token(token),))

    # --- API tokens ---
    def create_api_token(self, user_id: int, name: str, role: str) -> tuple[dict, str]:
        if role not in ROLES:
            raise ValueError("Invalid role")
        token = "dmm2_" + secrets.token_urlsafe(40)
        with self.conn() as c:
            cur = c.execute(
                "INSERT INTO api_tokens(user_id,name,token_hash,role,created_at) VALUES(?,?,?,?,?)",
                (user_id, name.strip()[:80], self.hash_token(token), role, iso()),
            )
            tid = cur.lastrowid
            row = c.execute("SELECT id,user_id,name,role,created_at,last_used FROM api_tokens WHERE id=?", (tid,)).fetchone()
        return dict(row), token

    def resolve_api_token(self, token: str) -> dict | None:
        with self.conn() as c:
            row = c.execute(
                """SELECT t.id AS token_id,t.role AS token_role,u.role AS user_role,u.id,u.username,u.display_name,u.is_active,u.created_at,u.last_login
                   FROM api_tokens t JOIN users u ON u.id=t.user_id WHERE t.token_hash=?""",
                (self.hash_token(token),),
            ).fetchone()
            if not row or not row["is_active"]:
                return None
            c.execute("UPDATE api_tokens SET last_used=? WHERE id=?", (iso(), row["token_id"]))
            data = dict(row)
            token_role = data.pop("token_role")
            user_role = data.pop("user_role")
            data["role"] = token_role if ROLE_RANK.get(token_role, 0) <= ROLE_RANK.get(user_role, 0) else user_role
            return data

    def list_api_tokens(self) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute(
                """SELECT t.id,t.name,t.role,t.created_at,t.last_used,u.username
                   FROM api_tokens t JOIN users u ON u.id=t.user_id ORDER BY t.created_at DESC"""
            )]

    def delete_api_token(self, token_id: int) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM api_tokens WHERE id=?", (token_id,))

    # --- nodes ---
    def upsert_node(self, *, node_id: int | None, name: str, base_url: str, token: str | None, verify_tls: bool, tls_fingerprint: str = "", notes: str = "") -> dict:
        enc = self._fernet.encrypt(token.encode()).decode() if token else None
        with self.conn() as c:
            if node_id:
                if token:
                    c.execute("UPDATE nodes SET name=?,base_url=?,token_enc=?,verify_tls=?,tls_fingerprint=?,notes=? WHERE id=?",
                              (name, base_url, enc, int(verify_tls), tls_fingerprint, notes, node_id))
                else:
                    c.execute("UPDATE nodes SET name=?,base_url=?,verify_tls=?,tls_fingerprint=?,notes=? WHERE id=?",
                              (name, base_url, int(verify_tls), tls_fingerprint, notes, node_id))
                nid = node_id
            else:
                cur = c.execute("INSERT INTO nodes(name,base_url,token_enc,verify_tls,tls_fingerprint,created_at,notes) VALUES(?,?,?,?,?,?,?)",
                                (name, base_url, enc, int(verify_tls), tls_fingerprint, iso(), notes))
                nid = cur.lastrowid
        return self.get_node(nid, include_token=False)

    def get_node(self, node_id: int, include_token: bool = False) -> dict | None:
        with self.conn() as c:
            row = c.execute("SELECT * FROM nodes WHERE id=?", (node_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        if include_token:
            try:
                d["token"] = self._fernet.decrypt(d["token_enc"].encode()).decode() if d.get("token_enc") else ""
            except Exception:
                d["token"] = ""
        d.pop("token_enc", None)
        return d

    def list_nodes(self) -> list[dict]:
        with self.conn() as c:
            rows = c.execute("SELECT id,name,base_url,verify_tls,tls_fingerprint,enabled,created_at,last_seen,notes,token_enc IS NOT NULL AS token_set FROM nodes ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    def touch_node(self, node_id: int) -> None:
        with self.conn() as c:
            c.execute("UPDATE nodes SET last_seen=? WHERE id=?", (iso(), node_id))

    def delete_node(self, node_id: int) -> None:
        with self.conn() as c:
            c.execute("DELETE FROM nodes WHERE id=?", (node_id,))

    # --- audit ---
    def audit(self, action: str, target: str = "", detail: Any = None, user: dict | None = None, source_ip: str = "") -> None:
        detail_text = json.dumps(detail, default=str, ensure_ascii=False)[:8000] if detail is not None else ""
        with self.conn() as c:
            c.execute(
                "INSERT INTO audit(ts,user_id,username,action,target,detail,source_ip) VALUES(?,?,?,?,?,?,?)",
                (iso(), user.get("id") if user else None, user.get("username") if user else None, action, target[:512], detail_text, source_ip[:128]),
            )

    def list_audit(self, limit: int = 200) -> list[dict]:
        with self.conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (max(1, min(limit, 1000)),))]
