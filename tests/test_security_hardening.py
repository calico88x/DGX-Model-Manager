from __future__ import annotations

from pathlib import Path

from dgx_manager.config import Config
from dgx_manager.db import Database


def make_db(tmp_path: Path) -> Database:
    cfg = Config(tmp_path / "config.json")
    cfg.data["paths"]["database"] = str(tmp_path / "db.sqlite")
    cfg.data["paths"]["secret_key"] = str(tmp_path / "secret.key")
    cfg.save()
    return Database(cfg)


def test_password_reset_revokes_sessions(tmp_path: Path):
    db = make_db(tmp_path)
    user = db.create_user("operator", "correct horse battery staple", "Operator", "operator")
    session, _ = db.create_session(user["id"], 12)
    assert db.get_session_user(session)
    db.update_user(user["id"], password="another correct horse battery staple")
    assert db.get_session_user(session) is None


def test_api_token_role_is_clamped_when_account_is_demoted(tmp_path: Path):
    db = make_db(tmp_path)
    user = db.create_user("admin", "correct horse battery staple", "Administrator", "admin")
    db.create_user("backup-admin", "correct horse battery staple", "Backup", "admin")
    _, token = db.create_api_token(user["id"], "automation", "admin")
    assert db.resolve_api_token(token)["role"] == "admin"
    db.update_user(user["id"], role="viewer")
    resolved = db.resolve_api_token(token)
    assert resolved and resolved["role"] == "viewer"
    assert db.list_api_tokens()[0]["role"] == "viewer"


def test_active_admin_count(tmp_path: Path):
    db = make_db(tmp_path)
    a = db.create_user("admin1", "correct horse battery staple", "One", "admin")
    b = db.create_user("admin2", "correct horse battery staple", "Two", "admin")
    assert db.active_admin_count() == 2
    db.update_user(a["id"], is_active=False)
    assert db.active_admin_count() == 1
    try:
        db.update_user(b["id"], role="operator")
        assert False, "last active admin demotion should fail"
    except ValueError as exc:
        assert "administrator" in str(exc)
    assert db.active_admin_count() == 1


def test_bootstrap_token_is_local_one_time_secret(tmp_path: Path):
    cfg = Config(tmp_path / "config.json")
    cfg.data["paths"]["database"] = str(tmp_path / "db.sqlite")
    cfg.data["paths"]["secret_key"] = str(tmp_path / "secret.key")
    cfg.data["paths"]["bootstrap_token"] = str(tmp_path / "bootstrap.token")
    cfg.save()
    db = Database(cfg)
    token, path, created = cfg.ensure_bootstrap_token()
    assert created and token and path.exists()
    assert path.stat().st_mode & 0o777 == 0o600
    assert cfg.verify_bootstrap_token(token)
    assert not cfg.verify_bootstrap_token(token + "x")
    admin = db.create_first_admin("admin", "correct horse battery staple", "Administrator")
    assert admin["role"] == "admin"
    try:
        db.create_first_admin("admin2", "correct horse battery staple", "Administrator 2")
        assert False, "a second bootstrap administrator must not be created"
    except ValueError as exc:
        assert "Bootstrap" in str(exc)
    cfg.clear_bootstrap_token()
    assert not path.exists()
    assert not cfg.verify_bootstrap_token(token)


def test_first_run_http_bootstrap_requires_one_time_token(tmp_path: Path):
    import json
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    cfg_path = tmp_path / "app-config.json"
    cfg = json.loads((root / "config.example.json").read_text())
    cfg["security"]["require_https"] = False
    cfg["security"]["cookie_secure"] = False
    cfg["tls"]["enabled"] = False
    cfg["paths"]["database"] = str(tmp_path / "app.sqlite")
    cfg["paths"]["secret_key"] = str(tmp_path / "app.key")
    cfg["paths"]["compose_root"] = str(tmp_path / "compose")
    cfg["paths"]["custom_dirs"] = str(tmp_path / "dirs.json")
    cfg["paths"]["bootstrap_token"] = str(tmp_path / "bootstrap.token")
    cfg_path.write_text(json.dumps(cfg))
    code = r'''
from pathlib import Path
from fastapi.testclient import TestClient
import app
p=Path(app.config.path_value("paths.bootstrap_token"))
token=p.read_text().strip()
with TestClient(app.app) as c:
    st=c.get("/api/auth/status").json()
    assert st["bootstrap_required"] and st["bootstrap_token_required"]
    bad=c.post("/api/auth/bootstrap",json={"username":"admin","display_name":"Administrator","password":"correct horse battery staple","bootstrap_token":"wrong-token"})
    assert bad.status_code==403
    good=c.post("/api/auth/bootstrap",json={"username":"admin","display_name":"Administrator","password":"correct horse battery staple","bootstrap_token":token})
    assert good.status_code==200, good.text
    assert not p.exists()
    assert c.get("/api/auth/status").json()["bootstrap_required"] is False
'''
    env = os.environ.copy()
    env["DMM_CONFIG"] = str(cfg_path)
    env.pop("DMM_DEMO_MODE", None)
    cp = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
    assert cp.returncode == 0, cp.stdout + cp.stderr
