#!/usr/bin/env python3
"""Offline pre-publish validation for DGX Model Manager v2."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TEXT_EXT={'.py','.js','.css','.html','.md','.json','.yaml','.yml','.sh','.txt'}
EXCLUDE_PARTS={'venv','.venv','__pycache__','.pytest_cache','.git'}

# Assemble development-only fixture values so the validator itself does not contain them verbatim.
BLOCKED_PRIVATE=["192.168."+"3.51","/home/"+"nova","zgx-"+"40e6"]
SECRET_PATTERNS=[
    re.compile(r'(?i)(api[_-]?key|password|token|secret|master[_-]?key)\s*[:=]\s*["\']?(?!<|\*\*\*|false|true|null|none|not-needed|example|placeholder)[A-Za-z0-9_\-]{20,}'),
    re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
]

def iter_text():
    for p in ROOT.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in TEXT_EXT or any(x in p.parts for x in EXCLUDE_PARTS):
            continue
        yield p,p.read_text(errors='ignore')

def check_source_privacy(errors:list[str]):
    for p,text in iter_text():
        rel=p.relative_to(ROOT)
        for term in BLOCKED_PRIVATE:
            if term in text: errors.append(f'private development fixture in {rel}')
        for pat in SECRET_PATTERNS:
            if pat.search(text): errors.append(f'possible embedded secret/private key in {rel}')

def check_config(errors:list[str]):
    try: cfg=json.loads((ROOT/'config.example.json').read_text())
    except Exception as exc: errors.append(f'config.example.json invalid: {exc}'); return
    if cfg.get('app',{}).get('port') != 8091: errors.append('test/coexistence port must default to 8091')
    if cfg.get('paths',{}).get('hf_cache') != '~/.cache/huggingface/hub': errors.append('HF cache compatibility default changed')
    if cfg.get('paths',{}).get('litellm_config') != '~/litellm/litellm_config.yaml': errors.append('LiteLLM path compatibility default changed')
    if not cfg.get('security',{}).get('require_https',False): errors.append('HTTPS must be required by default')
    if cfg.get('app',{}).get('legacy_scripts_enabled',True): errors.append('Legacy Script Mode must default off')

def run(cmd:list[str],errors:list[str]):
    cp=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True)
    if cp.returncode: errors.append(f"{' '.join(cmd)} failed: {(cp.stdout+cp.stderr).strip()[-1200:]}")

def main()->int:
    errors=[]
    check_source_privacy(errors); check_config(errors)
    run([sys.executable,'-m','compileall','-q','app.py','agent.py','dgx_manager','scripts','tools'],errors)
    if (ROOT/'static/app.js').exists():
        try: run(['node','--check','static/app.js'],errors)
        except FileNotFoundError: print('WARN: node not installed; JS syntax check skipped')
    for sh in ['setup.sh','setup-agent.sh','scripts/promote_v2.sh']:
        run(['bash','-n',sh],errors)
    required=[
        'README.md',
        'SECURITY.md',
        'CHANGELOG.md',
        'RELEASE_NOTES.md',
        'CONTRIBUTING.md',
        'docs.html',
        'LICENSE',
        'config.example.json',
        'engine_catalog.yaml',
        'docs/ARCHITECTURE.md',
        'docs/COMPOSE_BUILDER.md',
        'docs/UPGRADE.md',
        'scripts/bootstrap_token.py',
        'scripts/build_release.sh',
        'misc/screenshots/dashboard.png',
        'misc/screenshots/compose-builder.png',
        'misc/screenshots/inventory.png',
        'misc/screenshots/deployments.png',
        'misc/screenshots/cluster.png',
        'misc/screenshots/users-access.png',
    ]
    for rel in required:
        if not (ROOT/rel).exists(): errors.append(f'missing release artifact: {rel}')
    if errors:
        print('RELEASE VALIDATION FAILED')
        for e in errors: print(' -',e)
        return 1
    print('Release validation passed.')
    return 0

if __name__=='__main__': raise SystemExit(main())
