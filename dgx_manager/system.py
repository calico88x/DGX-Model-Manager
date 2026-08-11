from __future__ import annotations

import asyncio
import ipaddress
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from urllib.parse import urlparse

import httpx
import psutil


async def run_cmd(*cmd: str, timeout: float = 30, cwd: str | Path | None = None, env: dict | None = None) -> subprocess.CompletedProcess:
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd) if cwd else None,
        env=env,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill(); await proc.wait()
        return subprocess.CompletedProcess(cmd, 124, "", "timed out")
    return subprocess.CompletedProcess(cmd, proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace"))


def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except Exception:
        return "127.0.0.1"


def is_allowed_service_url(url: str, allow_public: bool = False) -> tuple[bool, str]:
    try:
        p = urlparse(url)
    except Exception:
        return False, "Invalid URL"
    if p.scheme not in {"http", "https"} or not p.hostname or p.username or p.password or p.fragment:
        return False, "Only plain http(s) service URLs without credentials or fragments are allowed"
    try:
        port = p.port
        if port is not None and not (1 <= port <= 65535):
            return False, "Invalid port"
    except ValueError:
        return False, "Invalid port"
    if allow_public:
        return True, ""
    try:
        infos = socket.getaddrinfo(p.hostname, port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False, "Hostname could not be resolved"
    addrs = {i[4][0] for i in infos}
    if not addrs:
        return False, "Hostname did not resolve"
    for addr in addrs:
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return False, "Invalid resolved address"
        if not (ip.is_loopback or ip.is_private):
            return False, f"Public service target {ip} is blocked by security policy"
        if ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return False, f"Unsafe service target {ip} is blocked"
    return True, ""


async def service_check(
    client: httpx.AsyncClient,
    base: str,
    path: str,
    timeout: float = 3.0,
    headers: dict[str,str] | None = None,
) -> dict:
    t0 = time.monotonic()
    try:
        r = await client.get(
            base.rstrip("/") + path,
            timeout=timeout,
            headers=headers,
        )
        return {
            "ok": r.status_code < 400,
            "status_code": r.status_code,
            "latency_ms": round((time.monotonic()-t0)*1000),
        }
    except Exception as exc:
        return {
            "ok": False,
            "latency_ms": None,
            "error": type(exc).__name__,
        }


def _gpu_info_sync() -> dict:
    data = {"available": False, "name": None, "utilization_pct": None, "temperature_c": None, "power_w": None, "memory_used_mb": None, "memory_total_mb": None}
    exe = shutil.which("nvidia-smi")
    if not exe:
        return data
    queries = "name,utilization.gpu,temperature.gpu,power.draw,memory.used,memory.total"
    try:
        cp = subprocess.run([exe, f"--query-gpu={queries}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
        if cp.returncode == 0 and cp.stdout.strip():
            first = cp.stdout.strip().splitlines()[0]
            vals = [x.strip() for x in first.split(",")]
            data["available"] = True
            data["name"] = vals[0] if vals else None
            keys = ["utilization_pct","temperature_c","power_w","memory_used_mb","memory_total_mb"]
            for k,v in zip(keys, vals[1:]):
                try: data[k] = float(v) if v not in {"N/A", "[N/A]"} else None
                except ValueError: data[k] = None
    except Exception:
        pass
    if not data["available"]:
        try:
            cp = subprocess.run([exe, "-L"], capture_output=True, text=True, timeout=3)
            if cp.returncode == 0 and cp.stdout.strip():
                data["available"] = True
                line = cp.stdout.strip().splitlines()[0]
                data["name"] = re.sub(r"^GPU \d+:\s*", "", line).split(" (UUID:")[0]
        except Exception:
            pass
    return data


def system_metrics(hf_cache: Path | None = None) -> dict:
    vm = psutil.virtual_memory()
    disk_path = hf_cache if hf_cache and hf_cache.exists() else Path.home()
    du = shutil.disk_usage(disk_path)
    gpu = _gpu_info_sync()
    gb10 = bool(gpu.get("name") and "GB10" in str(gpu["name"]).upper())
    mem_total_gb = round(vm.total / 1024**3, 1)
    mem_used_gb = round((vm.total - vm.available) / 1024**3, 1)
    return {
        "hostname": socket.gethostname(),
        "ip": local_ip(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "cpu_percent": psutil.cpu_percent(interval=0.05),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_total_gb": mem_total_gb,
        "memory_used_gb": mem_used_gb,
        "memory_available_gb": round(vm.available / 1024**3, 1),
        "memory_percent": vm.percent,
        "disk_total_gb": round(du.total / 1024**3, 1),
        "disk_used_gb": round(du.used / 1024**3, 1),
        "disk_free_gb": round(du.free / 1024**3, 1),
        "disk_percent": round(du.used / du.total * 100, 1) if du.total else 0,
        "gpu": gpu,
        "unified_memory": gb10,
        "platform_class": "DGX Spark / GB10" if gb10 else "NVIDIA GPU system" if gpu.get("available") else "Linux host",
    }


async def docker_available() -> bool:
    r = await run_cmd("docker", "info", "--format", "{{json .ServerVersion}}", timeout=5)
    return r.returncode == 0


async def docker_compose_version() -> str | None:
    r = await run_cmd("docker", "compose", "version", "--short", timeout=5)
    return r.stdout.strip() if r.returncode == 0 else None


async def docker_containers() -> list[dict]:
    r = await run_cmd("docker", "ps", "-a", "--format", "{{json .}}", timeout=10)
    if r.returncode != 0:
        return []
    out = []
    for line in r.stdout.splitlines():
        try:
            d = json.loads(line)
            out.append({
                "id": d.get("ID", ""), "name": d.get("Names", ""), "image": d.get("Image", ""),
                "status": d.get("Status", ""), "state": d.get("State", ""), "ports": d.get("Ports", ""),
                "labels": d.get("Labels", ""),
            })
        except Exception:
            pass
    return out
