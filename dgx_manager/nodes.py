from __future__ import annotations

import asyncio
import hashlib
import socket
import ssl
from urllib.parse import urlparse

import httpx

from .db import Database


def _cert_fingerprint(url: str) -> str:
    p=urlparse(url); host=p.hostname; port=p.port or 443
    if not host: raise ValueError("Invalid node URL")
    ctx=ssl._create_unverified_context()
    with socket.create_connection((host,port),timeout=6) as raw:
        with ctx.wrap_socket(raw,server_hostname=host) as s:
            cert=s.getpeercert(binary_form=True)
    return hashlib.sha256(cert).hexdigest().lower()


class NodeClient:
    def __init__(self, db: Database, timeout: float = 15):
        self.db=db; self.timeout=timeout

    async def request(self,node_id:int,method:str,path:str,*,json:dict|None=None)->dict:
        node=self.db.get_node(node_id,include_token=True)
        if not node or not node.get("enabled",1): raise ValueError("Node not found or disabled")
        headers={"Authorization":f"Bearer {node.get('token','')}"}
        verify=bool(node.get("verify_tls",1))
        if node["base_url"].startswith("https://") and not verify:
            expected=(node.get("tls_fingerprint") or "").replace(":","").lower()
            if not expected: raise ValueError("TLS verification is disabled but no pinned certificate fingerprint is configured")
            actual=await asyncio.to_thread(_cert_fingerprint,node["base_url"])
            if actual != expected: raise ValueError("Remote node TLS certificate fingerprint does not match the pinned value")
        async with httpx.AsyncClient(timeout=self.timeout,verify=verify) as c:
            r=await c.request(method,node["base_url"].rstrip("/")+path,headers=headers,json=json)
            r.raise_for_status(); data=r.json()
        self.db.touch_node(node_id)
        return data

    async def info(self,node_id:int)->dict: return await self.request(node_id,"GET","/v1/info")
    async def metrics(self,node_id:int)->dict: return await self.request(node_id,"GET","/v1/metrics")
    async def dashboard(self,node_id:int)->dict: return await self.request(node_id,"GET","/v1/dashboard")
    async def inventory(self,node_id:int)->dict: return await self.request(node_id,"GET","/v1/inventory")
    async def generate(self,node_id:int,payload:dict)->dict: return await self.request(node_id,"POST","/v1/compose/generate",json=payload)
    async def deployments(self,node_id:int)->dict: return await self.request(node_id,"GET","/v1/compose")
    async def save_plan(self,node_id:int,plan:dict)->dict: return await self.request(node_id,"POST","/v1/compose",json=plan)
    async def up(self,node_id:int,engine:str,slug:str)->dict: return await self.request(node_id,"POST",f"/v1/compose/{engine}/{slug}/up")
    async def down(self,node_id:int,engine:str,slug:str)->dict: return await self.request(node_id,"POST",f"/v1/compose/{engine}/{slug}/down")
    async def logs(self,node_id:int,engine:str,slug:str,lines:int=200)->dict: return await self.request(node_id,"GET",f"/v1/compose/{engine}/{slug}/logs?lines={lines}")
    async def status(self,node_id:int,engine:str,slug:str)->dict: return await self.request(node_id,"GET",f"/v1/compose/{engine}/{slug}/status")
    async def remove(self,node_id:int,engine:str,slug:str)->dict: return await self.request(node_id,"DELETE",f"/v1/compose/{engine}/{slug}")
