'use strict';

const state = {
  csrf: '', user: null, config: null, inventory: [], builderInventory: [], deployments: [], nodes: [], plan: null,
  currentPage: 'dashboard', hfResults: [], selectedNode: 'local',
};
const SERVICE_LABELS = {ollama:'Ollama',litellm:'LiteLLM',sglang:'SGLang',vllm:'vLLM',llamacpp:'llama.cpp',localai:'LocalAI',comfyui:'ComfyUI'};
const ENGINE_KEYS = ['sglang','vllm','llamacpp','localai','comfyui'];

function esc(value) {
  return String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
}
function fmtGB(v) {
  if (v == null) return '—';

  const gb = Number(v);
  if (!Number.isFinite(gb)) return '—';

  if (gb >= 1) {
    return `${gb.toFixed(gb >= 10 ? 1 : 2)} GB`;
  }

  const mb = gb * 1000;
  if (mb >= 1) {
    return `${mb.toFixed(mb >= 10 ? 1 : 2)} MB`;
  }

  const kb = mb * 1000;
  if (kb >= 1) {
    return `${kb.toFixed(kb >= 10 ? 0 : 1)} kB`;
  }

  return `${Math.round(kb * 1000)} B`;
}
function fmtNum(v) { return Number(v || 0).toLocaleString(); }
function fmtBytes(v) { if (v == null) return '—'; const n=Number(v); if(n>=1e9)return `${(n/1e9).toFixed(1)} GB`; if(n>=1e6)return `${(n/1e6).toFixed(1)} MB`; if(n>=1e3)return `${(n/1e3).toFixed(1)} KB`; return `${n} B`; }
function cap(s){ return String(s||'').replace(/(^|[-_ ])\w/g,m=>m.toUpperCase()); }
function toast(msg, kind='') { const el=document.getElementById('toast'); el.textContent=msg; el.style.borderLeftColor=kind==='error'?'var(--red)':kind==='ok'?'var(--green2)':'var(--amber)'; el.classList.add('show'); clearTimeout(window.__toast); window.__toast=setTimeout(()=>el.classList.remove('show'),2600); }

async function api(path, options={}) {
  const opts = {...options}; opts.headers = {...(opts.headers||{})};
  const method=(opts.method||'GET').toUpperCase();
  if (opts.json !== undefined) { opts.body=JSON.stringify(opts.json); opts.headers['Content-Type']='application/json'; delete opts.json; }
  if (!['GET','HEAD','OPTIONS'].includes(method) && state.csrf) opts.headers['X-CSRF-Token']=state.csrf;
  const r=await fetch(path,opts);
  if (r.status===401) { showAuth('Session expired. Sign in again.'); throw new Error('Authentication required'); }
  let data=null; const ct=r.headers.get('content-type')||'';
  if (ct.includes('application/json')) data=await r.json(); else data=await r.text();
  if (!r.ok) throw new Error(data?.detail || data?.error || data || `HTTP ${r.status}`);
  return data;
}
function modal(title, bodyHtml, footerHtml='') { document.getElementById('modal-title').textContent=title; document.getElementById('modal-body').innerHTML=bodyHtml; document.getElementById('modal-foot').innerHTML=footerHtml; document.getElementById('generic-modal').classList.add('show'); }
function closeModal(){
  const el=document.getElementById('generic-modal');
  el.classList.remove('show','log-viewer');
}
function showAuth(message='Sign in to continue.') { const lock=document.getElementById('auth-lock'); lock.classList.add('show'); document.getElementById('auth-message').textContent=message; }
function hideAuth(){ document.getElementById('auth-lock').classList.remove('show'); }
function showRegister(){
  document.getElementById('login-form').classList.add('hidden');
  document.getElementById('bootstrap-form').classList.add('hidden');
  document.getElementById('registration-form').classList.remove('hidden');
  document.getElementById('auth-message').textContent='Create a Viewer account. An administrator can grant additional privileges later.';
}
function showLogin(){
  document.getElementById('registration-form').classList.add('hidden');
  document.getElementById('bootstrap-form').classList.add('hidden');
  document.getElementById('login-form').classList.remove('hidden');
  document.getElementById('auth-message').textContent='Sign in to DGX Model Manager v2.';
}

function setRoleUI() {
  const role=state.user?.role||'viewer';
  const isAdmin=role==='admin';
  const canOperate=role==='operator'||role==='admin';
  document.querySelectorAll('.role-admin-only').forEach(el=>el.classList.toggle('hidden-role',!isAdmin));
  const operatorSelectors=[
    '[data-action="open-dir-dialog"]','[data-action="ollama-pull"]','[data-action="hf-download"]',
    '[data-action="save-plan"]','[data-action="apply-wildcard"]','[data-delete-path]','[data-remove-dir]',
    '[data-ollama-delete]','[data-hf-download]','[data-dep-up]','[data-dep-down]','[data-dep-route]',
    '[data-dep-remove]','[data-route-add]','[data-route-remove]','[data-engine-start]','[data-engine-stop]',
    '[data-legacy-start]'
  ].join(',');
  document.querySelectorAll(operatorSelectors).forEach(el=>{
    if(!canOperate){
      if(!el.disabled){el.disabled=true;el.dataset.roleDisabledByUi='1';}
      el.title='Operator or Admin role required';
    } else if(el.dataset.roleDisabledByUi==='1'){
      el.disabled=false;delete el.dataset.roleDisabledByUi;el.removeAttribute('title');
    }
  });
  document.getElementById('user-name').textContent=state.user?.display_name || state.user?.username || 'User';
  document.getElementById('user-avatar').textContent=(state.user?.display_name || state.user?.username || '?').slice(0,1).toUpperCase();
}

async function initAuth() {
  const st=await fetch('/api/auth/status').then(r=>r.json()).catch(()=>({bootstrap_required:false}));
  const registrationOpen=document.getElementById('registration-open');
  if(registrationOpen)registrationOpen.classList.toggle('hidden',!st.registration_enabled);
  if (st.bootstrap_required) {
    showAuth('First-run setup: create the administrator account.');
    document.getElementById('login-form').classList.add('hidden'); document.getElementById('bootstrap-form').classList.remove('hidden');
    return false;
  }
  try {
    const me=await api('/api/auth/me'); state.user=me.user; state.csrf=me.csrf_token||''; hideAuth(); setRoleUI(); return true;
  } catch(e) {
    showAuth('Sign in to DGX Model Manager v2.'); return false;
  }
}

async function login() {
  const username=document.getElementById('login-user').value.trim(); const password=document.getElementById('login-pass').value;
  try { const d=await api('/api/auth/login',{method:'POST',json:{username,password}}); state.user=d.user; state.csrf=d.csrf_token; hideAuth(); setRoleUI(); await initialLoad(); }
  catch(e){ document.getElementById('auth-message').textContent=e.message; }
}
async function registerAccount(){
  const username=document.getElementById('register-user').value.trim();
  const display_name=document.getElementById('register-name').value.trim();
  const password=document.getElementById('register-pass').value;
  try{
    await api('/api/auth/register',{method:'POST',json:{username,display_name,password}});
    document.getElementById('login-user').value=username;
    document.getElementById('login-pass').value='';
    showLogin();
    document.getElementById('auth-message').textContent='Account created. Sign in to continue.';
  }catch(e){document.getElementById('auth-message').textContent=e.message;}
}
async function bootstrap() {
  const username=document.getElementById('bootstrap-user').value.trim(); const display_name=document.getElementById('bootstrap-name').value.trim(); const password=document.getElementById('bootstrap-pass').value; const bootstrap_token=document.getElementById('bootstrap-token').value.trim();
  try { const d=await api('/api/auth/bootstrap',{method:'POST',json:{username,display_name,password,bootstrap_token}}); state.user=d.user; state.csrf=d.csrf_token; hideAuth(); setRoleUI(); await initialLoad(); }
  catch(e){ document.getElementById('auth-message').textContent=e.message; }
}
async function logout(){ try{await api('/api/auth/logout',{method:'POST'});}catch(e){} state.user=null; state.csrf=''; closeModal(); showAuth('Signed out.'); }

function go(page) {
  if (!document.getElementById(`page-${page}`)) return;
  state.currentPage=page;
  document.querySelectorAll('.page').forEach(x=>x.classList.remove('active'));
  document.getElementById(`page-${page}`).classList.add('active');
  document.querySelectorAll('.nav-item').forEach(x=>x.classList.toggle('active',x.dataset.page===page));
  const loaders={dashboard:loadDashboard,inventory:loadInventory,ollama:loadOllama,hf:()=>{},downloads:()=>{},deployments:loadDeployments,builder:loadBuilder,routing:loadRouting,engines:loadEngines,legacy:loadLegacy,cluster:loadCluster,logs:loadLogs,access:loadAccess,settings:loadSettings,docs:()=>{}};
  loaders[page]?.();
}

function metricCard(label,value,unit,foot,pct,accent='var(--amber)') {
  return `<div class="metric" style="--metric-accent:${accent}"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}<span class="metric-unit">${esc(unit||'')}</span></div><div class="bar"><span style="width:${Math.max(0,Math.min(100,Number(pct)||0))}%;--bar:${accent}"></span></div><div class="metric-foot"><span>${esc(foot||'')}</span></div></div>`;
}

async function loadDashboard() {
  const root=document.getElementById('dashboard-metrics'); root.innerHTML='<div class="empty-state">Loading dashboard...</div>';
  try {
    const remote=state.selectedNode!=='local';
    const endpoint=remote?`/api/nodes/${encodeURIComponent(state.selectedNode)}/dashboard`:'/api/dashboard';
    const d=await api(endpoint); const m=d.metrics;
    root.innerHTML=[
      metricCard('CPU load',`${Number(m.cpu_percent).toFixed(0)}`,'%',`${m.cpu_count} logical cores`,m.cpu_percent,'var(--blue)'),
      metricCard(m.unified_memory?'Unified memory':'System memory',`${Number(m.memory_used_gb).toFixed(1)}`,'GB',`${Number(m.memory_available_gb).toFixed(1)} GB available`,m.memory_percent,'var(--green2)'),
      metricCard('Model storage',`${Number(d.model_size_gb).toFixed(1)}`,'GB',`${d.model_count} discovered models`,m.disk_percent,'var(--amber)'),
      metricCard('Disk usage',`${Number(m.disk_used_gb).toFixed(0)}`,'GB',`${Number(m.disk_free_gb).toFixed(0)} GB free`,m.disk_percent,'var(--purple)'),
    ].join('');
    if(!remote)document.getElementById('nav-model-count').textContent=d.model_count;
    document.getElementById('nav-deploy-count').textContent=d.deployments.length;
    document.getElementById('nav-node-count').textContent=1+(d.nodes?.length||state.nodes.length||0);
    document.getElementById('brand-node').textContent=`${m.hostname} · ${m.ip} · ${m.platform_class}`;
    const selectedName=remote?(state.nodes.find(n=>String(n.id)===String(state.selectedNode))?.name||m.hostname):'Local node';
    document.getElementById('cluster-summary').textContent=`${1+(d.nodes?.length||state.nodes.length||0)} node${(d.nodes?.length||state.nodes.length||0)?'s':''} · ${selectedName} · ${d.compose_version?'Compose '+d.compose_version:'Compose unavailable'}`;
    const sv=document.getElementById('dashboard-services'); sv.innerHTML=Object.entries(d.status||{}).map(([k,v])=>`<div class="service-row"><span class="status-dot ${v.ok?'ok':'err'}"></span><div class="row-main"><div class="service-name">${esc(SERVICE_LABELS[k]||k)}</div><div class="service-meta">${remote?'Target-node service':esc(state.config?.services?.[`${k}_base`]||'')}</div></div><div class="row-right">${v.ok?`${esc(v.latency_ms??'—')} ms`:'Offline'}${v.model?` · ${esc(v.model)}`:''}</div></div>`).join('');
    const dep=document.getElementById('dashboard-deployments'); dep.innerHTML=d.deployments.length?d.deployments.slice(0,5).map(x=>`<div class="stack-row"><div class="iconbox">${esc((x.engine||'?').slice(0,2).toUpperCase())}</div><div class="row-main"><div class="stack-name">${esc(x.name||x.slug)}</div><div class="stack-meta">${esc(x.model_name||'')} · ${esc(x.node||selectedName)}</div></div><div class="row-right"><span class="badge ${x.fit_status==='good'?'green':x.fit_status==='risk'?'red':'amber'}">${esc(x.fit_status||'saved')}</span></div></div>`).join(''):'<div class="empty-state">No Compose deployments yet.</div>';
    document.getElementById('platform-summary').textContent=`${m.platform_class} · ${m.architecture}`;
    document.getElementById('platform-cards').innerHTML=`<div class="signal"><div class="signal-k">GPU</div><div class="signal-v">${esc(m.gpu?.name||'Not detected')}</div></div><div class="signal"><div class="signal-k">GPU utilization</div><div class="signal-v">${m.gpu?.utilization_pct==null?'N/A':esc(m.gpu.utilization_pct+'%')}</div></div><div class="signal"><div class="signal-k">GPU temperature</div><div class="signal-v">${m.gpu?.temperature_c==null?'N/A':esc(m.gpu.temperature_c+' °C')}</div></div>`;
  } catch(e) { root.innerHTML=`<div class="callout red">${esc(e.message)}</div>`; }
}

async function loadInventory() {
  try {
    const d=await api('/api/inventory?include_ollama=true'); state.inventory=d.models || (d.directories||[]).flatMap(x=>x.models||[]); renderInventory(); await loadDirs();
    document.getElementById('nav-model-count').textContent=state.inventory.length;
  } catch(e){ document.querySelector('#inventory-table tbody').innerHTML=`<tr><td colspan="8">${esc(e.message)}</td></tr>`; }
}
function inventoryFiltered(){ const q=document.getElementById('inv-search').value.toLowerCase(); const s=document.getElementById('inv-source').value; const f=document.getElementById('inv-format').value; const t=document.getElementById('inv-task').value; return state.inventory.filter(m=>(!q||`${m.name} ${m.owner} ${m.full_name}`.toLowerCase().includes(q))&&(!s||m.source===s)&&(!f||m.format===f)&&(!t||m.task_label===t)); }
function renderInventory(){
  const models=inventoryFiltered(); const total=models.reduce((a,m)=>a+Number(m.size_gb||0),0); document.getElementById('inventory-summary').textContent=`${models.length} models · ${total.toFixed(1)} GB`;
  const body=document.querySelector('#inventory-table tbody'); body.innerHTML=models.length?models.map(m=>`<tr><td><div class="model-title">${esc(m.name)}</div><div class="model-owner">${esc(m.owner||m.full_name||'')}</div></td><td><span class="badge blue">${esc(m.task_label||'Unknown')}</span></td><td>${esc(m.format)}</td><td><span class="badge ${m.dtype==='FP4'?'amber':''}">${esc(m.dtype)}</span></td><td>${m.params_b==null?'—':esc(m.params_b+'B')}</td><td>${fmtGB(m.size_gb)}</td><td><span class="badge ${m.source==='ollama'?'green':'amber'}">${esc(m.source)}</span></td><td><div class="actions">${m.source==='ollama'?'<button class="btn small" data-go="ollama">Open</button>':`<button class="btn small" data-serve-model="${esc(m.id)}">Serve</button><button class="btn danger small" data-delete-path="${esc(m.dir_path)}" data-delete-name="${esc(m.full_name||m.name)}">Delete</button>`}</div></td></tr>`).join(''):'<tr><td colspan="8"><div class="empty-state">No models match the filters.</div></td></tr>';
  setRoleUI();
}
async function loadDirs(){ try{const d=await api('/api/hf/inventory/dirs'); document.getElementById('inventory-dirs').innerHTML=d.dirs.map(x=>`<div class="service-row"><div class="iconbox">▣</div><div class="row-main"><div class="service-name">${esc(x.path)}</div><div class="service-meta">${x.default?'Default HuggingFace cache':'Custom scan directory'}</div></div>${x.default?'':`<button class="btn danger small" data-remove-dir="${esc(x.path)}">Remove</button>`}</div>`).join('');}catch(e){} }
function openDirDialog(){ modal('Add model directory',`<label class="label" for="modal-dir-path">Directory path</label><input class="input w100" id="modal-dir-path" placeholder="/mnt/models"><div class="callout amber mt12">The directory must already exist on the Model Manager host. System root directories are blocked.</div>`,`<button class="btn" data-action="modal-close">Cancel</button><button class="btn primary" data-action="dir-save">Add directory</button>`); }
async function saveDir(){ try{await api('/api/hf/inventory/dirs',{method:'POST',json:{path:document.getElementById('modal-dir-path').value}}); closeModal(); toast('Directory added','ok'); loadInventory();}catch(e){toast(e.message,'error')} }
async function removeDir(path){ if(!confirm(`Remove ${path} from inventory scanning? No files will be deleted.`))return; try{await api('/api/hf/inventory/dirs?path='+encodeURIComponent(path),{method:'DELETE'}); toast('Directory removed','ok'); loadInventory();}catch(e){toast(e.message,'error')} }
async function deleteInventory(path,name){ if(!confirm(`Delete ${name} from disk?\n\nThis cannot be undone.`))return; try{await api('/api/hf/inventory/delete',{method:'POST',json:{path}}); toast('Model deleted','ok'); loadInventory();}catch(e){toast(e.message,'error')} }

async function loadOllama(){
  const root=document.getElementById('ollama-models'); root.innerHTML='<div class="empty-state">Loading...</div>';
  try{const d=await api('/api/ollama/models'); const ms=d.models||[]; root.innerHTML=ms.length?ms.map(m=>`<div class="panel"><div class="panel-head"><div class="panel-title">${esc(m.name)}</div><div class="actions"><span class="badge green"><span class="status-dot ok"></span>Installed</span></div></div><div class="panel-body"><div class="service-meta">${esc(m.details?.parameter_size||'')} · ${esc(m.details?.quantization_level||'')} · ${fmtGB((m.size||0)/1e9)}</div><div class="actions mt16"><button class="btn danger small" data-ollama-delete="${esc(m.name)}">Delete</button></div></div></div>`).join(''):'<div class="empty-state">No Ollama models installed.</div>';}catch(e){root.innerHTML=`<div class="callout red">${esc(e.message)}</div>`} setRoleUI();
}
async function ollamaPull(){ const name=document.getElementById('ollama-pull-name').value.trim(); if(!name)return; const logEl=document.getElementById('ollama-progress-log'),bar=document.getElementById('ollama-progress-bar'); logEl.textContent=`Starting ${name}...`; bar.style.width='2%'; try{const r=await fetch('/api/ollama/pull',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf},body:JSON.stringify({name})}); if(!r.ok)throw new Error((await r.json()).detail||`HTTP ${r.status}`); await readSSE(r,ev=>{ if(ev.total&&ev.completed){bar.style.width=`${Math.min(100,ev.completed/ev.total*100)}%`;} if(ev.status)logEl.textContent+=`\n${ev.status}`; if(ev.error)throw new Error(ev.error);}); bar.style.width='100%'; toast('Ollama pull complete','ok'); loadOllama(); loadInventory();}catch(e){toast(e.message,'error');logEl.textContent+=`\nERROR: ${e.message}`;} }
async function ollamaDelete(name){if(!confirm(`Delete ${name}?`))return;try{await api('/api/ollama/models/'+encodeURIComponent(name),{method:'DELETE'});toast('Ollama model deleted','ok');loadOllama();}catch(e){toast(e.message,'error')}}

async function searchHF(){ const q=document.getElementById('hf-query').value.trim(); if(!q)return; const root=document.getElementById('hf-results');root.innerHTML='<div class="empty-state">Searching...</div>';try{const p=new URLSearchParams({q,sort:document.getElementById('hf-sort').value,limit:'20'}); const type=document.getElementById('hf-type').value;if(type)p.set('pipeline_tag',type); const d=await api('/api/hf/search?'+p); state.hfResults=d.models||[]; root.innerHTML=state.hfResults.length?state.hfResults.map((m,i)=>`<div class="search-result"><div class="row-main"><div class="search-result-title">${esc(m.id)}</div><div class="search-result-meta">↓ ${fmtNum(m.downloads)} · ♥ ${fmtNum(m.likes)} · ${esc(m.task_label||'Unknown')}</div><div class="search-result-tags">${(m.tags||[]).slice(0,10).map(t=>`<span class="badge">${esc(t)}</span>`).join('')}</div><div class="search-result-actions"><button class="btn primary small" data-hf-download="${esc(m.id)}">⇩ Download</button><button class="btn small" data-hf-details="${i}">Files & variants</button></div><div class="details-pane hidden" id="hf-details-${i}"></div></div></div>`).join(''):'<div class="empty-state">No results.</div>';}catch(e){root.innerHTML=`<div class="callout red">${esc(e.message)}</div>`} setRoleUI();}
async function hfDetails(index){const m=state.hfResults[index],box=document.getElementById(`hf-details-${index}`);if(!box)return; if(!box.classList.contains('hidden')){box.classList.add('hidden');return;}box.classList.remove('hidden');box.innerHTML='Loading...';try{const [owner,name]=m.id.split('/',2);const [files,variants]=await Promise.all([api(`/api/hf/model/${encodeURIComponent(owner)}/${encodeURIComponent(name)}/files`),api('/api/hf/search/variants?model_id='+encodeURIComponent(m.id))]);box.innerHTML=`<div class="screen-title">Files</div><div class="file-list">${(files.files||[]).slice(0,50).map(f=>`<div class="file-row"><span>${esc(f.name)}</span><span>${fmtBytes(f.size)}</span></div>`).join('')}</div><div class="screen-title mt12">Quantized variants</div><div class="actions mt12">${(variants.variants||[]).map(v=>`<button class="btn small" data-hf-download="${esc(v.id)}">${esc(v.type)} · ${esc(v.id)}</button>`).join('')||'<span class="muted">None discovered</span>'}</div>`;}catch(e){box.innerHTML=`<div class="callout red">${esc(e.message)}</div>`} setRoleUI();}
function prepareDownload(repo){document.getElementById('download-repo').value=repo;go('downloads');}
async function hfDownload(){const repo=document.getElementById('download-repo').value.trim(),dir=document.getElementById('download-dir').value.trim();if(!repo)return;const logEl=document.getElementById('download-log'),bar=document.getElementById('download-progress-bar');logEl.textContent=`Starting ${repo}...`;bar.style.width='1%';try{const r=await fetch('/api/hf/download',{method:'POST',headers:{'Content-Type':'application/json','X-CSRF-Token':state.csrf},body:JSON.stringify({repo_id:repo,local_dir:dir||null})});if(!r.ok)throw new Error((await r.json()).detail||`HTTP ${r.status}`);await readSSE(r,ev=>{if(ev.progress){bar.style.width=`${ev.progress.pct}%`;logEl.textContent=`[${ev.progress.idx}/${ev.progress.total_files}] ${ev.progress.file}\n${ev.progress.pct}% · ${ev.progress.done_mb} / ${ev.progress.total_mb} MB · ${ev.progress.speed}`;}else if(ev.file_error){logEl.textContent+=`\nFailed: ${ev.file_error.name}`;}else if(ev.status==='complete'){bar.style.width='100%';logEl.textContent+=`\nComplete → ${ev.path}\nAverage ${ev.avg_speed} · ${ev.elapsed}`;toast('HuggingFace download complete','ok');}else if(ev.status==='error'){throw new Error(ev.error);}else if(ev.log){logEl.textContent+=`\n${ev.log}`;}});loadInventory();}catch(e){toast(e.message,'error');logEl.textContent+=`\nERROR: ${e.message}`;}}
async function readSSE(response,onEvent){const reader=response.body.getReader(),dec=new TextDecoder();let buf='';while(true){const {done,value}=await reader.read();if(done)break;buf+=dec.decode(value,{stream:true});let pos;while((pos=buf.indexOf('\n'))>=0){const line=buf.slice(0,pos).trimEnd();buf=buf.slice(pos+1);if(line.startsWith('data: ')){try{onEvent(JSON.parse(line.slice(6)));}catch(e){if(e instanceof SyntaxError){}else throw e;}}}}}

async function loadDeployments(){try{const d=await api('/api/compose/deployments');state.deployments=d.deployments||[];document.getElementById('nav-deploy-count').textContent=state.deployments.length;const body=document.querySelector('#deployment-table tbody');body.innerHTML=state.deployments.length?state.deployments.map(x=>{const key=`${x.engine}|${x.slug}|${x.node_id??''}`;return `<tr><td><div class="model-title">${esc(x.name||x.slug)}</div><div class="model-owner">${esc(x.slug)}</div></td><td><span class="badge blue">${esc(SERVICE_LABELS[x.engine]||x.engine)}</span></td><td>${esc(x.model_name||'—')}</td><td>${esc(x.node||'local')}</td><td><span class="${x.fit_status==='good'?'fit-good':x.fit_status==='risk'?'fit-risk':'fit-tight'}">${esc(x.fit_status||'—')}</span></td><td>${esc(x.port||'—')}</td><td id="dep-status-${esc(x.engine)}-${esc(x.slug)}-${esc(x.node_id??'local')}"><span class="badge">Checking</span></td><td><div class="actions"><button class="btn primary small" data-dep-up="${esc(key)}">Start</button><button class="btn small" data-dep-down="${esc(key)}">Stop</button><button class="btn small" data-dep-logs="${esc(key)}">Logs</button>${x.node_id==null&&x.expose_litellm!==false&&['vllm','sglang','llamacpp'].includes(x.engine)?`<button class="btn small" data-dep-route="${esc(key)}">LiteLLM</button>`:''}<button class="btn danger small" data-dep-remove="${esc(key)}">Archive</button></div></td></tr>`}).join(''):'<tr><td colspan="8"><div class="empty-state">No Compose deployments. Use Compose Builder to create one.</div></td></tr>'; setRoleUI(); for(const x of state.deployments)loadDepStatus(x);}catch(e){toast(e.message,'error')}}
async function loadDepStatus(x){try{const qs=x.node_id!=null?'?node_id='+encodeURIComponent(x.node_id):'';const st=await api(`/api/compose/deployments/${encodeURIComponent(x.engine)}/${encodeURIComponent(x.slug)}/status${qs}`);const el=document.getElementById(`dep-status-${x.engine}-${x.slug}-${x.node_id??'local'}`);if(el)el.innerHTML=`<span class="badge ${st.running?'green':'red'}"><span class="status-dot ${st.running?'ok':'err'}"></span>${st.running?'Running':'Stopped'}</span>`;}catch(e){}}
async function depAction(kind,key){const [engine,slug,node]=key.split('|');const qs=node?'?node_id='+encodeURIComponent(node):'';try{const d=await api(`/api/compose/deployments/${encodeURIComponent(engine)}/${encodeURIComponent(slug)}/${kind}${qs}`,{method:'POST'});toast(d.ok?`Deployment ${kind==='up'?'started':'stopped'}`:d.output,d.ok?'ok':'error');loadDeployments();loadDashboard();}catch(e){toast(e.message,'error')}}
async function depLogs(key){
  const [engine,slug,node]=key.split('|');
  const qs='?lines=300'+(node?'&node_id='+encodeURIComponent(node):'');
  try{
    const d=await api(`/api/compose/deployments/${engine}/${slug}/logs${qs}`);
    document.getElementById('generic-modal').classList.add('log-viewer');
    modal(
      `Logs · ${slug}`,
      `<pre class="code-pane">${esc((d.lines||[]).join('\n'))}</pre>`,
      `<button class="btn" data-action="modal-close">Close</button>`
    );
  }catch(e){
    toast(e.message,'error');
  }
}
async function depRemove(key){const [engine,slug,node]=key.split('|');if(!confirm(`Stop and archive ${slug}?`))return;const qs=node?'?node_id='+encodeURIComponent(node):'';try{await api(`/api/compose/deployments/${engine}/${slug}${qs}`,{method:'DELETE'});toast('Deployment archived','ok');loadDeployments();}catch(e){toast(e.message,'error')}}
function depRouteModal(key){const [engine,slug,node]=key.split('|');if(node)return toast('Central LiteLLM routing is currently available for local deployments only','error');modal(`LiteLLM route · ${slug}`,`<div class="stack"><div class="callout amber">This writes a concrete OpenAI-compatible route into your existing LiteLLM configuration, creates a rollback backup, and restarts LiteLLM.</div><p class="subtle">The generated serving stack uses a stable served-model name so the route does not depend on an internal checkpoint path.</p></div>`,`<button class="btn" data-action="modal-close">Cancel</button><button class="btn danger" data-route-remove="${esc(engine+'|'+slug)}">Remove route</button><button class="btn primary" data-route-add="${esc(engine+'|'+slug)}">Add / update route</button>`)}
async function depRoute(kind,key){const [engine,slug]=key.split('|');try{const d=await api(`/api/compose/deployments/${encodeURIComponent(engine)}/${encodeURIComponent(slug)}/litellm`,{method:kind==='add'?'POST':'DELETE'});closeModal();const restart=d.restart||{};toast(`${kind==='add'?'Route saved':'Route removed'}${restart.ok===false?' · LiteLLM restart failed':''}`,restart.ok===false?'error':'ok');loadRouting();}catch(e){toast(e.message,'error')}}

async function loadBuilderModels(){
  const nodeSel=document.getElementById('build-node'),modelSel=document.getElementById('build-model');
  const previous=modelSel.value; const nodeId=nodeSel.value;
  try{
    if(nodeId){const d=await api(`/api/nodes/${encodeURIComponent(nodeId)}/inventory`);state.builderInventory=d.models||[];}
    else{if(!state.inventory.length)await loadInventory();state.builderInventory=state.inventory.filter(m=>m.source!=='ollama');}
    modelSel.innerHTML=state.builderInventory.map(m=>`<option value="${esc(m.id)}">${esc(m.full_name||m.name)} · ${esc(m.format)} · ${esc(m.dtype||'Unknown')}${m.quant_method?' · '+esc(m.quant_method):''} · ${fmtGB(m.size_gb)}</option>`).join('');
    if(previous&&state.builderInventory.some(m=>m.id===previous))modelSel.value=previous;
    if(!state.builderInventory.length)modelSel.innerHTML='<option value="">No Compose-eligible models found on this node</option>';
  }catch(e){state.builderInventory=[];modelSel.innerHTML='<option value="">Remote inventory unavailable</option>';toast(e.message,'error');}
}
async function loadBuilder(){
  if(!state.nodes.length)await loadNodesOnly();
  const nodeSel=document.getElementById('build-node'),currentNode=nodeSel.value;
  nodeSel.innerHTML='<option value="">Local node</option>'+state.nodes.map(n=>`<option value="${n.id}">${esc(n.name)}</option>`).join('');
  if(currentNode&&state.nodes.some(n=>String(n.id)===String(currentNode)))nodeSel.value=currentNode;
  await loadBuilderModels();
}
async function generateCompose(){const model_id=document.getElementById('build-model').value;if(!model_id)return toast('Select a model on the target node','error');const payload={model_id,engine:document.getElementById('build-engine').value,name:document.getElementById('build-name').value||null,context_length:Number(document.getElementById('build-context').value),memory_reserve_gb:Number(document.getElementById('build-reserve').value),profile:document.getElementById('build-profile').value,bind_host:document.getElementById('build-bind').value,expose_litellm:document.getElementById('build-litellm').checked,node_id:document.getElementById('build-node').value?Number(document.getElementById('build-node').value):null};try{const plan=await api('/api/compose/generate',{method:'POST',json:payload});state.plan=plan;document.getElementById('yaml-code').textContent=plan.yaml;document.getElementById('save-plan-btn').disabled=false;const fitClass=plan.fit_status==='good'?'fit-good':plan.fit_status==='risk'?'fit-risk':'fit-tight';const quant=plan.quant_method||state.builderInventory.find(m=>m.id===model_id)?.quant_method||'auto / none';document.getElementById('builder-signals').innerHTML=`<div class="signal"><div class="signal-k">Estimated runtime</div><div class="signal-v">${fmtGB(plan.estimated_runtime_gb)}</div></div><div class="signal"><div class="signal-k">Memory budget</div><div class="signal-v">${fmtGB(plan.memory_budget_gb)}</div></div><div class="signal"><div class="signal-k">Fit</div><div class="signal-v ${fitClass}">${esc(plan.fit_status)}</div></div><div class="signal"><div class="signal-k">Quantization</div><div class="signal-v">${esc(quant)}</div></div><div class="signal"><div class="signal-k">Port exposure</div><div class="signal-v">${esc(plan.bind_host)}:${esc(plan.port)}</div></div>`;document.getElementById('builder-decision').innerHTML=`<strong>Generator decision:</strong> ${esc(plan.notes.join(' · '))}`;setRoleUI();}catch(e){toast(e.message,'error')}}
async function savePlan(){if(!state.plan)return;try{const nodeId=state.plan.node_id||null;const saved=await api('/api/compose/deployments',{method:'POST',json:{plan:state.plan,node_id:nodeId}});toast(`Saved ${saved.slug||state.plan.slug}`,'ok');state.plan=null;document.getElementById('save-plan-btn').disabled=true;go('deployments');}catch(e){toast(e.message,'error')}}
async function copyYaml(){try{await navigator.clipboard.writeText(document.getElementById('yaml-code').textContent);toast('YAML copied','ok')}catch(e){toast('Clipboard unavailable','error')}}

async function loadRouting(){try{const [routes,cfg]=await Promise.all([api('/api/litellm/models'),api('/api/litellm/config')]);const ms=routes.data||[];document.getElementById('route-list').innerHTML=ms.length?ms.map(m=>`<div class="service-row"><div class="iconbox">⇄</div><div class="row-main"><div class="service-name">${esc(m.id)}</div><div class="service-meta">OpenAI-compatible route</div></div><span class="badge green">Active</span></div>`).join(''):'<div class="empty-state">No routes reported by LiteLLM.</div>';document.getElementById('litellm-config').textContent=cfg._raw||JSON.stringify(cfg,null,2);}catch(e){toast(e.message,'error')}}
async function applyWildcard(){try{const d=await api('/api/litellm/apply-wildcard',{method:'POST'});toast(d.message||'Wildcard applied','ok');loadRouting();}catch(e){toast(e.message,'error')}}

async function loadEngines(){const grid=document.getElementById('engine-grid');grid.innerHTML='<div class="empty-state">Loading engines...</div>';const cards=[];for(const k of ENGINE_KEYS){try{const [st,profiles]=await Promise.all([api(`/api/${k}/status`),api(`/api/${k}/profiles`)]);cards.push(`<div class="engine-card"><div class="engine-card-top"><span class="status-dot ${st.running?'ok':'err'}"></span><div class="row-main"><div class="service-name">${esc(SERVICE_LABELS[k])}</div><div class="service-meta">${st.running?`Running · ${esc(st.model||'service ready')}`:'Stopped'} · ${esc(state.config?.services?.[`${k}_base`]||'')}</div></div><span class="badge ${st.running?'green':'red'}">${st.running?'Online':'Offline'}</span></div><select class="select" id="engine-profile-${k}">${profiles.map(p=>`<option value="${esc(p.id)}">${esc(p.name)} · ${esc(p.kind)}</option>`).join('')||'<option value="">No deployment profiles</option>'}</select><div class="engine-actions"><button class="btn primary small" data-engine-start="${k}" ${profiles.length?'':'disabled'}>Start selected</button><button class="btn danger small" data-engine-stop="${k}" ${st.managed_running?'':'disabled'} title="${st.running&&!st.managed_running?'Detected service is not managed by DGX Model Manager v2':''}">Stop</button></div></div>`);}catch(e){cards.push(`<div class="engine-card"><div class="service-name">${esc(SERVICE_LABELS[k])}</div><div class="callout red mt12">${esc(e.message)}</div></div>`);}}grid.innerHTML=cards.join('');setRoleUI();}
async function engineStart(k){const profile=document.getElementById(`engine-profile-${k}`)?.value;if(!profile)return;try{await api(`/api/${k}/start`,{method:'POST',json:{profile}});toast(`${SERVICE_LABELS[k]} starting`,'ok');setTimeout(loadEngines,1200);}catch(e){toast(e.message,'error')}}
async function engineStop(k){if(!confirm(`Stop ${SERVICE_LABELS[k]}? Active inference requests will be interrupted.`))return;try{await api(`/api/${k}/stop`,{method:'POST'});toast(`${SERVICE_LABELS[k]} stopped`,'ok');setTimeout(loadEngines,1000);}catch(e){toast(e.message,'error')}}

async function loadLegacy(){const grid=document.getElementById('legacy-grid');const cards=[];for(const k of ENGINE_KEYS){try{const ps=(await api(`/api/${k}/profiles`)).filter(p=>p.kind==='legacy');cards.push(`<div class="panel"><div class="panel-head"><div class="panel-title">${esc(SERVICE_LABELS[k])}</div></div><div class="panel-body">${ps.length?ps.map(p=>`<div class="service-row"><div class="row-main"><div class="service-name">${esc(p.name)}</div><div class="service-meta">${esc(p.description)}</div></div><button class="btn small" data-legacy-start="${k}|${esc(p.id)}">Run</button></div>`).join(''):'<div class="empty-state">No legacy scripts detected.</div>'}</div></div>`);}catch(e){}}grid.innerHTML=cards.join('');setRoleUI();}

async function loadNodesOnly(){try{const d=await api('/api/nodes');state.nodes=d.nodes||[];const top=document.getElementById('top-node-select');top.innerHTML='<option value="local">Local node</option>'+state.nodes.map(n=>`<option value="${n.id}">${esc(n.name)}</option>`).join('');if(state.selectedNode!=='local'&&!state.nodes.some(n=>String(n.id)===String(state.selectedNode)))state.selectedNode='local';top.value=String(state.selectedNode);document.getElementById('nav-node-count').textContent=1+state.nodes.length;}catch(e){state.nodes=[];state.selectedNode='local'}}
async function loadCluster(){await loadNodesOnly();try{const d=await api('/api/dashboard');const m=d.metrics;document.getElementById('cluster-metrics').innerHTML=`<div class="metric"><div class="metric-label">Local node</div><div class="metric-value">${esc(m.hostname)}</div><div class="metric-foot"><span>${esc(m.platform_class)} · ${esc(m.ip)}</span></div></div><div class="metric"><div class="metric-label">Unified memory</div><div class="metric-value">${esc(m.memory_total_gb)}<span class="metric-unit">GB</span></div><div class="metric-foot"><span>${esc(m.memory_available_gb)} GB available</span></div></div><div class="metric"><div class="metric-label">Remote nodes</div><div class="metric-value">${state.nodes.length}</div><div class="metric-foot"><span>Optional agent enrollment</span></div></div>`;document.getElementById('node-list').innerHTML=`<div class="node-row"><span class="status-dot ok"></span><div class="row-main"><div class="node-name">${esc(m.hostname)} <span class="badge green">Local</span></div><div class="node-meta">${esc(m.ip)} · ${esc(m.platform_class)} · agent not required</div></div></div>`+state.nodes.map(n=>`<div class="node-row"><span class="status-dot ${n.last_seen?'ok':'warn'}"></span><div class="row-main"><div class="node-name">${esc(n.name)}</div><div class="node-meta">${esc(n.base_url)} · TLS verify ${n.verify_tls?'on':'off'}${n.last_seen?' · last seen '+esc(n.last_seen):''}</div></div><div class="actions role-admin-only"><button class="btn small" data-node-test="${n.id}">Test</button><button class="btn danger small" data-node-delete="${n.id}">Remove</button></div></div>`).join('');setRoleUI();}catch(e){toast(e.message,'error')}}
function addNodeModal(){modal('Add DGX node',`<div class="stack"><div><label class="label" for="node-name">Node name</label><input class="input w100" id="node-name" placeholder="spark-b"></div><div><label class="label" for="node-url">Agent URL</label><input class="input w100" id="node-url" placeholder="https://spark-b.example.invalid:8092"></div><div><label class="label" for="node-token">Enrollment token</label><input class="input w100" id="node-token" type="password" autocomplete="off"></div><div><label class="label" for="node-fingerprint">SHA-256 certificate fingerprint (for self-signed TLS)</label><input class="input w100" id="node-fingerprint" placeholder="64 hexadecimal characters"></div><label class="toggle-row"><div class="toggle-copy"><div class="toggle-title">Verify TLS certificate</div><div class="toggle-sub">Keep enabled for certificates trusted by the manager host.</div></div><input class="checkbox" id="node-verify" type="checkbox" checked></label></div>`,`<button class="btn" data-action="modal-close">Cancel</button><button class="btn primary" data-action="node-save">Add node</button>`)}
async function saveNode(){try{await api('/api/nodes',{method:'POST',json:{name:document.getElementById('node-name').value,base_url:document.getElementById('node-url').value,token:document.getElementById('node-token').value,verify_tls:document.getElementById('node-verify').checked,tls_fingerprint:document.getElementById('node-fingerprint').value}});closeModal();toast('Node added','ok');loadCluster();}catch(e){toast(e.message,'error')}}
async function testNode(id){try{const d=await api(`/api/nodes/${id}/test`);toast(`${d.name}: ${d.platform_class} · Compose ${d.compose_version||'n/a'}`,'ok');loadCluster();}catch(e){toast(e.message,'error')}}
async function deleteNode(id){if(!confirm('Remove this node from Model Manager? No workloads are changed.'))return;try{await api(`/api/nodes/${id}`,{method:'DELETE'});toast('Node removed','ok');loadCluster();}catch(e){toast(e.message,'error')}}

async function loadLogs(){try{const level=document.getElementById('log-level').value,search=document.getElementById('log-search').value;const [sys,logs,ll,dock]=await Promise.all([api('/api/debug/system'),api('/api/logs/app?'+new URLSearchParams({level,search,limit:'250'})),api('/api/logs/litellm?lines=150'),api('/api/debug/docker')]);document.getElementById('diagnostic-cards').innerHTML=`<div class="metric"><div class="metric-label">Host</div><div class="metric-value">${esc(sys.hostname)}</div><div class="metric-foot"><span>${esc(sys.architecture)} · Python ${esc(sys.python_version)}</span></div></div><div class="metric"><div class="metric-label">Docker</div><div class="metric-value">${sys.permissions.docker?'Ready':'No'}</div><div class="metric-foot"><span>Compose ${esc(sys.permissions.compose_version||'unavailable')}</span></div></div><div class="metric"><div class="metric-label">Uptime</div><div class="metric-value">${Math.floor(sys.uptime_seconds/60)}<span class="metric-unit">min</span></div><div class="metric-foot"><span>Started ${esc(sys.app_start_utc)}</span></div></div>`;document.getElementById('app-log-pane').innerHTML=(logs.entries||[]).map(x=>`<div class="log-line"><span class="log-time">${esc((x.ts||'').slice(11,19))}</span><span class="${x.level==='ERROR'?'log-err':x.level==='WARNING'?'log-warn':'log-info'}">${esc(x.level)}</span><span>${esc(x.msg)}</span></div>`).join('')||'<div class="muted">No application log entries.</div>';document.getElementById('litellm-log-pane').textContent=(ll.lines||[]).join('\n')||ll.error||'No log data.';document.getElementById('docker-pane').innerHTML=(dock.containers||[]).map(c=>`<div class="service-row"><span class="status-dot ${String(c.state||c.status).toLowerCase().includes('running')?'ok':'warn'}"></span><div class="row-main"><div class="service-name">${esc(c.name)}</div><div class="service-meta">${esc(c.image)} · ${esc(c.status)}</div></div></div>`).join('')||'<div class="empty-state">No containers reported.</div>';setRoleUI();}catch(e){toast(e.message,'error')}}
async function clearLogs(){if(!confirm('Clear the in-memory application log buffer?'))return;try{await api('/api/logs/app',{method:'DELETE'});loadLogs();}catch(e){toast(e.message,'error')}}

async function loadAccess(){
  if(state.user?.role!=='admin')return;
  try{
    const [u,t,a]=await Promise.all([
      api('/api/users'),
      api('/api/tokens'),
      api('/api/audit?limit=80')
    ]);

    const users=u.users||[];
    const activeAdmins=users.filter(x=>x.is_active&&x.role==='admin').length;

    document.getElementById('user-list').innerHTML=`
      <table class="user-table">
        <thead>
          <tr><th>User</th><th>Role</th><th>Status</th><th></th></tr>
        </thead>
        <tbody>
          ${users.map(x=>{
            const isSelf=Number(x.id)===Number(state.user?.id);
            const lastActiveAdmin=x.is_active&&x.role==='admin'&&activeAdmins<=1;
            return `
              <tr>
                <td>
                  <div class="service-name">
                    ${esc(x.display_name)}
                    ${isSelf?'<span class="badge amber">You</span>':''}
                  </div>
                  <div class="service-meta">${esc(x.username)}</div>
                </td>
                <td>
                  <select
                    class="select"
                    data-user-role="${x.id}"
                    ${lastActiveAdmin?'disabled title="At least one active administrator must remain"':''}
                  >
                    <option ${x.role==='viewer'?'selected':''}>viewer</option>
                    <option ${x.role==='operator'?'selected':''}>operator</option>
                    <option ${x.role==='admin'?'selected':''}>admin</option>
                  </select>
                </td>
                <td>
                  <span class="badge ${x.is_active?'green':'red'}">
                    ${x.is_active?'Active':'Disabled'}
                  </span>
                </td>
                <td>
                  <button
                    class="btn small"
                    data-user-toggle="${x.id}|${x.is_active?0:1}"
                    ${isSelf&&x.is_active?'disabled title="You cannot disable your own account"':''}
                  >
                    ${x.is_active?'Disable':'Enable'}
                  </button>
                </td>
              </tr>
            `;
          }).join('')}
        </tbody>
      </table>`;

    document.getElementById('token-list').innerHTML=(t.tokens||[]).map(x=>`
      <div class="service-row">
        <div class="row-main">
          <div class="service-name">${esc(x.name)}</div>
          <div class="service-meta">${esc(x.role)} · owner ${esc(x.username)} · ${esc(x.created_at)}</div>
        </div>
        <button class="btn danger small" data-token-delete="${x.id}">Revoke</button>
      </div>
    `).join('')||'<div class="empty-state">No API tokens.</div>';

    document.getElementById('audit-list').innerHTML=(a.entries||[]).map(x=>`
      <div class="activity-row">
        <div class="row-main">
          <div class="service-name">${esc(x.action)} · ${esc(x.target||'')}</div>
          <div class="service-meta">${esc(x.ts)} · ${esc(x.username||'system')} · ${esc(x.source_ip||'')}</div>
        </div>
      </div>
    `).join('')||'<div class="empty-state">No audit events.</div>';
  }catch(e){
    toast(e.message,'error');
  }
}

function userModal(){modal('Create account',`<div class="stack"><div><label class="label" for="new-user">Username</label><input class="input w100" id="new-user"></div><div><label class="label" for="new-display">Display name</label><input class="input w100" id="new-display"></div><div><label class="label" for="new-password">Password</label><input class="input w100" id="new-password" type="password" placeholder="Minimum 12 characters"></div><div><label class="label" for="new-role">Role</label><select class="select w100" id="new-role"><option>viewer</option><option>operator</option><option>admin</option></select></div></div>`,`<button class="btn" data-action="modal-close">Cancel</button><button class="btn primary" data-action="user-save">Create account</button>`)}
async function saveUser(){try{await api('/api/users',{method:'POST',json:{username:document.getElementById('new-user').value,display_name:document.getElementById('new-display').value,password:document.getElementById('new-password').value,role:document.getElementById('new-role').value}});closeModal();toast('Account created','ok');loadAccess();}catch(e){toast(e.message,'error')}}
async function updateUser(id,payload){try{await api(`/api/users/${id}`,{method:'PATCH',json:payload});toast('User updated','ok');loadAccess();}catch(e){toast(e.message,'error')}}
function tokenModal(){modal('Create API token',`<div class="stack"><div><label class="label" for="token-name">Token name</label><input class="input w100" id="token-name" placeholder="automation"></div><div><label class="label" for="token-role">Maximum role</label><select class="select w100" id="token-role"><option>viewer</option><option>operator</option><option>admin</option></select></div><div class="callout amber">The token is displayed once. Store it in a secrets manager.</div></div>`,`<button class="btn" data-action="modal-close">Cancel</button><button class="btn primary" data-action="token-save">Create token</button>`)}
async function saveToken(){try{const d=await api('/api/tokens',{method:'POST',json:{name:document.getElementById('token-name').value,role:document.getElementById('token-role').value}});document.getElementById('modal-title').textContent='API token created';document.getElementById('modal-body').innerHTML=`<div class="callout green">Copy this value now. It cannot be recovered later.</div><pre class="code-pane small mt12">${esc(d.token)}</pre>`;document.getElementById('modal-foot').innerHTML='<button class="btn primary" data-action="modal-close">Done</button>';loadAccess();}catch(e){toast(e.message,'error')}}
async function deleteToken(id){if(!confirm('Revoke this API token?'))return;try{await api(`/api/tokens/${id}`,{method:'DELETE'});toast('Token revoked','ok');loadAccess();}catch(e){toast(e.message,'error')}}

async function loadSettings(){if(state.user?.role!=='admin')return;try{state.config=await api('/api/config');const c=state.config;document.getElementById('settings-name').value=c.app.display_name||'';document.getElementById('settings-legacy').checked=!!c.app.legacy_scripts_enabled;document.getElementById('settings-registration').checked=!!c.app.allow_registration;document.getElementById('settings-bind').value=c.compose.bind_host||'127.0.0.1';document.getElementById('settings-reserve').value=c.compose.default_memory_reserve_gb??24;document.getElementById('settings-context').value=c.compose.default_context_length??32768;document.getElementById('service-settings').innerHTML=Object.entries(c.services).map(([k,v])=>{
  const type=k.replace(/_base$/,'');
  const label=SERVICE_LABELS[type]||k;
  return `<div>
    <label class="label" for="svc-${esc(k)}">${esc(label)}</label>
    <div class="toolbar">
      <input class="input" id="svc-${esc(k)}" value="${esc(v)}">
      <button class="btn small" data-test-service="${esc(type)}">Test</button>
      <span id="svc-test-${esc(type)}" aria-live="polite"></span>
    </div>
  </div>`;
}).join('');document.getElementById('image-settings').innerHTML=Object.entries(c.compose.images||{}).map(([k,v])=>`<div><label class="label" for="img-${esc(k)}">${esc(SERVICE_LABELS[k]||k)}</label><input class="input w100" id="img-${esc(k)}" value="${esc(v||'')}" placeholder="No default image configured"></div>`).join('');document.getElementById('security-summary').textContent=`HTTPS required: ${c.security.require_https?'yes':'no'} · Secure cookies: ${c.security.cookie_secure?'yes':'no'} · Public service targets: ${c.security.allow_public_service_targets?'allowed':'blocked'} · TLS: ${c.tls.enabled?'enabled':'disabled'}`;document.getElementById('legacy-nav').classList.toggle('hidden',!c.app.legacy_scripts_enabled);setRoleUI();}catch(e){toast(e.message,'error')}}
async function saveSettings(){const services={};document.querySelectorAll('#service-settings input[id^="svc-"]').forEach(i=>services[i.id.slice(4)]=i.value.trim());const images={};document.querySelectorAll('#image-settings input[id^="img-"]').forEach(i=>images[i.id.slice(4)]=i.value.trim());try{state.config=await api('/api/config',{method:'PUT',json:{display_name:document.getElementById('settings-name').value,legacy_scripts_enabled:document.getElementById('settings-legacy').checked,allow_registration:document.getElementById('settings-registration').checked,services,compose:{bind_host:document.getElementById('settings-bind').value,default_memory_reserve_gb:Number(document.getElementById('settings-reserve').value),default_context_length:Number(document.getElementById('settings-context').value),images}}});toast('Settings saved','ok');document.getElementById('legacy-nav').classList.toggle('hidden',!state.config.app.legacy_scripts_enabled);loadDashboard();}catch(e){toast(e.message,'error')}}
async function testService(type,btn){
  const input=document.getElementById(`svc-${type}_base`)||document.getElementById(`svc-${type}`);
  if(!input)return;

  const label=SERVICE_LABELS[type]||type;
  const result=document.getElementById(`svc-test-${type}`);

  btn.disabled=true;
  if(result)result.innerHTML='<span class="badge">Testing…</span>';

  try{
    const d=await api('/api/test-service',{
      method:'POST',
      json:{url:input.value,type}
    });

    if(d.ok){
      const latency=d.latency_ms??'—';
      if(result)result.innerHTML=`<span class="badge green">✓ ${esc(latency)} ms</span>`;
      toast(`${label}: ${latency} ms`,'ok');
    }else{
      const reason=d.error||(d.status_code?`HTTP ${d.status_code}`:'failed');
      if(result)result.innerHTML=`<span class="badge red">✕ ${esc(reason)}</span>`;
      toast(`${label}: ${reason}`,'error');
    }
  }catch(e){
    if(result)result.innerHTML='<span class="badge red">✕ Error</span>';
    toast(`${label}: ${e.message}`,'error');
  }finally{
    btn.disabled=false;
  }
}

function userMenu(){if(!state.user)return;modal('Signed-in account',`<dl class="key-value"><dt>Display name</dt><dd>${esc(state.user.display_name)}</dd><dt>Username</dt><dd>${esc(state.user.username)}</dd><dt>Role</dt><dd><span class="badge amber">${esc(state.user.role)}</span></dd></dl>`,`<button class="btn" data-action="modal-close">Close</button><button class="btn danger" data-action="logout">Sign out</button>`)}

async function initialLoad(){await loadSettingsSafe();await loadNodesOnly();await Promise.all([loadDashboard(),loadInventory()]);}
async function loadSettingsSafe(){try{state.config=await api('/api/config');document.getElementById('legacy-nav').classList.toggle('hidden',!state.config.app.legacy_scripts_enabled);}catch(e){}}

function delegatedClick(ev){
  const el=ev.target.closest('button,a'); if(!el)return;
  if(el.dataset.go){ev.preventDefault();go(el.dataset.go);return;}
  const a=el.dataset.action;
  const actions={login,bootstrap,register:registerAccount,'show-register':showRegister,'show-login':showLogin,'refresh-dashboard':loadDashboard,'inventory-refresh':loadInventory,'open-dir-dialog':openDirDialog,'dir-save':saveDir,'ollama-refresh':loadOllama,'ollama-pull':ollamaPull,'hf-search':searchHF,'hf-download':hfDownload,'deployments-refresh':loadDeployments,'generate-compose':generateCompose,'save-plan':savePlan,'copy-yaml':copyYaml,'routing-refresh':loadRouting,'apply-wildcard':applyWildcard,'engines-refresh':loadEngines,'cluster-refresh':loadCluster,'node-add':addNodeModal,'node-save':saveNode,'logs-refresh':loadLogs,'logs-clear':clearLogs,'access-refresh':loadAccess,'user-add':userModal,'user-save':saveUser,'token-add':tokenModal,'token-save':saveToken,'settings-save':saveSettings,'modal-close':closeModal,logout};
  if(a&&actions[a]){ev.preventDefault();actions[a]();return;}
  if(el.dataset.serveModel){document.getElementById('build-model').value=el.dataset.serveModel;go('builder');setTimeout(()=>{document.getElementById('build-model').value=el.dataset.serveModel;},50);return;}
  if(el.dataset.deletePath){deleteInventory(el.dataset.deletePath,el.dataset.deleteName);return;}
  if(el.dataset.removeDir){removeDir(el.dataset.removeDir);return;}
  if(el.dataset.ollamaDelete){ollamaDelete(el.dataset.ollamaDelete);return;}
  if(el.dataset.hfDownload){prepareDownload(el.dataset.hfDownload);return;}
  if(el.dataset.hfDetails){hfDetails(Number(el.dataset.hfDetails));return;}
  if(el.dataset.depUp){depAction('up',el.dataset.depUp);return;} if(el.dataset.depDown){depAction('down',el.dataset.depDown);return;} if(el.dataset.depLogs){depLogs(el.dataset.depLogs);return;} if(el.dataset.depRoute){depRouteModal(el.dataset.depRoute);return;} if(el.dataset.routeAdd){depRoute('add',el.dataset.routeAdd);return;} if(el.dataset.routeRemove){depRoute('remove',el.dataset.routeRemove);return;} if(el.dataset.depRemove){depRemove(el.dataset.depRemove);return;}
  if(el.dataset.engineStart){engineStart(el.dataset.engineStart);return;} if(el.dataset.engineStop){engineStop(el.dataset.engineStop);return;}
  if(el.dataset.legacyStart){const [k,p]=el.dataset.legacyStart.split('|');api(`/api/${k}/start`,{method:'POST',json:{profile:p}}).then(()=>toast('Legacy script launched','ok')).catch(e=>toast(e.message,'error'));return;}
  if(el.dataset.nodeTest){testNode(Number(el.dataset.nodeTest));return;} if(el.dataset.nodeDelete){deleteNode(Number(el.dataset.nodeDelete));return;}
  if(el.dataset.userToggle){const [id,active]=el.dataset.userToggle.split('|');updateUser(Number(id),{is_active:active==='1'});return;}
  if(el.dataset.tokenDelete){deleteToken(Number(el.dataset.tokenDelete));return;}
  if(el.dataset.testService){testService(el.dataset.testService,el);return;}
}

function delegatedChange(ev){const el=ev.target;if(el.matches('[data-user-role]'))updateUser(Number(el.dataset.userRole),{role:el.value});if(['inv-search','inv-source','inv-format','inv-task'].includes(el.id))renderInventory();if(el.id==='top-node-select'){state.selectedNode=el.value||'local';if(state.currentPage==='dashboard')loadDashboard();}if(el.id==='build-node'){state.plan=null;document.getElementById('save-plan-btn').disabled=true;document.getElementById('yaml-code').textContent='# Generate a plan to preview Compose YAML';document.getElementById('builder-signals').innerHTML='';document.getElementById('builder-decision').textContent='Loading target-node inventory…';loadBuilderModels().then(()=>{document.getElementById('builder-decision').textContent='Select a model and generate a deployment plan.';});}}

window.addEventListener('DOMContentLoaded',async()=>{
  document.addEventListener('click',delegatedClick);document.addEventListener('change',delegatedChange);document.getElementById('inv-search').addEventListener('input',renderInventory);document.getElementById('user-menu-btn').addEventListener('click',userMenu);document.querySelectorAll('.nav-item').forEach(b=>b.addEventListener('click',()=>go(b.dataset.page)));document.getElementById('login-pass').addEventListener('keydown',e=>{if(e.key==='Enter')login()});document.getElementById('bootstrap-pass').addEventListener('keydown',e=>{if(e.key==='Enter')bootstrap()});document.getElementById('register-pass').addEventListener('keydown',e=>{if(e.key==='Enter')registerAccount()});document.getElementById('hf-query').addEventListener('keydown',e=>{if(e.key==='Enter')searchHF()});
  const authed=await initAuth(); if(authed)await initialLoad();
});
