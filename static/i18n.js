(function () {
  'use strict';

  // English remains the source language. Chinese is additive so existing
  // deployments keep the upstream UI and no API/security code is involved.
  const zh = {
    'Overview':'概览','Dashboard':'控制台','Models':'模型','Inventory':'模型清单','HF Browser':'HF 浏览器','Downloads':'下载','Serving':'服务','Deployments':'部署','Compose Builder':'Compose 构建器','LiteLLM Routes':'LiteLLM 路由','Engines':'推理引擎','Legacy Scripts':'旧脚本','System':'系统','Cluster':'集群','Logs & Diagnostics':'日志与诊断','Administration':'管理','Users & Access':'用户与权限','Settings':'设置','Documentation':'文档',
    'Local node':'本机节点','Selected node':'已选节点','Language':'语言','Sign in':'登录','Sign out':'退出登录','Refresh':'刷新','Open all':'查看全部','Save settings':'保存设置','Cancel':'取消','Close':'关闭','Done':'完成','Search':'搜索','Create user':'创建用户','Create account':'创建账户','Create viewer account':'创建 Viewer 账户','Back to sign in':'返回登录','Remove':'移除','Delete':'删除','Download':'下载','Pull':'拉取','Start':'启动','Stop':'停止','Run':'运行','Apply Ollama wildcard':'应用 Ollama 通配路由',
    'Infrastructure dashboard':'基础设施控制台','Model inventory':'模型清单','Inference engines':'推理引擎','LiteLLM routing':'LiteLLM 路由','Compose deployments':'Compose 部署','Users & access':'用户与权限','Application logs':'应用日志','Service health':'服务健康状态','Managed deployments':'受管部署','Platform summary':'平台摘要','Recent audit events':'最近审计事件','API tokens':'API token','Accounts':'账户','Service endpoints':'服务端点','Engine images':'引擎镜像','Compose defaults':'Compose 默认值',
    'Loading dashboard...':'正在加载控制台…','Loading...':'正在加载…','Loading security posture...':'正在加载安全状态…','Detecting host...':'正在检测主机…','Ready.':'就绪。','No Compose deployments yet.':'还没有 Compose 部署。','No deployment profiles':'没有部署配置','No legacy scripts detected.':'未检测到旧脚本。','No application log entries.':'没有应用日志记录。','No log data.':'没有日志数据。','No containers reported.':'没有检测到容器。','No model matches the current filters.':'没有模型符合当前筛选条件。','Authentication required':'需要登录','Session expired. Sign in again.':'会话已过期，请重新登录。','Sign in to continue.':'请登录后继续。','Checking authentication...':'正在检查登录状态…','Sign in to DGX Model Manager v2.':'登录 DGX Model Manager v2。','Create a Viewer account. An administrator can grant additional privileges later.':'创建 Viewer 账户。管理员之后可以授予更多权限。',
    'Model name — e.g. qwen3:8b':'模型名称，例如 qwen3:8b','Search models':'搜索模型','All sources':'全部来源','All formats':'全部格式','All tasks':'全部任务','Repository ID':'仓库 ID','Local directory (optional)':'本地目录（可选）','Leave blank for HF cache':'留空则使用 HF 缓存','Optional friendly name':'可选的显示名称','Default bind':'默认绑定地址','System reserve (GB)':'系统保留内存（GB）','Default context':'默认上下文长度','Display name':'显示名称','Username':'用户名','Password':'密码','Minimum 12 characters':'至少 12 个字符','Administrator username':'管理员用户名','One-time bootstrap token':'一次性 bootstrap token','Shown by setup.sh or stored in bootstrap.token':'由 setup.sh 显示，或保存在 bootstrap.token 中',
    'Legacy Script Mode':'旧脚本模式','Expose existing start_*.sh profiles without converting them.':'直接显示现有 start_*.sh 配置，不进行转换。','Allow self-registration':'允许自行注册','New users become Viewers. Recommended off for an untrusted LAN.':'新用户将成为 Viewer。非可信局域网建议关闭。','private/loopback targets by default':'默认使用私有 / loopback 目标','HTTPS required':'需要 HTTPS','Secure cookies':'安全 Cookie','Public service targets':'公开服务目标','TLS':'TLS',
    'The first account is an administrator and requires the one-time local bootstrap token. Administrators can create additional accounts. Optional self-registration creates Viewer accounts only and is disabled by default.':'第一个账户是管理员，需要一次性本地 bootstrap token。管理员可以创建其他账户。可选的自行注册仅创建 Viewer 账户，默认关闭。'
  };

  // Keep the source UI English, but cover the complete v2 screens and the
  // strings produced by the API-driven views as well. This is intentionally
  // additive: switching back to English remains lossless.
  Object.assign(zh, {
    'DGX Model Manager v2':'DGX Model Manager v2','Model Manager':'模型管理器','DGX Spark control plane':'DGX Spark 控制平面','COMPOSE-FIRST CONTROL PLANE':'以 Compose 为核心的控制平面',
    'authenticated control-plane view':'已认证的控制平面视图','Service health':'服务健康状态','Compose stacks':'Compose 堆栈','Open all':'查看全部','Platform summary':'平台摘要','Detecting host...':'正在检测主机…','Operational state for the selected DGX node, model inventory, inference services, and managed Compose deployments.':'所选 DGX 节点的运行状态、模型清单、推理服务和受管 Compose 部署。',
    'Existing HuggingFace cache and Ollama paths are retained. Custom directories can be added without moving model data.':'现有 HuggingFace 缓存和 Ollama 路径保持不变。添加自定义目录不会移动模型文件。','All sources':'全部来源','HF Cache':'HF 缓存','Custom':'自定义','All formats':'全部格式','safetensors':'safetensors','gguf':'GGUF','pytorch':'PyTorch','ollama':'Ollama','All tasks':'全部任务','Text Gen':'文本生成','Vision LLM':'视觉语言模型','Embedding':'嵌入','Audio':'音频','STT':'语音转文字','TTS':'文字转语音','Inventory':'模型清单','Add directory':'添加目录','Model':'模型','Task':'任务','Format':'格式','Dtype':'数据类型','Params':'参数量','Size':'大小','Source':'来源','Action':'操作','Scan directories':'扫描目录','No models match the filters.':'没有模型符合当前筛选条件。','Unknown':'未知','Open':'打开','Serve':'启动服务','Delete':'删除','Default HuggingFace cache':'默认 HuggingFace 缓存','Custom scan directory':'自定义扫描目录',
    'Pull, inspect, and remove Ollama models. The existing Ollama service and storage remain untouched by the v2 application install.':'拉取、查看和删除 Ollama 模型。v2 应用安装不会修改现有 Ollama 服务和存储。','Pull model':'拉取模型','Model name — e.g. qwen3:8b':'模型名称，例如 qwen3:8b','Pull':'拉取','Ready.':'就绪。','Loading...':'正在加载…','Installed':'已安装','Ollama pull complete':'Ollama 模型拉取完成','Ollama unreachable':'Ollama 无法连接','Ollama model deleted':'Ollama 模型已删除',
    'Browse HuggingFace':'浏览 HuggingFace','Search Hub metadata, inspect repository files, discover common quantized variants, then hand the result to Download or Compose Builder.':'搜索 Hub 元数据、查看仓库文件、发现常见量化变体，然后交给下载或 Compose 构建器。','All types':'全部类型','Text generation':'文本生成','Vision':'视觉','Embeddings':'嵌入','Speech recognition':'语音识别','Most downloads':'下载最多','Most likes':'点赞最多','Trending':'趋势','Recent':'最新','Search':'搜索','Search HuggingFace to begin.':'搜索 HuggingFace 以开始。','HuggingFace downloads':'HuggingFace 下载','Downloads default to':'下载默认保存到','Repository ID':'仓库 ID','Local directory (optional)':'本地目录（可选）','Download':'下载','HuggingFace download complete':'HuggingFace 下载完成',
    'Declarative model-serving stacks managed with Docker Compose. Existing shell scripts remain available only when Legacy Script Mode is enabled.':'通过 Docker Compose 管理的声明式模型服务堆栈。只有启用旧脚本模式后才会显示现有 Shell 脚本。','New deployment':'新建部署','Deployment':'部署','Engine':'引擎','Node':'节点','Fit':'适配度','Port':'端口','Status':'状态','Actions':'操作','No Compose deployments. Use Compose Builder to create one.':'还没有 Compose 部署，请使用 Compose 构建器创建。','Compose Builder':'Compose 构建器','Generate a DGX-aware Compose stack from model metadata and target-node capacity. The planner uses the existing model path rather than copying the checkpoint.':'根据模型元数据和目标节点容量生成适配 DGX 的 Compose 堆栈。规划器使用现有模型路径，不会复制模型文件。','Deployment inputs':'部署参数','Target node':'目标节点','Deployment name':'部署名称','Context length':'上下文长度','System reserve (GB)':'系统保留（GB）','Optimization profile':'优化配置','Conservative':'保守','Balanced':'均衡','Performance':'性能','Bind address':'绑定地址','Loopback — recommended':'回环地址 — 推荐','LAN — advanced':'局域网 — 高级','Prepare for LiteLLM routing':'准备接入 LiteLLM 路由','Records routing intent in deployment metadata.':'在部署元数据中记录路由意图。','Generate Compose':'生成 Compose','Select a model and generate a deployment plan.':'选择模型并生成部署计划。','Generated compose.yaml':'生成的 compose.yaml','Copy':'复制','Save deployment':'保存部署','Generate a deployment to preview YAML.':'生成部署后预览 YAML。','Deployment saved':'部署已保存','Select a model on the target node':'请在目标节点选择模型','No Compose-eligible models found on this node':'此节点没有符合 Compose 条件的模型','Remote inventory unavailable':'远程模型清单不可用',
    'Unified OpenAI-compatible routing is retained. Secrets in the LiteLLM configuration are redacted before they reach the browser.':'保留统一的 OpenAI 兼容路由。LiteLLM 配置中的敏感信息在发送到浏览器前会被隐藏。','Apply Ollama wildcard':'应用 Ollama 通配路由','Active routes':'活动路由','Redacted configuration':'已隐藏敏感信息的配置','No routes reported by LiteLLM.':'LiteLLM 没有报告路由。','Inference engines':'推理引擎','One control surface for vLLM, SGLang, llama.cpp, LocalAI, and ComfyUI. Compose deployments are the primary launch mechanism.':'统一管理 vLLM、SGLang、llama.cpp、LocalAI 和 ComfyUI。Compose 部署是主要启动方式。','No deployment profiles':'没有部署配置','Start selected':'启动所选部署','Stop':'停止','Running':'运行中','Stopped':'已停止','Offline':'离线','Online':'在线','service ready':'服务就绪','Compatibility':'兼容性','Legacy scripts':'旧脚本','Compatibility view for existing':'现有脚本的兼容视图：','profiles. Legacy Script Mode is disabled by default and never removes existing files.':'配置。旧脚本模式默认关闭，且不会删除现有文件。','Legacy scripts execute arbitrary shell commands with the permissions of the Model Manager service account. Keep this mode disabled unless you need migration compatibility.':'旧脚本会以 Model Manager 服务账户权限执行任意 Shell 命令。除非需要迁移兼容，否则请保持关闭。','No legacy scripts detected.':'未检测到旧脚本。','Start':'启动',
    'The local Spark works without an agent. Additional DGX Spark nodes can enroll through the optional authenticated node agent.':'本地 Spark 无需代理即可工作。其他 DGX Spark 节点可以通过可选的认证节点代理加入。','Add node':'添加节点','Nodes':'节点','Add DGX node':'添加 DGX 节点','Node name':'节点名称','Agent URL':'代理地址','Enrollment token':'注册令牌','SHA-256 certificate fingerprint (for self-signed TLS)':'SHA-256 证书指纹（用于自签名 TLS）','Verify TLS certificate':'验证 TLS 证书','Keep enabled for certificates trusted by the manager host.':'对于管理器主机信任的证书，请保持启用。','Add node':'添加节点','Node added':'节点已添加','Node removed':'节点已移除','Remove this node from Model Manager? No workloads are changed.':'从 Model Manager 中移除此节点？不会修改任何工作负载。',
    'Authenticated application, engine, LiteLLM, Docker, and host diagnostics. Sensitive configuration values are not returned to the browser.':'已认证的应用、引擎、LiteLLM、Docker 和主机诊断。敏感配置不会返回到浏览器。','Application logs':'应用日志','All levels':'全部级别','INFO':'信息','WARNING':'警告','ERROR':'错误','Refresh':'刷新','Clear':'清空','LiteLLM journal':'LiteLLM 日志','Docker containers':'Docker 容器','Host':'主机','Docker':'Docker','Ready':'就绪','No application log entries.':'没有应用日志记录。','No log data.':'没有日志数据。','No containers reported.':'没有检测到容器。','Clear the in-memory application log buffer?':'清空内存中的应用日志缓冲区？','Application log cleared':'应用日志已清空',
    'Local accounts, role-based access, session authentication, and scoped API tokens. Public self-registration is disabled by default.':'本地账户、基于角色的访问控制、会话认证和作用域 API token。公开自行注册默认关闭。','Create user':'创建用户','Accounts':'账户','API tokens':'API token','Recent audit events':'最近审计事件','Create account':'创建账户','Display name':'显示名称','Role':'角色','viewer':'查看者','operator':'操作员','admin':'管理员','Active':'启用','Disabled':'已停用','Enable':'启用','Disable':'停用','You':'你','No API tokens.':'没有 API token。','No audit events.':'没有审计事件。','Create API token':'创建 API token','Token name':'Token 名称','Maximum role':'最高角色','The token is displayed once. Store it in a secrets manager.':'Token 只显示一次，请保存到密钥管理器。','API token created':'API token 已创建','Copy this value now. It cannot be recovered later.':'请立即复制此值，之后无法恢复。','Done':'完成','Revoke this API token?':'撤销此 API token？','Token revoked':'API token 已撤销','Account created':'账户已创建','User updated':'用户已更新',
    'Security-sensitive runtime settings and service endpoints. The v2 test install uses its own configuration, database, service name, Compose directory, and port.':'安全敏感的运行时设置和服务端点。v2 测试安装使用独立的配置、数据库、服务名、Compose 目录和端口。','Save settings':'保存设置','Application':'应用','Legacy Script Mode':'旧脚本模式','Expose existing start_*.sh profiles without converting them.':'显示现有 start_*.sh 配置，不进行转换。','Allow self-registration':'允许自行注册','New users become Viewers. Recommended off for an untrusted LAN.':'新用户将成为查看者。非可信局域网建议关闭。','Compose defaults':'Compose 默认值','Default bind':'默认绑定','System reserve (GB)':'系统保留（GB）','Default context':'默认上下文','Service endpoints':'服务端点','private/loopback targets by default':'默认使用私有 / 回环目标','Engine images':'引擎镜像','No default image configured':'未配置默认镜像','Loading security posture...':'正在加载安全状态…','HTTPS required':'需要 HTTPS','Secure cookies':'安全 Cookie','Public service targets':'公开服务目标','blocked':'已阻止','allowed':'已允许','TLS':'TLS','enabled':'已启用','disabled':'已禁用','Settings saved':'设置已保存',
    'The repository includes a complete v2 README, security model, migration procedure, multi-node agent guide, Compose Builder notes, and upgrade/promotion workflow.':'仓库包含完整的 v2 README、安全模型、迁移流程、多节点代理指南、Compose 构建器说明以及升级/发布流程。','Open field manual ↗':'打开使用手册 ↗','Migration':'迁移','Coexistence first':'优先共存','v2 defaults to port 8091 and service':'v2 默认使用 8091 端口和服务','It can import v1 paths and service URLs without modifying the running v1 application.':'它可以导入 v1 路径和服务地址，不会修改正在运行的 v1 应用。','Security':'安全','Untrusted LAN':'不可信局域网','Authenticated reads and writes, Argon2id passwords, HttpOnly sessions, CSRF, TLS, RBAC, redaction, and audit events replace the old mutation-only API key model.':'认证读写、Argon2id 密码、HttpOnly 会话、CSRF、TLS、RBAC、敏感信息隐藏和审计事件取代了旧的仅修改 API key 模型。','Serving':'服务','Compose-first':'以 Compose 为核心','Generated deployments live in the v2 data directory and reference existing model caches. Legacy script files are left in place for rollback.':'生成的部署保存在 v2 数据目录中，并引用现有模型缓存。旧脚本文件会保留以便回滚。',
    'Secure v2 control plane':'安全的 v2 控制平面','Checking authentication...':'正在检查身份认证…','Username':'用户名','Password':'密码','Sign in':'登录','Create viewer account':'创建查看者账户','Back to sign in':'返回登录','Administrator username':'管理员用户名','One-time bootstrap token':'一次性 bootstrap token','Create administrator':'创建管理员','First-run setup: create the administrator account.':'首次设置：创建管理员账户。','Account created. Sign in to continue.':'账户已创建，请登录继续。','Signed out.':'已退出登录。','Session expired. Sign in again.':'会话已过期，请重新登录。','Sign in to continue.':'请登录后继续。','Sign in to DGX Model Manager v2.':'登录 DGX Model Manager v2。','Create a Viewer account. An administrator can grant additional privileges later.':'创建查看者账户。管理员之后可以授予更多权限。','Minimum 12 characters':'至少 12 个字符','Operator or Admin role required':'需要操作员或管理员角色','At least one active administrator must remain':'至少需要保留一名启用的管理员','You cannot disable your own account':'不能停用自己的账户','Signed-in account':'当前登录账户','Close':'关闭','Cancel':'取消','Remove':'移除','No Compose deployments yet.':'还没有 Compose 部署。','Compose unavailable':'Compose 不可用','Local':'本地','agent not required':'无需代理','Target-node service':'目标节点服务','N/A':'不适用','Not detected':'未检测到','GPU utilization':'GPU 利用率','GPU temperature':'GPU 温度','Unified memory':'统一内存','System memory':'系统内存','CPU load':'CPU 负载','Model storage':'模型存储','Disk usage':'磁盘使用率','Remote nodes':'远程节点','logical cores':'逻辑核心','available':'可用','free':'剩余','discovered models':'已发现模型','No':'否','yes':'是','Failed':'失败','Testing…':'测试中…','Error':'错误','failed':'失败','HTTP':'HTTP'
  });

  const preferred = localStorage.getItem('dmm-language');
  let current = preferred || ((navigator.language || '').toLowerCase().startsWith('zh') ? 'zh-CN' : 'en');
  const reverse = Object.fromEntries(Object.entries(zh).map(([en, translated]) => [translated, en]));
  function translateDynamic(value) {
    if (current !== 'zh-CN') return value;
    let out = value;
    out = out.replace(/^(\d+) models · ([\d.]+) GB$/, '$1 个模型 · $2 GB');
    out = out.replace(/^(\d+) discovered models$/, '发现 $1 个模型');
    out = out.replace(/^(\d+) logical cores$/, '$1 个逻辑核心');
    out = out.replace(/^(\d+(?:\.\d+)?) GB available$/, '$1 GB 可用');
    out = out.replace(/^(\d+(?:\.\d+)?) GB free$/, '$1 GB 剩余');
    out = out.replace(/^Started (.+)$/, '启动于 $1');
    out = out.replace(/^last seen (.+)$/, '最后发现于 $1');
    out = out.replace(/^Target-node service$/, '目标节点服务');
    out = out.replace(/^Compose (.+)$/, 'Compose $1');
    out = out.replace(/^Saved (.+)$/, '已保存 $1');
    out = out.replace(/^Starting (.+)\.\.\.$/, '正在启动 $1…');
    out = out.replace(/^ERROR: (.+)$/, '错误：$1');
    out = out.replace(/^Complete → (.+)$/, '完成 → $1');
    out = out.replace(/^Loading target-node inventory…$/, '正在加载目标节点模型清单…');
    out = out.replace(/^Select a model and generate a deployment plan\.$/, '请选择模型并生成部署计划。');
    out = out.replace(/^Generator decision:$/, '生成器决策：');
    out = out.replace(/^Estimated runtime$/, '预计运行内存');
    out = out.replace(/^Memory budget$/, '内存预算');
    out = out.replace(/^Port exposure$/, '端口暴露');
    out = out.replace(/^Quantization$/, '量化方式');
    out = out.replace(/^Fit$/, '适配度');
    out = out.replace(/^([\d.]+) ms$/, '$1 毫秒');
    return out;
  }
  const dictionary = value => {
    const source = current === 'zh-CN' ? value : (reverse[value] || value);
    return current === 'zh-CN' ? translateDynamic(zh[source] || value) : source;
  };

  function translate(root = document) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(node => {
      const raw = node.nodeValue || '', trimmed = raw.trim();
      if (!trimmed) return;
      const translated = dictionary(trimmed);
      if (translated !== trimmed) node.nodeValue = raw.replace(trimmed, translated);
    });
    root.querySelectorAll?.('[placeholder],[title],[aria-label]').forEach(el => ['placeholder','title','aria-label'].forEach(attr => {
      if (el.hasAttribute(attr)) el.setAttribute(attr, dictionary(el.getAttribute(attr)));
    }));
  }

  function setLanguage(next) {
    current = next === 'zh-CN' ? 'zh-CN' : 'en';
    localStorage.setItem('dmm-language', current);
    document.documentElement.lang = current;
    const select = document.getElementById('language-select');
    if (select) select.value = current;
    translate();
  }

  window.dmmI18n = { t: dictionary, setLanguage, translate };
  document.addEventListener('DOMContentLoaded', () => {
    setLanguage(current);
    document.getElementById('language-select')?.addEventListener('change', e => setLanguage(e.target.value));
    new MutationObserver(records => records.forEach(record => record.addedNodes.forEach(node => {
      if (node.nodeType === Node.ELEMENT_NODE) translate(node);
    }))).observe(document.body, { childList: true, subtree: true });
  });
})();
