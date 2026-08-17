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

  const preferred = localStorage.getItem('dmm-language');
  let current = preferred || ((navigator.language || '').toLowerCase().startsWith('zh') ? 'zh-CN' : 'en');
  const reverse = Object.fromEntries(Object.entries(zh).map(([en, translated]) => [translated, en]));
  const dictionary = value => {
    const source = current === 'zh-CN' ? value : (reverse[value] || value);
    return current === 'zh-CN' ? (zh[source] || value) : source;
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
