const $ = (selector) => document.querySelector(selector);
const state = {
  sessionId: null,
  sessionArchived: false,
  showArchived: false,
  sessions: [],
  messages: [],
  memories: [],
  summary: "",
  stats: {},
  tools: [],
  skills: [],
  files: [],
  trace: [],
  tab: "trace",
  busy: false,
  answerReceived: false,
  pendingText: "",
  model: "",
  models: [],
  modelProfile: sessionStorage.getItem("agent-model-profile") || "default",
  embedding: "",
  backend: ["sdk", "langchain", "langgraph"].includes(sessionStorage.getItem("agent-backend"))
    ? sessionStorage.getItem("agent-backend") : "langchain",
  workspaceId: sessionStorage.getItem("agent-workspace") || "default",
  workspaces: [],
};

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function icon(name) {
  const node = element("span", "icon");
  node.style.maskImage = `url('/icons/${name}.svg')`;
  return node;
}

function toast(message) {
  const node = $("#toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, 4200);
}

async function api(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "请求失败");
  return data;
}

function renderSessions() {
  const list = $("#session-list");
  list.replaceChildren();
  const sessions = state.sessions.filter((session) => Boolean(session.archived_at) === state.showArchived);
  $("#session-count").textContent = String(sessions.length);
  $("#session-list-title").textContent = state.showArchived ? "已归档" : "会话";
  const toggle = $("#toggle-archives");
  toggle.title = state.showArchived ? "返回会话" : "查看已归档会话";
  toggle.setAttribute("aria-label", toggle.title);
  toggle.setAttribute("aria-pressed", String(state.showArchived));
  toggle.classList.toggle("active", state.showArchived);
  if (!sessions.length) list.append(element("div", "side-empty", state.showArchived ? "暂无已归档会话" : "暂无会话"));
  for (const session of sessions) {
    const row = element("div", "session-row");
    const button = element("button", "session-item" + (session.session_id === state.sessionId ? " active" : ""));
    button.type = "button";
    button.title = session.title || session.session_id;
    button.append(icon("message-square"), element("span", "", session.title || "新会话"));
    button.addEventListener("click", () => {
      if (state.busy || session.session_id === state.sessionId) return;
      closePanels();
      loadState(session.session_id, true).catch((error) => toast(error.message));
    });
    const action = element("button", "session-action");
    action.type = "button";
    action.title = session.archived_at ? "恢复会话" : "归档会话";
    action.setAttribute("aria-label", action.title);
    action.append(icon(session.archived_at ? "archive-restore" : "archive"));
    action.disabled = state.busy;
    action.addEventListener("click", () => archiveSession(session));
    row.append(button, action);
    list.append(row);
  }
  const current = state.sessions.find((item) => item.session_id === state.sessionId);
  state.sessionArchived = Boolean(current?.archived_at);
  $("#session-title").textContent = current?.title || "新会话";
  $("#session-id").textContent = state.sessionId || "正在连接";
}

function renderWorkspaces() {
  const select = $("#workspace-select");
  select.replaceChildren();
  for (const workspace of state.workspaces) {
    const option = element("option", "", workspace.name);
    option.value = workspace.workspace_id;
    option.selected = workspace.workspace_id === state.workspaceId;
    select.append(option);
  }
  const current = state.workspaces.find((item) => item.workspace_id === state.workspaceId);
  if (current) select.title = current.path;
}

function renderFiles() {
  const list = $("#file-list");
  list.replaceChildren();
  $("#file-count").textContent = String(state.files.length);
  if (!state.files.length) list.append(element("div", "side-empty", "暂无文件"));
  for (const file of state.files) {
    const button = element("button", "file-item");
    button.type = "button";
    button.title = file.path;
    button.append(icon("file-text"), element("span", "", file.path));
    button.addEventListener("click", () => openFile(file.path));
    list.append(button);
  }
}

function messageNode(message) {
  const isUser = message.role === "user";
  const row = element("article", "message " + (isUser ? "user" : "assistant"));
  if ((message.content || "").startsWith("（调用失败：")) row.classList.add("error");
  const label = element("div", "message-label");
  label.append(element("span", "avatar", isUser ? "你" : "A"));
  if (!isUser) label.append(element("span", "", "Agent"));
  if (!isUser && message.content) {
    const actions = element("span", "message-actions");
    const copy = element("button", "icon-button");
    copy.type = "button";
    copy.title = "复制回答";
    copy.setAttribute("aria-label", "复制回答");
    copy.append(icon("copy"));
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(message.content); toast("已复制"); }
      catch { toast("复制失败"); }
    });
    actions.append(copy);
    label.append(actions);
  }
  row.append(label, element("div", "message-body", message.content || ""));
  return row;
}

function renderMessages() {
  const container = $("#messages");
  const scroll = $("#chat-scroll");
  const nearBottom = scroll.scrollHeight - scroll.scrollTop - scroll.clientHeight < 120;
  container.replaceChildren();
  const visible = state.messages.filter((message) =>
    message.role === "user" || (message.role === "assistant" && !(message.tool_calls || []).length)
  );
  if (!visible.length && !state.busy) {
    const empty = element("div", "empty-chat");
    empty.append(element("span", "empty-mark", "A"), element("h1", "", "新会话"));
    container.append(empty);
  }
  for (const message of visible) container.append(messageNode(message));
  if (state.busy && !state.answerReceived) {
    const pending = element("article", "message assistant");
    const label = element("div", "message-label");
    label.append(element("span", "avatar", "A"), element("span", "", "Agent"));
    const line = element("div", "working-line");
    line.append(element("span", "working-dot"), element("span", "", "运行中"));
    pending.append(label, line);
    container.append(pending);
  }
  if (nearBottom || state.busy) requestAnimationFrame(() => { scroll.scrollTop = scroll.scrollHeight; });
}

function historyTrace(messages) {
  const items = [];
  const callNames = new Map();
  for (const message of messages) {
    if (message.role === "assistant" && message.tool_calls?.length) {
      const calls = message.tool_calls.map((call) => {
        callNames.set(call.id, call.function.name);
        let args = call.function.arguments;
        try { args = JSON.parse(args); } catch { /* keep raw arguments */ }
        return { name: call.function.name, args };
      });
      items.push({ type: "model_tool_calls", calls });
    } else if (message.role === "tool") {
      items.push({ type: "tool_result", name: callNames.get(message.tool_call_id) || "工具", result: message.content });
    }
  }
  return items.slice(-40);
}

function traceDescription(event) {
  switch (event.type) {
    case "context": return ["上下文", `${event.short_term_messages} 条短期消息 · 摘要${event.summary_used ? "已使用" : "未使用"} · 召回 ${(event.recalled_memories || []).length} 条记忆`, event.recalled_memories];
    case "tool_search": return [`工具检索 · 第 ${event.step} 步`, (event.available || []).join(" · "), event.candidates];
    case "model_start": return [`请求模型 · 第 ${event.step} 步`, "", null];
    case "model_retry": return [`模型重试 · 第 ${event.step} 步`, `${event.message} · ${event.delay} 秒后重连`, null];
    case "model_tool_calls": return ["模型选择工具", (event.calls || []).map((call) => call.name).join(" · "), event.calls];
    case "tool_start": return [`执行 ${event.name}`, "", event.args];
    case "tool_result": return [`${event.name} 返回`, String(event.result || "").slice(0, 180), event.result];
    case "skill_loaded": return [`读取 Skill · ${event.name || "未知 Skill"}`, event.loaded ? "已读取指令" : "读取失败", event.content];
    case "model_text": return ["模型输出", String(event.text || "").slice(0, 180), null];
    case "research_plan": return ["调研计划", (event.queries || []).join(" · "), event.queries];
    case "research_search": return [`搜索资料 · 第 ${event.round} 轮`, `${event.source_count || 0} 个来源`, { queries: event.queries, errors: event.errors }];
    case "research_critique": return [`资料评审 · 第 ${event.round} 轮`, event.sufficient ? "资料足够" : "需要补充搜索", { critique: event.critique, missing_topics: event.missing_topics, follow_up_queries: event.follow_up_queries }];
    case "research_write_start": return ["撰写调研报告", "评审完成", null];
    case "memory_write": return ["长期记忆", event.written ? `已写入 · ${event.kind}` : "本轮未写入", null];
    case "summary_start": return ["会话摘要", "开始归档旧消息", null];
    case "summary_saved": return ["会话摘要", `已保存 · ${event.length} 字`, null];
    case "summary_error": return ["摘要失败", event.message, null];
    case "error": return ["执行失败", event.message, null];
    case "answer": return ["回答完成", "", null];
    case "done": return ["本轮结束", "", null];
    default: return [event.type, "", null];
  }
}

function renderTrace(parent) {
  if (!state.trace.length) return parent.append(element("div", "inspect-empty", "暂无执行记录"));
  for (const event of state.trace) {
    const [title, detail, expanded] = traceDescription(event);
    const item = element("div", "trace-item" + ((event.type.startsWith("tool") || event.type === "skill_loaded") ? " tool" : "") + (event.type.includes("error") || (event.type === "skill_loaded" && !event.loaded) ? " error" : ""));
    item.append(element("div", "trace-title", title));
    if (detail) item.append(element("div", "trace-detail", detail));
    if (expanded && (event.type === "skill_loaded" || typeof expanded !== "string" || expanded.length > 180)) {
      const disclosure = element("details");
      disclosure.append(element("summary", "", event.type === "skill_loaded" ? "查看 Skill 指令" : "查看详情"));
      disclosure.append(element("pre", "", typeof expanded === "string" ? expanded : JSON.stringify(expanded, null, 2)));
      item.append(disclosure);
    }
    parent.append(item);
  }
}

function section(parent, title) {
  const node = element("section", "inspect-section");
  node.append(element("h2", "", title));
  parent.append(node);
  return node;
}

function renderMemory(parent) {
  const status = section(parent, "当前会话");
  const stats = element("div", "stat-row");
  for (const [value, label] of [
    [state.stats.active_messages || 0, "活跃消息"],
    [state.stats.active_memories || 0, "长期记忆"],
  ]) {
    const item = element("div", "stat");
    item.append(element("strong", "", String(value)), element("span", "", label));
    stats.append(item);
  }
  status.append(stats);
  const summary = section(parent, "会话摘要");
  summary.append(element("div", state.summary ? "summary-text" : "inspect-empty", state.summary || "暂无摘要"));
  const memories = section(parent, "长期记忆");
  if (!state.memories.length) memories.append(element("div", "inspect-empty", "暂无长期记忆"));
  for (const memory of state.memories) {
    const item = element("div", "memory-item");
    const top = element("div", "item-top");
    top.append(element("span", "badge", memory.kind), element("span", "", `#${memory.memory_id}`));
    item.append(top, element("div", "item-body", memory.content));
    memories.append(item);
  }
}

function renderTools(parent) {
  const tools = section(parent, `工具 · ${state.tools.length}`);
  for (const tool of state.tools) {
    const item = element("div", "tool-item");
    item.append(element("div", "item-top", tool.id), element("div", "item-body", tool.description));
    tools.append(item);
  }
  const skills = section(parent, `Skill · ${state.skills.length}`);
  if (!state.skills.length) skills.append(element("div", "inspect-empty", "暂无 skill"));
  for (const skill of state.skills) {
    const item = element("div", "tool-item");
    item.append(element("div", "item-top", skill.name), element("div", "item-body", skill.description));
    skills.append(item);
  }
}

function renderInspector() {
  const content = $("#inspector-content");
  content.replaceChildren();
  if (state.tab === "trace") renderTrace(content);
  if (state.tab === "memory") renderMemory(content);
  if (state.tab === "tools") renderTools(content);
  for (const button of document.querySelectorAll(".tab")) {
    const active = button.dataset.tab === state.tab;
    button.classList.toggle("active", active);
    button.setAttribute("aria-selected", String(active));
  }
  $("#inspector-subtitle").textContent = state.tab === "trace" ? "当前会话" : (state.tab === "memory" ? "状态与召回" : "可用能力");
}

function renderModels() {
  const select = $("#model-select");
  select.replaceChildren();
  for (const model of state.models) {
    const option = element("option", "", model.label);
    option.value = model.id;
    option.selected = model.id === state.modelProfile;
    select.append(option);
  }
  if (!state.models.length) {
    const option = element("option", "", state.model || "默认模型");
    option.value = "default";
    select.append(option);
  }
  select.disabled = state.busy || state.models.length < 2;
}

function render() {
  renderSessions();
  renderWorkspaces();
  renderFiles();
  renderMessages();
  renderInspector();
  renderModels();
  $("#model-pill").textContent = state.model || "Agent";
  $("#sidebar-model").textContent = state.model || "Agent";
  $("#embedding-label").textContent = state.embedding ? `Embedding · ${state.embedding}` : "";
  $("#composer-state").textContent = state.busy ? "处理中" : (state.sessionArchived ? "已归档" : "就绪");
  $("#send-button").disabled = state.busy || state.sessionArchived || !$("#message-input").value.trim();
  $("#message-input").disabled = state.busy || state.sessionArchived;
  $("#message-input").placeholder = state.sessionArchived ? "恢复会话后继续对话" : "输入消息…";
  $("#new-session").disabled = state.busy;
  for (const button of document.querySelectorAll(".agent-option")) {
    const active = button.dataset.backend === state.backend;
    button.classList.toggle("active", active);
    button.setAttribute("aria-pressed", String(active));
    button.disabled = state.busy;
  }
}

async function loadState(sessionId, resetTrace = false) {
  const params = new URLSearchParams({
    workspace: state.workspaceId,
    model_profile: state.modelProfile,
  });
  if (sessionId) params.set("session", sessionId);
  const suffix = `?${params}`;
  const data = await api(`/api/state${suffix}`);
  Object.assign(state, data);
  state.sessionId = data.session_id;
  state.workspaceId = data.workspace.workspace_id;
  sessionStorage.setItem("agent-workspace", state.workspaceId);
  if (resetTrace) state.trace = historyTrace(data.messages);
  render();
}

async function newSession() {
  if (state.busy) return;
  try {
    const data = await api("/api/session", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ workspace_id: state.workspaceId }) });
    state.showArchived = false;
    closePanels();
    await loadState(data.session_id, true);
    $("#message-input").focus();
  } catch (error) { toast(error.message); }
}

async function archiveSession(session) {
  if (state.busy) return;
  const archived = !session.archived_at;
  try {
    await api("/api/session/archive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: session.session_id, archived, workspace_id: state.workspaceId }),
    });
    if (!archived) {
      state.showArchived = false;
      await loadState(session.session_id, true);
    } else if (session.session_id === state.sessionId) {
      state.showArchived = false;
      const next = state.sessions.find((item) => item.session_id !== session.session_id && !item.archived_at);
      if (next) await loadState(next.session_id, true);
      else await newSession();
    } else {
      await loadState(state.sessionId);
    }
    toast(archived ? "会话已归档" : "会话已恢复");
  } catch (error) { toast(error.message); }
}

async function openFile(path) {
  try {
    const file = await api(`/api/file?workspace=${encodeURIComponent(state.workspaceId)}&path=${encodeURIComponent(path)}`);
    $("#file-title").textContent = file.path;
    $("#file-content").textContent = file.content || "（空文件）";
    closePanels();
    $("#file-dialog").showModal();
  } catch (error) { toast(error.message); }
}

function processEvent(event) {
  state.trace.push(event);
  if (state.trace.length > 100) state.trace.shift();
  if (event.type === "answer") {
    state.answerReceived = true;
    state.messages.push({ role: "assistant", content: event.text, tool_calls: [] });
    renderMessages();
  }
  if (event.type === "error") toast(event.message || "调用失败");
  if (state.tab === "trace") renderInspector();
  $("#composer-state").textContent = state.busy ? "处理中" : "就绪";
}

async function readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) processEvent(JSON.parse(line));
    if (done) break;
  }
  if (buffer.trim()) processEvent(JSON.parse(buffer));
}

async function send(event) {
  event.preventDefault();
  const input = $("#message-input");
  const text = input.value.trim();
  if (!text || !state.sessionId || state.busy || state.sessionArchived) return;
  state.busy = true;
  state.answerReceived = false;
  state.trace = [];
  state.messages.push({ role: "user", content: text, tool_calls: [] });
  input.value = "";
  render();
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        message: text,
        backend: state.backend,
        workspace_id: state.workspaceId,
        model_profile: state.modelProfile,
      }),
    });
    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error || "发送失败");
    }
    await readEvents(response);
    await loadState(state.sessionId);
  } catch (error) {
    toast(error.message);
    if (!state.trace.some((item) => item.type === "answer")) {
      state.messages.push({ role: "assistant", content: `（调用失败：${error.message}）`, tool_calls: [] });
    }
  } finally {
    state.busy = false;
    state.answerReceived = false;
    render();
    input.focus();
  }
}

function closePanels() {
  $("#sidebar").classList.remove("open");
  $("#inspector").classList.remove("open");
  $("#backdrop").hidden = true;
}

function togglePanel(name) {
  if (name === "inspector" && window.innerWidth > 1100) {
    document.body.classList.toggle("inspector-hidden");
    return;
  }
  const target = name === "sidebar" ? $("#sidebar") : $("#inspector");
  const opening = !target.classList.contains("open");
  closePanels();
  if (opening) {
    target.classList.add("open");
    $("#backdrop").hidden = false;
  }
}

$("#new-session").addEventListener("click", newSession);
$("#workspace-select").addEventListener("change", async (event) => {
  if (state.busy) return;
  state.workspaceId = event.target.value;
  state.showArchived = false;
  sessionStorage.setItem("agent-workspace", state.workspaceId);
  try { await loadState(null, true); } catch (error) { toast(error.message); }
});
$("#new-workspace").addEventListener("click", () => {
  $("#workspace-name").value = "";
  $("#workspace-dialog").showModal();
  $("#workspace-name").focus();
});
$("#workspace-form").addEventListener("submit", async (event) => {
  if (event.submitter?.value !== "create") return;
  event.preventDefault();
  try {
    const data = await api("/api/workspace", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: $("#workspace-name").value.trim() }) });
    $("#workspace-dialog").close();
    state.workspaces.push(data.workspace);
    state.workspaceId = data.workspace.workspace_id;
    state.showArchived = false;
    await loadState(null, true);
  } catch (error) { toast(error.message); }
});
$("#toggle-archives").addEventListener("click", () => {
  state.showArchived = !state.showArchived;
  renderSessions();
});
for (const button of document.querySelectorAll(".agent-option")) {
  button.addEventListener("click", () => {
    if (state.busy || state.backend === button.dataset.backend) return;
    state.backend = button.dataset.backend;
    sessionStorage.setItem("agent-backend", state.backend);
    render();
  });
}
$("#model-select").addEventListener("change", async (event) => {
  if (state.busy) return;
  const previous = state.modelProfile;
  state.modelProfile = event.target.value;
  sessionStorage.setItem("agent-model-profile", state.modelProfile);
  render();
  try { await loadState(state.sessionId); }
  catch (error) {
    state.modelProfile = previous;
    sessionStorage.setItem("agent-model-profile", previous);
    render();
    toast(error.message);
  }
});
$("#composer-form").addEventListener("submit", send);
$("#message-input").addEventListener("input", () => {
  $("#send-button").disabled = state.busy || state.sessionArchived || !$("#message-input").value.trim();
});
$("#message-input").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) send(event);
});
$("#toggle-sidebar").addEventListener("click", () => togglePanel("sidebar"));
$("#toggle-inspector").addEventListener("click", () => togglePanel("inspector"));
$("#close-inspector").addEventListener("click", closePanels);
$("#backdrop").addEventListener("click", closePanels);
$("#close-file").addEventListener("click", () => $("#file-dialog").close());
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => { state.tab = tab.dataset.tab; renderInspector(); });
}
api("/api/workspaces").then((data) => {
  state.workspaces = data.workspaces || [];
  if (!state.workspaces.some((item) => item.workspace_id === state.workspaceId)) state.workspaceId = state.workspaces[0]?.workspace_id || "default";
  return loadState(null, true);
}).catch((error) => toast(error.message));
