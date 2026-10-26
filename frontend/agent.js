"use strict";

const byId = (id) => document.getElementById(id);
const loginPanel = byId("loginPanel");
const inboxPanel = byId("inboxPanel");
const caseList = byId("caseList");
const messageList = byId("messageList");
let socket = null;
let agentId = null;
let selectedCaseId = null;
let cases = new Map();
let messages = new Map();

function connectAgent(event) {
  event.preventDefault();
  byId("loginError").textContent = "";
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}/ws/agent`);
  socket.onopen = () => socket.send(JSON.stringify({
    type: "auth",
    token: byId("agentToken").value,
    name: byId("agentName").value,
  }));
  socket.onmessage = (eventMessage) => {
    const message = JSON.parse(eventMessage.data);
    if (message.type === "error") {
      byId("loginError").textContent = message.message || "Authentication failed.";
      return;
    }
    if (message.type === "agent_ready") {
      agentId = message.agent_id;
      loginPanel.hidden = true;
      inboxPanel.hidden = false;
      byId("connectionState").textContent = `Online · ${message.name}`;
      byId("connectionState").classList.add("online");
      byId("agentToken").value = "";
      return;
    }
    if (message.type === "queue") {
      cases = new Map(message.cases.map((item) => [item.case_id, item]));
      for (const item of message.cases) messages.set(item.case_id, item.messages || []);
      if (!cases.has(selectedCaseId)) selectedCaseId = message.cases[0]?.case_id || null;
      render();
    } else if (message.type === "queue_updated") {
      cases = new Map((message.cases || []).map((item) => [item.case_id, item]));
      if (!cases.has(selectedCaseId)) selectedCaseId = message.cases?.[0]?.case_id || null;
      render();
    } else if (message.type === "case_updated") {
      if (message.case.status === "resolved") {
        cases.delete(message.case.case_id);
        messages.delete(message.case.case_id);
        if (selectedCaseId === message.case.case_id) selectedCaseId = null;
      } else {
        cases.set(message.case.case_id, message.case);
      }
      render();
    } else if (message.type === "action_ok") {
      byId("inboxNotice").textContent = message.message || "Action completed.";
    } else if (message.type === "call_started") {
      byId("callStatus").textContent = `${message.message || "Call queued."} · ${message.call_sid}`;
      byId("inboxNotice").textContent = `Phone call ${message.status || "queued"}.`;
    } else if (message.type === "case_history") {
      messages.set(message.case_id, message.messages || []);
      renderMessages();
    } else if (message.type === "case_message") {
      const current = messages.get(message.case_id) || [];
      current.push(message.message);
      messages.set(message.case_id, current);
      renderMessages();
    } else if (message.type === "action_error") {
      byId("inboxNotice").textContent = message.message || "Action failed.";
      if (!byId("callPanel").hidden) byId("callStatus").textContent = message.message || "Call failed.";
    }
  };
  socket.onerror = () => { byId("loginError").textContent = "Could not connect to the advisor service."; };
  socket.onclose = () => {
    byId("connectionState").textContent = "Offline";
    byId("connectionState").classList.remove("online");
    inboxPanel.hidden = true;
    loginPanel.hidden = false;
  };
}

function render() {
  caseList.replaceChildren();
  const ordered = [...cases.values()].sort((a, b) => a.created_at.localeCompare(b.created_at));
  if (!ordered.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No open cases. New handoffs will appear here.";
    caseList.append(empty);
  }
  for (const item of ordered) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `case-card${item.case_id === selectedCaseId ? " selected" : ""}`;
    button.addEventListener("click", () => { selectedCaseId = item.case_id; render(); });
    const title = document.createElement("strong");
    title.textContent = `${item.case_id} · ${item.category.replaceAll("_", " ")}`;
    const summary = document.createElement("span");
    summary.textContent = item.summary;
    const meta = document.createElement("small");
    meta.textContent = `${item.status.replaceAll("_", " ")} · CRM ${item.crm_status.replaceAll("_", " ")}`;
    button.append(title, summary, meta);
    caseList.append(button);
  }
  renderDetails();
}

function renderDetails() {
  const item = cases.get(selectedCaseId);
  byId("caseEmpty").hidden = Boolean(item);
  byId("caseDetail").hidden = !item;
  if (!item) return;
  byId("caseTitle").textContent = `${item.case_id} · ${item.category.replaceAll("_", " ")}`;
  byId("caseStatus").textContent = item.status.replaceAll("_", " ");
  byId("caseSummary").textContent = item.summary;
  byId("caseMeta").textContent = `Created ${new Date(item.created_at).toLocaleString()} · CRM ${item.crm_status}${item.crm_reference ? ` · ${item.crm_reference}` : ""}`;
  const owned = item.assigned_to === agentId;
  byId("claimButton").hidden = item.status !== "waiting";
  byId("claimButton").disabled = item.status !== "waiting";
  byId("replyForm").hidden = !owned;
  byId("resolveButton").hidden = !owned;
  byId("retryCrmButton").hidden = !owned || item.crm_status !== "failed";
  byId("callPanel").hidden = !owned;
  byId("callButton").disabled = !owned;
  renderMessages();
}

function renderMessages() {
  messageList.replaceChildren();
  const messagesForCase = messages.get(selectedCaseId) || [];
  for (const item of messagesForCase) {
    const bubble = document.createElement("article");
    bubble.className = `message ${item.sender === "advisor" ? "advisor" : "customer"}`;
    const text = document.createElement("span");
    text.textContent = item.text;
    const sender = document.createElement("small");
    sender.textContent = item.sender;
    bubble.append(text, sender);
    messageList.append(bubble);
  }
  messageList.scrollTop = messageList.scrollHeight;
}

function sendCommand(command) {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(command));
}

byId("loginForm").addEventListener("submit", connectAgent);
byId("refreshButton").addEventListener("click", () => sendCommand({ type: "refresh" }));
byId("claimButton").addEventListener("click", () => sendCommand({ type: "claim", case_id: selectedCaseId }));
byId("resolveButton").addEventListener("click", () => sendCommand({ type: "resolve", case_id: selectedCaseId }));
byId("retryCrmButton").addEventListener("click", () => sendCommand({ type: "retry_crm", case_id: selectedCaseId }));
byId("callButton").addEventListener("click", () => {
  const phone = byId("customerPhone").value.trim();
  if (!selectedCaseId) return;
  if (!/^\+[1-9]\d{7,14}$/.test(phone)) {
    byId("callStatus").textContent = "Use E.164 format, for example +33753414072.";
    return;
  }
  byId("callButton").disabled = true;
  byId("callStatus").textContent = "Requesting Twilio call…";
  sendCommand({ type: "call_customer", case_id: selectedCaseId, to_number: phone });
  window.setTimeout(() => {
    if (!byId("callPanel").hidden) byId("callButton").disabled = false;
  }, 5000);
});
byId("replyForm").addEventListener("submit", (event) => {
  event.preventDefault();
  const text = byId("replyText").value.trim();
  if (!text) return;
  sendCommand({ type: "reply", case_id: selectedCaseId, text });
  byId("replyText").value = "";
});
