"use strict";

const el = (id) => document.getElementById(id);
const startButton = el("startButton");
const stopButton = el("stopButton");
const clearButton = el("clearButton");
const conversation = el("conversation");
const sources = el("sources");
const searchInput = el("transcriptSearch");
const navigationLinks = [...document.querySelectorAll(".nav-link[href^='#']")];
const playbackSources = new Set();

let socket = null;
let peerConnection = null;
let voiceTransport = "websocket";
let publicConfig = { voice_transport: "websocket", webrtc_available: false };
let mediaStream = null;
let audioContext = null;
let pcmPlayer = null;
let pcmOutputDrained = true;
let microphoneSource = null;
let processor = null;
let isActive = false;
let pendingSamples = new Float32Array(0);
let assistantMessageBody = null;
let lastAssistantText = "";
let audioPending = false;
let audioQueue = [];
let audioDecodeActive = false;
let nextAudioStartTime = 0;
let currentAudioMetadata = null;
let playbackGeneration = 0;
let sessionStartedAt = null;
let sessionClockInterval = null;
let currentRetrievedDocuments = [];
let pendingDemoToolLabel = "";

function voiceEndpoint(path) {
  const tenantId = new URLSearchParams(location.search).get("tenant_id");
  return tenantId ? `${path}?tenant_id=${encodeURIComponent(tenantId)}` : path;
}

const configPromise = fetch("/api/config")
  .then((response) => response.ok ? response.json() : Promise.reject(new Error("config unavailable")))
  .then((config) => { publicConfig = config; })
  .catch(() => { publicConfig = { voice_transport: "websocket", webrtc_available: false }; });

function activateNavigationLink(link) {
  navigationLinks.forEach((item) => {
    const active = item === link;
    item.classList.toggle("is-active", active);
    if (active) item.setAttribute("aria-current", "location");
    else item.removeAttribute("aria-current");
  });
}

navigationLinks.forEach((link) => {
  link.addEventListener("click", (event) => {
    const target = document.querySelector(link.hash);
    if (!target) return;
    event.preventDefault();
    activateNavigationLink(link);
    history.replaceState(null, "", link.hash);
    target.scrollIntoView({ behavior: "smooth", block: "start" });
  });
});
if (navigationLinks.length) {
  activateNavigationLink(navigationLinks.find((link) => link.hash === "#voicePanel") || navigationLinks[0]);
}

for (let index = 0; index < 72; index += 1) {
  const ray = document.createElement("i");
  ray.style.setProperty("--ray", index);
  el("orbSpectrum").append(ray);
}

function setStatus(label, detail, mode = "") {
  el("statusLabel").textContent = label;
  el("statusDetail").textContent = detail;
  el("statusDot").className = `status-dot ${mode}`.trim();
  el("orbSpectrum").classList.toggle("is-active", mode === "active" || mode === "busy");
  el("voicePanel").classList.toggle("is-active", mode === "active");
  el("voicePanel").classList.toggle("is-busy", mode === "busy");
  const headlines = {
    Disconnected: "Ready when you are",
    Connecting: "Connecting your microphone",
    Listening: "I'm listening...",
    "Processing ASR": "Transcribing your voice",
    Retrieving: "Searching knowledge sources",
    Generating: "Preparing your answer",
    Speaking: "Speaking...",
    Interrupted: "I'm listening...",
    Error: "Something went wrong",
  };
  el("voiceHeadline").textContent = headlines[label] || label;
  el("voiceStateTag").textContent = mode === "busy" ? "processing" : mode === "active" ? "listening" : "ready";
  el("activeControlLabel").textContent = label;
}

function setButtons(active) {
  startButton.disabled = active;
  stopButton.disabled = !active;
}

function hideEmptyState() {
  const empty = el("emptyConversation");
  if (empty) empty.remove();
}

function addMessage(role, text, meta = "") {
  hideEmptyState();
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("span");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "YOU" : role === "advisor" ? "AGENT" : "N";
  const content = document.createElement("div");
  content.className = "message-content";
  const label = document.createElement("p");
  label.className = "message-label";
  label.textContent = role === "user" ? "YOU" : role === "advisor" ? "ADVISOR" : "NOVA";
  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;
  content.append(label, body);
  if (meta) {
    const hint = document.createElement("p");
    hint.className = "message-meta";
    hint.textContent = meta;
    content.append(hint);
  }
  article.append(avatar, content);
  conversation.append(article);
  conversation.scrollTop = conversation.scrollHeight;
  updateTurnCount();
  filterTranscript();
  return body;
}

function updateTurnCount() {
  const turns = conversation.querySelectorAll(".message.user").length;
  el("turnCount").textContent = `${turns} ${turns === 1 ? "turn" : "turns"}`;
}

function filterTranscript() {
  const query = searchInput.value.trim().toLocaleLowerCase();
  conversation.querySelectorAll(".message").forEach((message) => {
    message.hidden = query !== "" && !message.textContent.toLocaleLowerCase().includes(query);
  });
}

function showEmptyConversation() {
  conversation.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-state";
  const orb = document.createElement("span");
  orb.className = "empty-orb";
  orb.setAttribute("aria-hidden", "true");
  const core = document.createElement("span");
  orb.append(core);
  const title = document.createElement("strong");
  title.textContent = "Your conversation will appear here";
  const detail = document.createElement("span");
  detail.textContent = "Speak naturally. The agent listens for the end of your turn.";
  empty.append(orb, title, detail);
  conversation.append(empty);
  updateTurnCount();
}

function resetMetrics() {
  ["metricVad", "metricAsr", "metricRag", "metricLlm", "metricTts", "metricTotal", "metricTtft", "metricFirstAudio"]
    .forEach((id) => { el(id).textContent = "—"; });
  el("latencyTotalBadge").textContent = "— ms";
  ["barVad", "barAsr", "barRag", "barLlm", "barTts"].forEach((id) => { el(id).style.width = "0%"; });
}

function updateSessionClock() {
  if (!sessionStartedAt) return;
  const elapsed = Math.floor((Date.now() - sessionStartedAt) / 1000);
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");
  el("sessionTimer").textContent = `${minutes}:${seconds}`;
}

function startSessionClock() {
  stopSessionClock();
  sessionStartedAt = Date.now();
  updateSessionClock();
  sessionClockInterval = window.setInterval(updateSessionClock, 1000);
}

function stopSessionClock() {
  if (sessionClockInterval) window.clearInterval(sessionClockInterval);
  sessionClockInterval = null;
  sessionStartedAt = null;
}

function stopPlayback() {
  playbackGeneration += 1;
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  for (const source of playbackSources) {
    try { source.stop(); } catch { /* The source may already have ended. */ }
  }
  playbackSources.clear();
  pcmPlayer?.port.postMessage({ type: "stop" });
  pcmOutputDrained = true;
  audioQueue = [];
  nextAudioStartTime = 0;
  audioPending = false;
  currentAudioMetadata = null;
}

function speakWithBrowser(text) {
  if (!("speechSynthesis" in window) || !window.SpeechSynthesisUtterance || !text.trim()) return false;
  const utterance = new window.SpeechSynthesisUtterance(text);
  utterance.lang = "fr-FR";
  const frenchVoice = window.speechSynthesis.getVoices().find((voice) => voice.lang.toLowerCase().startsWith("fr"));
  if (frenchVoice) utterance.voice = frenchVoice;
  utterance.onstart = () => setStatus("Speaking", "Browser voice fallback is speaking.", "busy");
  utterance.onend = () => {
    if (isActive) setStatus("Listening", "You can continue the conversation.", "active");
  };
  utterance.onerror = () => setStatus("Text response", "The browser could not read the response aloud.", "active");
  try {
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
    return true;
  } catch (error) {
    console.warn("Browser speech synthesis failed", error);
    return false;
  }
}

function setAssistantMeta(text) {
  if (!assistantMessageBody) return;
  let meta = assistantMessageBody.parentElement.querySelector(".message-meta");
  if (!meta) {
    meta = document.createElement("p");
    meta.className = "message-meta";
    assistantMessageBody.parentElement.append(meta);
  }
  meta.textContent = text;
}

function appendAssistantSourceTags(documents) {
  if (!assistantMessageBody) return;
  assistantMessageBody.parentElement.querySelector(".message-sources")?.remove();
  if (!documents.length) return;
  const tags = document.createElement("div");
  tags.className = "message-sources";
  documents.forEach((retrievedDocument, index) => {
    const tag = document.createElement("span");
    const sourceName = (retrievedDocument.source || "NovaTel KB").replaceAll("\\", "/").split("/").pop();
    tag.textContent = `Source ${index + 1} · ${sourceName.replace(/\.md$/i, "")}`;
    tags.append(tag);
  });
  assistantMessageBody.parentElement.append(tags);
}

function appendSourceCards(documents, mode = "faiss", intent = "") {
  sources.replaceChildren();
  el("sourceCount").textContent = `${documents.length} ${documents.length === 1 ? "chunk" : "chunks"}`;
  el("intentBadge").textContent = `Detected intent · ${intent ? intent.replaceAll("_", " ") : "waiting"}`;
  if (!documents.length) {
    const note = document.createElement("p");
    note.className = "empty-note";
    note.textContent = mode === "conversation"
      ? "Conversational reply · no document search needed."
      : mode === "tool_confirmation" || mode === "demo_tool"
        ? "Demo customer-service action · no knowledge search needed."
      : mode === "out_of_scope"
        ? "Outside NovaTel support scope · general-knowledge answers are disabled."
        : "No relevant NovaTel passages found · the assistant will not guess.";
    sources.append(note);
    return;
  }
  for (const retrievedDocument of documents) {
    const card = document.createElement("article");
    card.className = "source-card";
    const head = document.createElement("div");
    head.className = "source-card-head";
    const rank = document.createElement("span");
    rank.className = "source-rank";
    rank.textContent = String(retrievedDocument.rank);
    const name = document.createElement("span");
    name.className = "source-name";
    const basename = (retrievedDocument.source || "knowledge base").split(/[\\/]/).pop().replace(/\.md$/i, "").replaceAll("_", " ");
    const content = retrievedDocument.content || retrievedDocument.text || "";
    const heading = content.split("\n").find((line) => line.startsWith("#"))?.replace(/^#+\s*/, "").trim();
    name.textContent = heading ? `${basename} › ${heading}` : basename;
    const score = document.createElement("span");
    score.className = "source-score";
    const scoreValue = Number(retrievedDocument.score) || 0;
      const scoreLabel = mode.includes("reranked")
        ? "rerank"
        : mode.startsWith("hybrid_rrf")
          ? "RRF"
          : mode.includes("bm25")
            ? "BM25"
            : mode === "conversation_context" ? "context" : "cos";
    score.textContent = `${scoreLabel} ${scoreValue.toFixed(2)}`;
    head.append(rank, name, score);
    const meter = document.createElement("div");
    meter.className = "source-meter";
    const meterFill = document.createElement("i");
    meterFill.style.width = `${Math.max(0, Math.min(100, scoreValue * 100))}%`;
    meter.append(meterFill);
    const text = document.createElement("p");
    text.className = "source-text";
    text.textContent = content;
    card.append(head, meter, text);
    if (retrievedDocument.source_url) {
      try {
        const url = new URL(retrievedDocument.source_url);
        if (url.protocol === "https:" || url.protocol === "http:") {
          const provenance = document.createElement("a");
          provenance.className = "source-link";
          provenance.href = url.href;
          provenance.target = "_blank";
          provenance.rel = "noopener noreferrer";
          provenance.textContent = `Source web · ${url.hostname}`;
          provenance.title = retrievedDocument.retrieved_at
            ? `Collecté le ${retrievedDocument.retrieved_at}`
            : url.href;
          card.append(provenance);
        }
      } catch {
        console.warn("Ignoring an invalid RAG source URL");
      }
    }
    sources.append(card);
  }
  const modeNote = document.createElement("p");
  modeNote.className = "retrieval-mode";
  modeNote.textContent = mode === "bm25_fallback"
      ? "Dense retrieval unavailable · local BM25 fallback"
      : mode === "hybrid_bm25_fallback"
        ? "Dense retrieval unavailable · hybrid request fell back to BM25"
      : mode.startsWith("bm25")
      ? "Local keyword retrieval · BM25"
      : mode.startsWith("hybrid_rrf")
        ? mode.includes("reranked")
          ? "Hybrid BM25 + FAISS ranking · CrossEncoder reranked"
          : "Hybrid keyword + semantic retrieval · Reciprocal Rank Fusion"
      : mode === "conversation_context"
        ? "Previous relevant passages reused for this follow-up."
        : mode.includes("reranked")
          ? "Dense retrieval · CrossEncoder reranked"
        : "Dense semantic retrieval · FAISS";
  sources.append(modeNote);
}

function formatMs(value) {
  return value == null ? "—" : `${Math.round(value)} ms`;
}

function renderMetrics(metrics) {
  el("metricVad").textContent = formatMs(metrics.vad_ms);
  el("metricAsr").textContent = formatMs(metrics.asr_ms);
  el("metricRag").textContent = formatMs(metrics.retrieval_ms);
  el("metricLlm").textContent = formatMs(metrics.llm_ms);
  el("metricTts").textContent = formatMs(metrics.tts_ms);
  el("metricTotal").textContent = formatMs(metrics.total_ms);
  el("metricTtft").textContent = formatMs(metrics.time_to_first_token_ms);
  el("metricFirstAudio").textContent = formatMs(metrics.time_to_first_audio_ms);
  el("latencyTotalBadge").textContent = formatMs(metrics.total_ms);
  const values = [metrics.vad_ms, metrics.asr_ms, metrics.retrieval_ms, metrics.llm_ms, metrics.tts_ms]
    .map((value) => (Number.isFinite(Number(value)) ? Math.max(0, Number(value)) : 0));
  const sum = values.reduce((total, value) => total + value, 0);
  ["barVad", "barAsr", "barRag", "barLlm", "barTts"].forEach((id, index) => {
    el(id).style.width = sum ? `${(values[index] / sum) * 100}%` : "0%";
  });
}

function resample(input, inputRate, outputRate) {
  if (inputRate === outputRate) return input.slice();
  const outputLength = Math.floor(input.length * outputRate / inputRate);
  const output = new Float32Array(outputLength);
  const ratio = inputRate / outputRate;
  for (let i = 0; i < outputLength; i += 1) {
    const position = i * ratio;
    const left = Math.floor(position);
    const fraction = position - left;
    const right = Math.min(left + 1, input.length - 1);
    output[i] = input[left] * (1 - fraction) + input[right] * fraction;
  }
  return output;
}

function sendPcmChunk(samples) {
  if (!socket || socket.readyState !== WebSocket.OPEN || !isActive) return;
  const bytes = new ArrayBuffer(samples.length * 2);
  const view = new DataView(bytes);
  for (let i = 0; i < samples.length; i += 1) {
    const value = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, value < 0 ? value * 32768 : value * 32767, true);
  }
  socket.send(bytes);
}

function setupMicrophone() {
  microphoneSource = audioContext.createMediaStreamSource(mediaStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  processor.onaudioprocess = (event) => {
    const output = event.outputBuffer.getChannelData(0);
    output.fill(0);
    if (!isActive || !socket || socket.readyState !== WebSocket.OPEN) return;
    const input = event.inputBuffer.getChannelData(0);
    const samples = resample(input, audioContext.sampleRate, 16000);
    const merged = new Float32Array(pendingSamples.length + samples.length);
    merged.set(pendingSamples);
    merged.set(samples, pendingSamples.length);
    let offset = 0;
    const chunkSize = 1024;
    while (merged.length - offset >= chunkSize) {
      sendPcmChunk(merged.subarray(offset, offset + chunkSize));
      offset += chunkSize;
    }
    pendingSamples = merged.slice(offset);
  };
  microphoneSource.connect(processor);
  processor.connect(audioContext.destination);
}

async function playWav(arrayBuffer) {
  if (!audioContext || !isActive) return;
  audioQueue.push({ data: arrayBuffer, generation: playbackGeneration });
  if (audioDecodeActive) return;
  audioDecodeActive = true;
  try {
    while (audioQueue.length) {
      const item = audioQueue.shift();
      if (!audioContext || !isActive || item.generation !== playbackGeneration) continue;
      const audioBuffer = await audioContext.decodeAudioData(item.data.slice(0));
      if (!audioContext || !isActive || item.generation !== playbackGeneration) continue;
      const source = audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContext.destination);
      playbackSources.add(source);
      source.onended = () => {
        playbackSources.delete(source);
        finishPlaybackIfIdle();
      };
      const startAt = Math.max(audioContext.currentTime, nextAudioStartTime);
      nextAudioStartTime = startAt + audioBuffer.duration;
      source.start(startAt);
      setStatus("Speaking", "The generated voice response is playing.", "busy");
    }
  } catch (error) {
    console.error("Audio decoding failed", error);
    setStatus("Text response", "This browser could not decode the generated audio.", "active");
  } finally {
    audioDecodeActive = false;
    finishPlaybackIfIdle();
  }
}

function finishPlaybackIfIdle() {
  if (!audioPending && pcmOutputDrained && !audioDecodeActive && !audioQueue.length && playbackSources.size === 0 && isActive) {
    setStatus("Listening", "You can continue the conversation.", "active");
  }
}

function handleEvent(message) {
  switch (message.type) {
    case "session":
      el("sessionBadge").textContent = `session · ${(message.session_id || "").slice(0, 8)}`;
      break;
    case "state": {
      const states = {
        idle: ["Disconnected", "Start a voice session when you are ready."],
        listening: ["Listening", "Speak naturally; voice activity detection is on.", "active"],
        processing_asr: ["Processing ASR", "Preparing Whisper for transcription…", "busy"],
        retrieving: ["Retrieving", "Searching NovaTel knowledge sources.", "busy"],
        generating: ["Generating", "The language model is preparing a grounded answer.", "busy"],
        speaking: ["Speaking", "Preparing the voice response.", "busy"],
      };
      setStatus(...(states[message.state] || ["Connected", "Voice session is active.", "active"]));
      break;
    }
    case "speech_started":
      stopPlayback();
      el("vadSignal").textContent = "Speech detected";
      setStatus("Listening", "Speech detected.", "active");
      break;
    case "speech_ended":
      el("vadSignal").textContent = "Speech segment ended";
      break;
    case "speech_discarded":
      el("vadSignal").textContent = "Speak a little longer";
      if (!message.busy) {
        setStatus("Listening", "That sounded too short to transcribe. Please repeat and pause briefly.", "active");
      }
      break;
    case "asr_progress": {
      const elapsed = Number(message.elapsed_seconds) || 0;
      const elapsedLabel = elapsed ? ` (${elapsed}s)` : "";
      const detail = message.stage === "loading_model"
        ? `Loading Whisper for the first time; its files are cached for later.${elapsedLabel}`
        : `Whisper is decoding this speech segment.${elapsedLabel}`;
      setStatus("Processing ASR", detail, "busy");
      break;
    }
    case "assistant_interrupted":
      setStatus("Interrupted", "Assistant playback stopped; listening to you.", "active");
      break;
    case "transcription":
      if (message.text) addMessage("user", message.text, message.language || "");
      el("languageLabel").textContent = message.language ? message.language.toUpperCase() : "—";
      currentRetrievedDocuments = [];
      assistantMessageBody = null;
      lastAssistantText = "";
      pendingDemoToolLabel = "";
      resetMetrics();
      break;
    case "retrieval":
      currentRetrievedDocuments = message.documents || [];
      appendSourceCards(currentRetrievedDocuments, message.mode || "faiss", message.intent || "");
      break;
    case "llm_token":
      lastAssistantText += message.text || "";
      if (!assistantMessageBody) assistantMessageBody = addMessage("assistant", "");
      assistantMessageBody.textContent += message.text || "";
      conversation.scrollTop = conversation.scrollHeight;
      break;
    case "llm_response":
      lastAssistantText = message.text || lastAssistantText;
      if (!assistantMessageBody) {
        assistantMessageBody = addMessage("assistant", message.text || "", pendingDemoToolLabel);
      }
      else assistantMessageBody.textContent = message.text || assistantMessageBody.textContent;
      appendAssistantSourceTags(currentRetrievedDocuments);
      pendingDemoToolLabel = "";
      break;
    case "demo_tool":
      pendingDemoToolLabel = message.ticket_id
        ? `Demo tool · ${message.ticket_id}`
        : `Demo tool · ${message.status || "prepared"}`;
      break;
    case "support_case":
      pendingDemoToolLabel = `Support case · ${message.case_id || "created"} · CRM ${message.crm_status || "not configured"}`;
      el("sessionBadge").textContent = `case · ${message.case_id || "open"}`;
      setStatus("Advisor queue", message.crm_status === "delivered"
        ? "Your request was sent to the configured support system."
        : "Your case is in the local advisor inbox; external CRM delivery is not active.", "active");
      break;
    case "handoff_prompt":
      pendingDemoToolLabel = "Support handoff · confirmation required";
      break;
    case "handoff_status":
      if (message.status === "in_progress") {
        setStatus("Advisor connected", `${message.agent || "An advisor"} joined this conversation.`, "active");
      } else if (message.status === "resolved") {
        setStatus("Case resolved", "The advisor closed this support request.", "active");
      } else {
        setStatus("Waiting for advisor", "Your request remains in the support queue.", "busy");
      }
      break;
    case "handoff_message_sent":
      setStatus("Waiting for advisor", "Your message was sent to the assigned advisor.", "active");
      break;
    case "advisor_message":
      lastAssistantText = message.text || "";
      assistantMessageBody = addMessage("advisor", message.text || "", `Advisor${message.agent ? ` · ${message.agent}` : ""}`);
      break;
    case "tts_audio_start":
      audioPending = true;
      currentAudioMetadata = message;
      pcmOutputDrained = false;
      pcmPlayer?.port.postMessage({ type: "start", sampleRate: Number(message.sample_rate) || 24_000 });
      if (message.backend) {
        setAssistantMeta(`Server voice · ${message.backend}${message.chunked ? " · sentence streaming" : ""}`);
        setStatus("Speaking", `Audio generated by ${message.backend}.`, "busy");
      }
      break;
    case "tts_audio_end":
      audioPending = false;
      pcmPlayer?.port.postMessage({ type: "end" });
      if (!pcmPlayer) pcmOutputDrained = true;
      finishPlaybackIfIdle();
      break;
    case "metrics":
      renderMetrics(message.latencies || {});
      break;
    case "conversation_cleared":
      showEmptyConversation();
      sources.replaceChildren();
      el("sourceCount").textContent = "0 chunks";
      el("intentBadge").textContent = "Detected intent · waiting";
      el("sessionBadge").textContent = isActive ? "session · active" : "session · waiting";
      el("languageLabel").textContent = "—";
      el("vadSignal").textContent = isActive ? "VAD ready" : "VAD idle";
      searchInput.value = "";
      currentRetrievedDocuments = [];
      assistantMessageBody = null;
      lastAssistantText = "";
      resetMetrics();
      break;
    case "error":
      if (message.code === "TTSFailure") {
        stopPlayback();
      }
      if (message.code === "TTSFailure" && speakWithBrowser(lastAssistantText)) {
        setAssistantMeta("Browser speech fallback · server Piper/eSpeak unavailable");
        setStatus("Speaking", "Using the browser's built-in voice fallback.", "active");
        console.warn("Server TTS unavailable; using browser speech synthesis", message.message);
        break;
      }
      addMessage("assistant", message.message || "An error occurred.", `Error · ${message.code || "unknown"}`);
      setStatus("Error", message.message || "An error occurred.", "busy");
      break;
    case "pong":
      break;
    default:
      console.debug("Unrecognized server event", message.type);
  }
}

async function startConversation() {
  if (isActive) return;
  startButton.disabled = true;
  setStatus("Connecting", "Requesting microphone permission…", "busy");
  try {
    await configPromise;
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const preferredTransport = String(publicConfig.voice_transport || "websocket").toLowerCase();
    const canUseWebRTC = publicConfig.webrtc_available && "RTCPeerConnection" in window;
    if (canUseWebRTC && preferredTransport !== "websocket") {
      try {
        await startWebRTCTransport();
        return;
      } catch (error) {
        console.warn("WebRTC unavailable; falling back to WebSocket PCM", error);
        cleanupAudio();
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
      }
    }
    voiceTransport = "websocket";
    audioContext = new AudioContext();
    await audioContext.resume();
    if (audioContext.audioWorklet && "AudioWorkletNode" in window) {
      try {
        await audioContext.audioWorklet.addModule("/static/audio-player-worklet.js");
        pcmPlayer = new AudioWorkletNode(audioContext, "pcm-output", {
          numberOfInputs: 0,
          numberOfOutputs: 1,
          outputChannelCount: [1],
        });
        pcmPlayer.port.onmessage = (event) => {
          if (event.data?.type === "drained") {
            pcmOutputDrained = true;
            finishPlaybackIfIdle();
          }
        };
        pcmPlayer.connect(audioContext.destination);
      } catch (error) {
        console.warn("AudioWorklet unavailable; using scheduled PCM playback", error);
      }
    }
    const protocol = location.protocol === "https:" ? "wss:" : "ws:";
    socket = new WebSocket(`${protocol}//${location.host}${voiceEndpoint("/ws/voice")}`);
    socket.binaryType = "arraybuffer";
    socket.onopen = () => {
      isActive = true;
      startSessionClock();
      pendingSamples = new Float32Array(0);
      setButtons(true);
      setupMicrophone();
      socket.send(JSON.stringify({ type: "start" }));
      setStatus("Listening", "Speak naturally; voice activity detection is on.", "active");
    };
    socket.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        playPcmChunk(event.data);
        return;
      }
      try { handleEvent(JSON.parse(event.data)); }
      catch (error) { console.error("Invalid server event", error); }
    };
    socket.onerror = () => setStatus("Error", "Could not connect to the voice server. Check FastAPI and retry.", "busy");
    socket.onclose = (event) => {
      cleanupAudio();
      stopSessionClock();
      console.warn("Voice WebSocket closed", { code: event.code, reason: event.reason });
      setStatus("Disconnected", "Start a conversation to enable the microphone.");
    };
  } catch (error) {
    console.error("Could not start microphone session", error);
    cleanupAudio();
    setButtons(false);
    setStatus("Error", error?.message || "Microphone unavailable. Allow access and try again.", "busy");
  }
}

async function startWebRTCTransport() {
  voiceTransport = "webrtc";
  const protocol = location.protocol === "https:" ? "wss:" : "ws:";
  socket = new WebSocket(`${protocol}//${location.host}${voiceEndpoint("/ws/webrtc")}`);
  peerConnection = new RTCPeerConnection({ iceServers: [] });
  peerConnection.addTrack(mediaStream.getAudioTracks()[0], mediaStream);
  peerConnection.ontrack = (event) => {
    const remoteAudio = el("remoteAudio");
    remoteAudio.srcObject = event.streams[0];
    remoteAudio.play().catch(() => console.warn("Browser autoplay policy blocked WebRTC audio"));
  };
  let answerReadyResolve;
  let answerReadyReject;
  let signalingComplete = false;
  const result = new Promise((resolve, reject) => {
    answerReadyResolve = resolve;
    answerReadyReject = reject;
    window.setTimeout(() => reject(new Error("WebRTC answer timeout")), 15000);
  });
  socket.onopen = async () => {
      try {
        const offer = await peerConnection.createOffer({ offerToReceiveAudio: true });
        await peerConnection.setLocalDescription(offer);
        for (let index = 0; index < 100 && peerConnection.iceGatheringState !== "complete"; index += 1) {
          await new Promise((done) => window.setTimeout(done, 10));
        }
        socket.send(JSON.stringify({ type: "offer", sdp: peerConnection.localDescription.sdp, sdp_type: peerConnection.localDescription.type }));
      } catch (error) { answerReadyReject(error); }
  };
  socket.onerror = () => answerReadyReject(new Error("WebRTC signaling failed"));
  socket.binaryType = "arraybuffer";
  socket.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) return;
    try {
      const message = JSON.parse(event.data);
      if (message.type === "error" && !signalingComplete && answerReadyReject) {
        answerReadyReject(new Error(message.message || "WebRTC server error"));
        return;
      }
      if (message.type === "answer") {
        await peerConnection.setRemoteDescription({ type: message.sdp_type, sdp: message.sdp });
        signalingComplete = true;
        isActive = true;
        startSessionClock();
        pendingSamples = new Float32Array(0);
        setButtons(true);
        socket.send(JSON.stringify({ type: "start" }));
        setStatus("Listening", "WebRTC microphone connected; speak naturally.", "active");
        answerReadyResolve?.();
        return;
      }
      handleEvent(message);
    } catch (error) { console.error("Invalid WebRTC signaling message", error); }
  };
  await result;
  socket.onclose = (event) => {
    cleanupAudio();
    stopSessionClock();
    console.warn("Voice WebRTC closed", { code: event.code, reason: event.reason });
    setStatus("Disconnected", "Start a conversation to enable the microphone.");
  };
}

function playPcmChunk(arrayBuffer) {
  if (!audioContext || !isActive) return;
  const view = new DataView(arrayBuffer);
  const samples = new Float32Array(Math.floor(arrayBuffer.byteLength / 2));
  for (let index = 0; index < samples.length; index += 1) {
    samples[index] = view.getInt16(index * 2, true) / 32768;
  }
  if (pcmPlayer) {
    pcmPlayer.port.postMessage({ type: "pcm", samples }, [samples.buffer]);
    return;
  }

  // Compatibility path for browsers without AudioWorklet: schedule PCM buffers.
  const sourceRate = Number(currentAudioMetadata?.sample_rate) || 24_000;
  const audioBuffer = audioContext.createBuffer(1, samples.length, sourceRate);
  audioBuffer.copyToChannel(samples, 0);
  const source = audioContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(audioContext.destination);
  playbackSources.add(source);
  source.onended = () => {
    playbackSources.delete(source);
    finishPlaybackIfIdle();
  };
  const startAt = Math.max(audioContext.currentTime, nextAudioStartTime);
  nextAudioStartTime = startAt + audioBuffer.duration;
  source.start(startAt);
}

function cleanupAudio() {
  isActive = false;
  el("vadSignal").textContent = "VAD idle";
  stopPlayback();
  if (peerConnection) {
    try { peerConnection.close(); } catch { /* already closed */ }
  }
  peerConnection = null;
  if (socket && socket.readyState !== WebSocket.CLOSED) {
    try { socket.close(); } catch { /* already closed */ }
  }
  socket = null;
  const remoteAudio = el("remoteAudio");
  if (remoteAudio) {
    remoteAudio.pause();
    remoteAudio.srcObject = null;
  }
  if (processor) {
    processor.onaudioprocess = null;
    try { processor.disconnect(); } catch { /* already disconnected */ }
  }
  if (microphoneSource) {
    try { microphoneSource.disconnect(); } catch { /* already disconnected */ }
  }
  if (mediaStream) mediaStream.getTracks().forEach((track) => track.stop());
  if (audioContext && audioContext.state !== "closed") audioContext.close();
  pcmPlayer = null;
  processor = null;
  microphoneSource = null;
  mediaStream = null;
  audioContext = null;
  voiceTransport = "websocket";
  pendingSamples = new Float32Array(0);
  setButtons(false);
}

function stopConversation() {
  stopSessionClock();
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "stop" }));
  }
  cleanupAudio();
  if (socket && socket.readyState !== WebSocket.CLOSED) {
    window.setTimeout(() => { if (socket && socket.readyState === WebSocket.OPEN) socket.close(); }, 500);
  }
}

function clearConversation() {
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: "clear" }));
  }
  stopPlayback();
  showEmptyConversation();
  sources.replaceChildren();
  el("sourceCount").textContent = "0 chunks";
  el("intentBadge").textContent = "Detected intent · waiting";
  el("languageLabel").textContent = "—";
  el("vadSignal").textContent = isActive ? "VAD ready" : "VAD idle";
  searchInput.value = "";
  currentRetrievedDocuments = [];
  assistantMessageBody = null;
  lastAssistantText = "";
  resetMetrics();
}

startButton.addEventListener("click", startConversation);
stopButton.addEventListener("click", stopConversation);
clearButton.addEventListener("click", clearConversation);
searchInput.addEventListener("input", filterTranscript);
document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    event.preventDefault();
    searchInput.focus();
  }
});

if (!navigator.mediaDevices?.getUserMedia) {
  startButton.disabled = true;
  setStatus("Unsupported browser", "Use a recent browser on localhost or HTTPS.", "busy");
}
