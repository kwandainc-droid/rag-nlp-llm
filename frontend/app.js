const API_BASE = "http://127.0.0.1:8000";
const INTRO_MESSAGES = [
  "Ready to chat with your documents...",
  "Signal acquired. Interpreting your files...",
  "Indexing thought-space for deeper study...",
  "Grounded answers are ready. Let us begin..."
];

const docUpload = document.getElementById("docUpload");
const attachBtn = document.getElementById("attachBtn");
const sidebarToggle = document.getElementById("sidebarToggle");
const fileStatus = document.getElementById("fileStatus");
const docName = document.getElementById("docName");
const msgCount = document.getElementById("msgCount");
const sourceRefs = document.getElementById("sourceRefs");
const docList = document.getElementById("docList");
const queueList = document.getElementById("queueList");
const introScreen = document.getElementById("introScreen");
const introText = document.getElementById("introText");
const homeView = document.getElementById("homeView");

const chatForm = document.getElementById("chatForm");
const chatInput = document.getElementById("chatInput");
const chatBox = document.getElementById("chatBox");

const takeTestBtn = document.getElementById("takeTestBtn");
const submitQuizBtn = document.getElementById("submitQuizBtn");
const backToChatBtn = document.getElementById("backToChatBtn");

const chatView = document.getElementById("chatView");
const quizView = document.getElementById("quizView");
const quizContainer = document.getElementById("quizContainer");
const quizResult = document.getElementById("quizResult");

const state = {
  sessionId: null,
  documentNames: [],
  uploadQueue: [],
  uploading: false,
  readyPromptShown: false,
  messages: [],
  quizQuestions: []
};

function addMessage(role, text) {
  state.messages.push({ role, text });

  const item = document.createElement("div");
  item.className = `msg ${role === "user" ? "user" : "bot"}`;
  item.textContent = text;
  chatBox.appendChild(item);

  chatBox.scrollTop = chatBox.scrollHeight;
  msgCount.textContent = String(state.messages.length);
  renderMainState();
}

function setSidebarOpen(isOpen) {
  document.body.classList.toggle("sidebar-open", isOpen);
  if (sidebarToggle) {
    sidebarToggle.textContent = isOpen ? "✕" : "☰";
    sidebarToggle.setAttribute("aria-label", isOpen ? "Hide sidebar" : "Show sidebar");
  }
}

function clearChat() {
  state.messages = [];
  chatBox.innerHTML = "";
  msgCount.textContent = "0";
  renderMainState();
}

function clearSources() {
  sourceRefs.textContent = "No sources yet.";
}

function clearDocumentList() {
  state.documentNames = [];
  docList.textContent = "No files uploaded yet.";
  renderMainState();
}

function clearQueueList() {
  queueList.textContent = "Nothing in queue.";
}

function renderQueueList(currentFileName = "") {
  const queued = state.uploadQueue.map((entry) => entry.file.name);
  if (!currentFileName && queued.length === 0) {
    clearQueueList();
    return;
  }

  queueList.innerHTML = "";

  if (currentFileName) {
    const current = document.createElement("div");
    current.className = "queue-item working";
    current.innerHTML = `<strong>Indexing</strong>${currentFileName}`;
    queueList.appendChild(current);
  }

  queued.forEach((name) => {
    const item = document.createElement("div");
    item.className = "queue-item waiting";
    item.innerHTML = `<strong>Waiting</strong>${name}`;
    queueList.appendChild(item);
  });
}

function renderDocumentList(names) {
  if (!Array.isArray(names) || !names.length) {
    clearDocumentList();
    return;
  }
  state.documentNames = [...names];
  docList.innerHTML = "";
  names.forEach((name) => {
    const item = document.createElement("div");
    item.className = "doc-item";
    item.textContent = name;
    docList.appendChild(item);
  });
}

function renderSources(sources) {
  if (!Array.isArray(sources) || !sources.length) {
    clearSources();
    return;
  }

  sourceRefs.innerHTML = "";
  sources.forEach((s) => {
    const item = document.createElement("div");
    item.className = "source-item";
    const title = document.createElement("strong");
    const chunkValue = Number(s.chunk_index);
    const chunkLabel = chunkValue >= 0 ? `chunk ${chunkValue + 1}` : "metadata";
    title.textContent = `${s.source} (${chunkLabel})`;
    const body = document.createElement("span");
    body.textContent = s.snippet || "";
    item.appendChild(title);
    item.appendChild(body);
    sourceRefs.appendChild(item);
  });
}

function renderMainState() {
  const hasDocs = state.documentNames.length > 0;
  const shouldShowChatList = hasDocs || state.uploading || state.messages.length > 0;
  const sectionHead = chatView.querySelector(".section-head");
  const showHero = !hasDocs && !state.uploading && state.messages.length === 0;

  homeView.classList.toggle("active", showHero);
  if (sectionHead) {
    sectionHead.style.display = hasDocs ? "block" : "none";
  }
  chatBox.style.display = shouldShowChatList ? "flex" : "none";
}

function startLoopAnimation(text = "Something is cooking...") {
  const item = document.createElement("div");
  item.className = "msg bot loading";
  const textNode = document.createElement("span");
  item.appendChild(textNode);
  chatBox.appendChild(item);
  chatBox.scrollTop = chatBox.scrollHeight;

  let index = 0;
  let forward = true;

  const timer = setInterval(() => {
    textNode.textContent = text.slice(0, index);
    if (forward) {
      index += 1;
      if (index > text.length) {
        forward = false;
      }
    } else {
      index -= 1;
      if (index < 3) {
        forward = true;
      }
    }
  }, 55);

  return {
    stop() {
      clearInterval(timer);
      item.remove();
    }
  };
}

function addTypewriterMessage(text, speed = 24) {
  const item = document.createElement("div");
  item.className = "msg bot";
  chatBox.appendChild(item);
  chatBox.scrollTop = chatBox.scrollHeight;

  state.messages.push({ role: "bot", text });
  msgCount.textContent = String(state.messages.length);

  let index = 0;
  const timer = setInterval(() => {
    item.textContent = text.slice(0, index);
    index += 1;
    if (index > text.length) {
      clearInterval(timer);
    }
  }, speed);
}

function setupPromptAnimation() {
  if (!chatInput) return;
  if (chatInput.dataset.placeholderInit === "1") return;
  chatInput.dataset.placeholderInit = "1";

  const phrases = [
    "Summarize this chapter...",
    "Make a mind map from this topic...",
    "Explain a key term...",
    "Compare two sections..."
  ];
  let phraseIdx = 0;
  let typingTimer = null;
  let cycleTimer = null;

  const shouldAnimate = () => document.activeElement !== chatInput && !chatInput.value;

  const clearTyping = () => {
    if (typingTimer) {
      clearInterval(typingTimer);
      typingTimer = null;
    }
  };

  const typePhrase = (text) => {
    clearTyping();
    chatInput.placeholder = "";
    let i = 0;
    typingTimer = setInterval(() => {
      if (!shouldAnimate()) {
        clearTyping();
        return;
      }
      i += 1;
      chatInput.placeholder = text.slice(0, i);
      if (i >= text.length) {
        clearTyping();
      }
    }, 32);
  };

  const startCycle = () => {
    if (cycleTimer) clearInterval(cycleTimer);
    cycleTimer = setInterval(() => {
      if (!shouldAnimate()) return;
      typePhrase(phrases[phraseIdx]);
      phraseIdx = (phraseIdx + 1) % phrases.length;
    }, 2100);
    typePhrase(phrases[phraseIdx]);
    phraseIdx = (phraseIdx + 1) % phrases.length;
  };

  const stopCycle = () => {
    clearTyping();
    if (cycleTimer) {
      clearInterval(cycleTimer);
      cycleTimer = null;
    }
    chatInput.placeholder = "";
  };

  chatInput.addEventListener("focus", stopCycle);
  chatInput.addEventListener("input", () => {
    if (chatInput.value) stopCycle();
  });
  chatInput.addEventListener("blur", () => {
    if (!chatInput.value) startCycle();
  });

  startCycle();
}

function switchView(view) {
  const isQuiz = view === "quiz";
  chatView.classList.toggle("active", !isQuiz);
  quizView.classList.toggle("active", isQuiz);
}

function renderQuiz(quizItems) {
  state.quizQuestions = quizItems;
  quizContainer.innerHTML = "";
  quizResult.textContent = "";

  quizItems.forEach((q, index) => {
    const card = document.createElement("div");
    card.className = "quiz-card";

    const qText = document.createElement("p");
    qText.textContent = `${index + 1}. ${q.question}`;
    card.appendChild(qText);

    if (q.question_type === "short_answer") {
      const input = document.createElement("textarea");
      input.name = `q-${index}`;
      input.rows = 3;
      input.placeholder = "Write your short answer...";
      input.style.width = "100%";
      card.appendChild(input);
    } else {
      q.options.forEach((option, optionIndex) => {
        const label = document.createElement("label");
        const radio = document.createElement("input");
        radio.type = "radio";
        radio.name = `q-${index}`;
        radio.value = String(optionIndex);
        label.appendChild(radio);
        label.append(` ${option}`);
        card.appendChild(label);
      });
    }

    quizContainer.appendChild(card);
  });
}

async function uploadDocument(file, existingSessionId = null) {
  const formData = new FormData();
  formData.append("file", file);
  if (existingSessionId) {
    formData.append("session_id", existingSessionId);
  }

  const res = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Upload failed");
  }

  return res.json();
}

function enqueueFiles(files) {
  files.forEach((file, idx) => {
    state.uploadQueue.push({
      id: `${Date.now()}-${idx}`,
      file
    });
  });
}

function updateUploadStatus(currentFileName = "") {
  const queuedCount = state.uploadQueue.length;
  if (state.uploading) {
    const readyCount = state.documentNames.length;
    const tail = queuedCount > 0 ? ` (${queuedCount} waiting)` : "";
    fileStatus.textContent = `Indexed: ${readyCount} file(s). Indexing now: ${currentFileName}${tail}`;
    renderQueueList(currentFileName);
    return;
  }
  if (state.documentNames.length > 0) {
    fileStatus.textContent = `Ready. Indexed ${state.documentNames.length} file(s).`;
  } else {
    fileStatus.textContent = "No document selected.";
  }
  renderQueueList("");
}

async function processUploadQueue() {
  if (state.uploading || state.uploadQueue.length === 0) {
    return;
  }

  state.uploading = true;
  renderMainState();
  const uploadAnimation = startLoopAnimation("Something is cooking...");

  while (state.uploadQueue.length > 0) {
    const next = state.uploadQueue.shift();
    updateUploadStatus(next.file.name);
    renderQueueList(next.file.name);

    try {
      const result = await uploadDocument(next.file, state.sessionId);
      state.sessionId = result.session_id;
      renderDocumentList(result.document_names || []);
      docName.textContent = result.uploaded_document || next.file.name;

      if (!state.readyPromptShown) {
        addTypewriterMessage("First file is ready. Ask me now while I keep indexing the rest.");
        state.readyPromptShown = true;
      }
    } catch (error) {
      console.error("Upload failed:", error);
      addMessage("bot", `Upload failed for ${next.file.name}: ${error.message}`);
    }
  }

  uploadAnimation.stop();
  state.uploading = false;
  updateUploadStatus();
  renderQueueList("");
  docUpload.value = "";
  renderMainState();
}

async function askQuestion(question) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, question })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Chat request failed");
  }

  return res.json();
}

async function generateQuiz() {
  const res = await fetch(`${API_BASE}/quiz/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, num_questions: 5 })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Quiz generation failed");
  }

  return res.json();
}

async function gradeQuiz(answers) {
  const res = await fetch(`${API_BASE}/quiz/grade`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: state.sessionId, answers })
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || "Quiz grading failed");
  }

  return res.json();
}

docUpload.addEventListener("change", async () => {
  const files = Array.from(docUpload.files || []);

  if (!files.length) {
    return;
  }

  if (!state.sessionId && state.documentNames.length === 0 && !state.uploading) {
    clearChat();
    clearSources();
    clearDocumentList();
    clearQueueList();
    docName.textContent = "-";
    state.readyPromptShown = false;
    renderMainState();
  }

  enqueueFiles(files);
  updateUploadStatus(files[0].name);
  await processUploadQueue();
});

if (attachBtn) {
  attachBtn.addEventListener("click", () => {
    docUpload.click();
  });
}

if (sidebarToggle) {
  sidebarToggle.addEventListener("click", () => {
    const isOpen = !document.body.classList.contains("sidebar-open");
    setSidebarOpen(isOpen);
  });
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const text = chatInput.value.trim();
  if (!text) return;

  addMessage("user", text);
  chatInput.value = "";

  if (!state.sessionId) {
    addMessage("bot", "Upload a document first so I can answer from it.");
    return;
  }

  const responseAnimation = startLoopAnimation("Generating response...");
  try {
    const result = await askQuestion(text);
    const content = result.answer || "No answer returned.";
    renderSources(result.sources || []);
    addMessage("bot", content);
  } catch (error) {
    addMessage("bot", `Chat error: ${error.message}`);
  } finally {
    responseAnimation.stop();
  }
});

takeTestBtn.addEventListener("click", async () => {
  if (!state.sessionId) {
    fileStatus.textContent = "Upload a document before taking test.";
    return;
  }

  fileStatus.textContent = "Generating quiz...";

  try {
    const result = await generateQuiz();
    renderQuiz(result.quiz || []);
    switchView("quiz");
    fileStatus.textContent = `Quiz ready (${result.count || 0} questions).`;
  } catch (error) {
    fileStatus.textContent = `Quiz error: ${error.message}`;
  }
});

submitQuizBtn.addEventListener("click", async () => {
  if (!state.sessionId || !state.quizQuestions.length) return;

  const answers = state.quizQuestions.map((_, index) => {
    const question = state.quizQuestions[index];
    if (question.question_type === "short_answer") {
      const textInput = document.querySelector(`textarea[name=\"q-${index}\"]`);
      return textInput ? textInput.value.trim() : "";
    }
    const selected = document.querySelector(`input[name=\"q-${index}\"]:checked`);
    return selected ? Number(selected.value) : -1;
  });

  try {
    const result = await gradeQuiz(answers);
    const details = Array.isArray(result.details) ? result.details : [];
    const feedbackText = details
      .map((d) => `Q${d.question_number}: ${d.correct ? "Correct" : "Review"} - ${d.explanation || ""}`)
      .join(" | ");
    quizResult.textContent = `Score: ${result.score}/${result.total} (${result.percent}%). ${feedbackText}`;
  } catch (error) {
    quizResult.textContent = `Grading error: ${error.message}`;
  }
});

backToChatBtn.addEventListener("click", () => {
  switchView("chat");
});

async function playIntro() {
  const message = INTRO_MESSAGES[Math.floor(Math.random() * INTRO_MESSAGES.length)];
  introText.textContent = "";
  for (let i = 1; i <= message.length; i += 1) {
    introText.textContent = message.slice(0, i);
    await new Promise((resolve) => setTimeout(resolve, 35));
  }
  await new Promise((resolve) => setTimeout(resolve, 450));
  document.body.classList.add("app-ready");
  setSidebarOpen(false);
  renderMainState();
  if (introScreen) {
    introScreen.setAttribute("aria-hidden", "true");
  }
  setupPromptAnimation();
}

playIntro();
