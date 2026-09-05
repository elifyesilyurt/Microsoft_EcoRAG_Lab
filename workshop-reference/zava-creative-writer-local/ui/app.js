/**
 * Zava Creative Writer — Frontend Application
 *
 * Connects to the backend API (Python / JavaScript / C#) via
 * POST /api/article and consumes the newline-delimited JSON
 * streaming response to update the UI in real time.
 */

const API_URL = "/api/article";

// DOM references
const generateBtn = document.getElementById("generateBtn");
const researchInput = document.getElementById("research");
const productsInput = document.getElementById("products");
const assignmentInput = document.getElementById("assignment");
const articleOutput = document.getElementById("articleOutput");
const researchContent = document.getElementById("researchContent");
const productContent = document.getElementById("productContent");
const editorContent = document.getElementById("editorContent");

// Agent status elements
const statusResearcher = document.querySelector("#status-researcher .agent-state");
const statusMarketing = document.querySelector("#status-marketing .agent-state");
const statusWriter = document.querySelector("#status-writer .agent-state");
const statusEditor = document.querySelector("#status-editor .agent-state");

/**
 * Update an agent status badge.
 * @param {HTMLElement} el - The .agent-state span.
 * @param {"waiting"|"running"|"done"|"error"} state
 */
function setAgentState(el, state) {
  const labels = { waiting: "Waiting", running: "Running\u2026", done: "Done", error: "Error" };
  el.setAttribute("data-state", state);
  el.textContent = labels[state] || state;
}

/** Reset every agent badge to "waiting". */
function resetStatuses() {
  [statusResearcher, statusMarketing, statusWriter, statusEditor].forEach(
    (el) => setAgentState(el, "waiting")
  );
  researchContent.textContent = "No data yet.";
  productContent.textContent = "No data yet.";
  editorContent.textContent = "No data yet.";
}

/**
 * Process a single NDJSON message from the streaming response.
 * @param {object} msg - Parsed JSON message.
 */
function handleMessage(msg) {
  // The Python backend emits { type, message, data }
  const { type, message, data } = msg;

  switch (type) {
    case "message":
      // Status updates (e.g. "Starting researcher agent task...")
      if (message && message.toLowerCase().includes("researcher")) {
        setAgentState(statusResearcher, "running");
      } else if (message && message.toLowerCase().includes("marketing")) {
        setAgentState(statusMarketing, "running");
      } else if (message && message.toLowerCase().includes("writer")) {
        setAgentState(statusWriter, "running");
      } else if (message && message.toLowerCase().includes("editor")) {
        setAgentState(statusEditor, "running");
      }
      break;

    case "researcher":
      setAgentState(statusResearcher, "done");
      researchContent.textContent = JSON.stringify(data, null, 2);
      break;

    case "marketing":
      setAgentState(statusMarketing, "done");
      productContent.textContent = JSON.stringify(data, null, 2);
      break;

    case "partial":
      // Streamed article text (token by token)
      if (data && data.text) {
        articleOutput.textContent += data.text;
      }
      break;

    case "writer":
      if (data && data.start) {
        setAgentState(statusWriter, "running");
        articleOutput.textContent = "";
      } else if (data && data.complete) {
        setAgentState(statusWriter, "done");
      }
      break;

    case "editor":
      setAgentState(statusEditor, "done");
      editorContent.textContent = JSON.stringify(data, null, 2);
      break;

    case "error":
      articleOutput.textContent += "\n[Error] " + (message || "Unknown error");
      break;

    default:
      break;
  }
}

/**
 * Read a streaming response line by line using the Streams API.
 * Each line is a JSON object terminated by a newline character.
 * @param {Response} response - The fetch Response object.
 */
async function readStream(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split on newlines; each complete line is a JSON message
    const lines = buffer.split("\n");
    // The last element may be an incomplete line, so keep it in the buffer
    buffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      try {
        const msg = JSON.parse(trimmed);
        // Only process objects with { type } (NDJSON from orchestrator)
        if (msg && typeof msg === "object" && msg.type) {
          handleMessage(msg);
        }
      } catch {
        // Skip lines that are not valid JSON (e.g. tuples from yield)
      }
    }
  }

  // Process any remaining data in the buffer
  if (buffer.trim()) {
    try {
      const msg = JSON.parse(buffer.trim());
      if (msg && typeof msg === "object" && msg.type) {
        handleMessage(msg);
      }
    } catch {
      // Ignore
    }
  }
}

/** Main click handler: send the request and stream the response. */
async function generate() {
  const research = researchInput.value.trim();
  const products = productsInput.value.trim();
  const assignment = assignmentInput.value.trim();

  if (!research || !products || !assignment) {
    alert("Please fill in all three fields before generating.");
    return;
  }

  // Disable button and reset UI
  generateBtn.disabled = true;
  generateBtn.textContent = "Generating\u2026";
  resetStatuses();
  articleOutput.textContent = "";
  articleOutput.classList.remove("placeholder-text");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ research, products, assignment }),
    });

    if (!response.ok) {
      throw new Error("Server returned " + response.status);
    }

    await readStream(response);
  } catch (err) {
    articleOutput.textContent += "\n[Error] " + err.message;
    [statusResearcher, statusMarketing, statusWriter, statusEditor].forEach(
      (el) => { if (el.getAttribute("data-state") === "running") setAgentState(el, "error"); }
    );
  } finally {
    generateBtn.disabled = false;
    generateBtn.textContent = "Generate Article";
  }
}

generateBtn.addEventListener("click", generate);
