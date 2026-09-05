/**
 * Zava Creative Writer — Web Server (JavaScript)
 *
 * Wraps the existing multi-agent orchestrator behind an HTTP server
 * and serves the shared UI from the ../ui/ directory.
 *
 * Endpoints:
 *   GET  /          — serves the static UI
 *   POST /api/article — runs the pipeline, streams NDJSON
 *
 * Usage: node server.mjs
 */

import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { research } from "./researcher.mjs";
import { findProducts } from "./product.mjs";
import { write, processWriterOutput } from "./writer.mjs";
import { edit } from "./editor.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const UI_DIR = path.resolve(__dirname, "..", "ui");
const PORT = 3000;

// MIME types for static files
const MIME = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
};

/** Send a single NDJSON line to the response stream. */
function sendLine(res, obj) {
  res.write(JSON.stringify(obj).replace(/\n/g, "") + "\n");
}

/** Serve a static file from the UI directory. */
function serveStatic(req, res) {
  let filePath = req.url === "/" ? "/index.html" : req.url;

  // Prevent path-traversal attacks
  const safePath = path.normalize(filePath).replace(/^(\.\.[/\\])+/, "");
  const fullPath = path.join(UI_DIR, safePath);
  if (!fullPath.startsWith(UI_DIR)) {
    res.writeHead(403);
    res.end("Forbidden");
    return;
  }

  const ext = path.extname(fullPath);
  const contentType = MIME[ext] || "application/octet-stream";

  fs.readFile(fullPath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end("Not found");
      return;
    }
    res.writeHead(200, { "Content-Type": contentType });
    res.end(data);
  });
}

/** Run the multi-agent pipeline and stream NDJSON to the client. */
async function handleArticle(req, res) {
  // Read the JSON body
  let body = "";
  for await (const chunk of req) {
    body += chunk;
  }

  let input;
  try {
    input = JSON.parse(body);
  } catch {
    res.writeHead(400, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: "Invalid JSON" }));
    return;
  }

  const researchContext = input.research || "";
  const productContext = input.products || "";
  const assignment = input.assignment || "";

  res.writeHead(200, {
    "Content-Type": "text/event-stream; charset=utf-8",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });

  let feedback = "No Feedback";

  try {
    // 1. Researcher
    sendLine(res, { type: "message", message: "Starting researcher agent task...", data: {} });
    let researchResult = await research(researchContext, feedback);
    sendLine(res, { type: "researcher", message: "Completed researcher task", data: researchResult });

    // 2. Product search
    sendLine(res, { type: "message", message: "Starting marketing agent task...", data: {} });
    const productResult = await findProducts(productContext);
    sendLine(res, { type: "marketing", message: "Completed marketing task", data: productResult });

    // 3. Writer (streaming)
    sendLine(res, { type: "message", message: "Starting writer agent task...", data: {} });
    sendLine(res, { type: "writer", message: "Writer started", data: { start: true } });

    let fullText = "";
    for await (const token of write(
      researchContext, researchResult,
      productContext, productResult,
      assignment, feedback
    )) {
      fullText += token;
      sendLine(res, { type: "partial", message: "token", data: { text: token } });
    }

    let processed = processWriterOutput(fullText);
    sendLine(res, { type: "writer", message: "Writer complete", data: { complete: true } });

    // 4. Editor
    sendLine(res, { type: "message", message: "Starting editor agent task...", data: {} });
    let editorResponse = await edit(processed.article, processed.feedback);
    sendLine(res, { type: "editor", message: "Completed editor task", data: editorResponse });

    // Feedback loop (max 2 retries)
    let retryCount = 0;
    while (
      editorResponse.decision &&
      editorResponse.decision.toLowerCase().startsWith("revise") &&
      retryCount < 2
    ) {
      retryCount++;
      const researchFeedback = (editorResponse.researchFeedback || "No Feedback").slice(0, 500);
      const editorFeedback = (editorResponse.editorFeedback || "No Feedback").slice(0, 500);

      sendLine(res, { type: "message", message: `Revision ${retryCount}: re-running pipeline...`, data: {} });

      researchResult = await research(researchContext, researchFeedback);
      sendLine(res, { type: "researcher", message: "Completed researcher task", data: researchResult });

      sendLine(res, { type: "writer", message: "Writer started", data: { start: true } });
      fullText = "";
      for await (const token of write(
        researchContext, researchResult,
        productContext, productResult,
        assignment, editorFeedback
      )) {
        fullText += token;
        sendLine(res, { type: "partial", message: "token", data: { text: token } });
      }

      processed = processWriterOutput(fullText);
      sendLine(res, { type: "writer", message: "Writer complete", data: { complete: true } });

      editorResponse = await edit(processed.article, processed.feedback);
      sendLine(res, { type: "editor", message: "Completed editor task", data: editorResponse });
    }
  } catch (err) {
    sendLine(res, { type: "error", message: err.message, data: { error: String(err) } });
  } finally {
    res.end();
  }
}

// Create HTTP server
const server = http.createServer((req, res) => {
  if (req.method === "POST" && req.url === "/api/article") {
    handleArticle(req, res);
  } else {
    serveStatic(req, res);
  }
});

server.listen(PORT, () => {
  console.log(`\nZava Creative Writer UI is running at http://localhost:${PORT}\n`);
});
