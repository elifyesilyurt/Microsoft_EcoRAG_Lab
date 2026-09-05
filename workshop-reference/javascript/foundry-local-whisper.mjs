/**
 * Whisper Voice Transcription with Foundry Local
 * Transcribes WAV audio files using the OpenAI Whisper model running locally.
 *
 * Uses the Foundry Local SDK's built-in AudioClient for transcription,
 * which handles all ONNX inference and audio preprocessing internally.
 *
 * Requires: foundry-local-sdk
 * Usage: node foundry-local-whisper.mjs [path-to-wav-file]
 */

import { FoundryLocalManager } from "foundry-local-sdk";
import fs from "node:fs";
import path from "node:path";
import os from "node:os";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const modelAlias = "whisper-medium";

// ---- Audio file discovery ------------------------------------------------

const samplesDir = path.join(__dirname, "..", "samples", "audio");
let audioFiles;

if (process.argv[2]) {
  const filePath = process.argv[2];
  if (!fs.existsSync(filePath)) {
    console.error(`Audio file not found: ${filePath}`);
    process.exit(1);
  }
  audioFiles = [filePath];
} else {
  if (!fs.existsSync(samplesDir)) {
    console.error(`Samples directory not found: ${samplesDir}`);
    process.exit(1);
  }
  audioFiles = fs
    .readdirSync(samplesDir)
    .filter((f) => f.startsWith("zava-") && f.endsWith(".wav"))
    .sort()
    .map((f) => path.join(samplesDir, f));
  if (audioFiles.length === 0) {
    console.error(`No WAV files found in ${samplesDir}`);
    process.exit(1);
  }
}

// ---- Step 1: Start Foundry Local and prepare the model -------------------

console.log(`Initialising Foundry Local with model: ${modelAlias}...`);
FoundryLocalManager.create({ appName: "FoundryLocalWorkshop" });
const manager = FoundryLocalManager.instance;
await manager.startWebService();

const catalog = manager.catalog;
const model = await catalog.getModel(modelAlias);

if (!model.isCached) {
  console.log(`Downloading model: ${modelAlias} (this may take several minutes)...`);
  await model.download();
  console.log(`Download complete: ${modelAlias}`);
}

await model.load();
console.log(`Model ready: ${model.id}`);

// ---- Step 2: Create an AudioClient for transcription ---------------------

const audioClient = model.createAudioClient();
audioClient.settings.language = "en";
console.log("Audio client ready.\n");

// ---- Helper: split a WAV file into ≤30-second chunk files ----------------
// Whisper's encoder processes exactly 30 seconds per pass. For longer files
// we split the PCM data, write temporary WAV files, and transcribe each.
// See https://github.com/microsoft/Foundry-Local/issues/517

const CHUNK_SECONDS = 30;

function splitWavIntoChunks(wavPath) {
  const buf = fs.readFileSync(wavPath);
  // Parse WAV header (standard 44-byte RIFF/PCM header)
  const numChannels = buf.readUInt16LE(22);
  const sampleRate = buf.readUInt32LE(24);
  const bitsPerSample = buf.readUInt16LE(34);
  const bytesPerSample = (bitsPerSample / 8) * numChannels;
  const headerSize = 44;
  const pcmData = buf.subarray(headerSize);
  const totalSamples = pcmData.length / bytesPerSample;
  const chunkSamples = CHUNK_SECONDS * sampleRate;

  if (totalSamples <= chunkSamples) return null; // no splitting needed

  const numChunks = Math.ceil(totalSamples / chunkSamples);
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "whisper-chunks-"));
  const chunkPaths = [];

  for (let i = 0; i < numChunks; i++) {
    const startByte = i * chunkSamples * bytesPerSample;
    const endByte = Math.min(startByte + chunkSamples * bytesPerSample, pcmData.length);
    const chunkPcm = pcmData.subarray(startByte, endByte);
    const chunkLen = chunkPcm.length;

    // Build a new WAV header for this chunk
    const header = Buffer.alloc(headerSize);
    header.write("RIFF", 0);
    header.writeUInt32LE(36 + chunkLen, 4);
    header.write("WAVE", 8);
    header.write("fmt ", 12);
    header.writeUInt32LE(16, 16); // fmt chunk size
    header.writeUInt16LE(1, 20);  // PCM format
    header.writeUInt16LE(numChannels, 22);
    header.writeUInt32LE(sampleRate, 24);
    header.writeUInt32LE(sampleRate * bytesPerSample, 28);
    header.writeUInt16LE(bytesPerSample, 32);
    header.writeUInt16LE(bitsPerSample, 34);
    header.write("data", 36);
    header.writeUInt32LE(chunkLen, 40);

    const chunkPath = path.join(tmpDir, `chunk-${i}.wav`);
    fs.writeFileSync(chunkPath, Buffer.concat([header, chunkPcm]));
    chunkPaths.push(chunkPath);
  }

  return { chunkPaths, tmpDir, numChunks, duration: totalSamples / sampleRate };
}

// ---- Step 3: Transcribe each audio file ----------------------------------

for (const audioPath of audioFiles) {
  const filename = path.basename(audioPath);
  console.log("=".repeat(60));
  console.log(`File: ${filename}`);
  console.log("=".repeat(60));

  const t0 = Date.now();
  const chunks = splitWavIntoChunks(audioPath);

  let text;
  if (!chunks) {
    // Short audio (≤30 s) — single pass
    const result = await audioClient.transcribe(audioPath);
    text = result.text;
  } else {
    // Long audio — transcribe each chunk and concatenate
    console.log(`  Audio is ${chunks.duration.toFixed(1)}s — splitting into ${chunks.numChunks} chunks of ${CHUNK_SECONDS}s`);
    const parts = [];
    for (let i = 0; i < chunks.chunkPaths.length; i++) {
      const chunkDur = Math.min(CHUNK_SECONDS, chunks.duration - i * CHUNK_SECONDS);
      process.stdout.write(`  Chunk ${i + 1}/${chunks.numChunks} (${chunkDur.toFixed(1)}s)... `);
      const result = await audioClient.transcribe(chunks.chunkPaths[i]);
      console.log("done");
      if (result.text) parts.push(result.text);
    }
    text = parts.join(" ");
    // Clean up temp files
    for (const p of chunks.chunkPaths) fs.unlinkSync(p);
    fs.rmdirSync(chunks.tmpDir);
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(text);
  console.log(`(${elapsed}s)\n`);
}

console.log(`Done — transcribed ${audioFiles.length} file(s).`);

// Cleanup: unload the model to release resources
await model.unload();
await manager.stopWebService();
