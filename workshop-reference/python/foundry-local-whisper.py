"""
Whisper Voice Transcription with Foundry Local
Transcribes WAV audio files using the OpenAI Whisper model running locally.

Uses the Foundry Local SDK to download and manage the whisper model, then
runs inference directly with ONNX Runtime and the transformers feature
extractor. Requires: foundry-local-sdk, onnxruntime, transformers, librosa

Usage: python foundry-local-whisper.py [path-to-wav-file]
"""

import sys
import os
import glob
import time

import numpy as np
import onnxruntime as ort
import librosa
from transformers import WhisperFeatureExtractor, WhisperTokenizer
from foundry_local import FoundryLocalManager

model_alias = "whisper-medium"

# Default: transcribe all Zava sample WAV files
samples_dir = os.path.join(os.path.dirname(__file__), "..", "samples", "audio")

if len(sys.argv) > 1:
    audio_files = [sys.argv[1]]
    for f in audio_files:
        if not os.path.exists(f):
            print(f"Audio file not found: {f}")
            print("Usage: python foundry-local-whisper.py [path-to-wav-file]")
            sys.exit(1)
else:
    audio_files = sorted(glob.glob(os.path.join(samples_dir, "zava-*.wav")))
    if not audio_files:
        print(f"No WAV files found in {samples_dir}")
        print("Run 'python samples/audio/generate_samples.py' first to generate them.")
        sys.exit(1)

# ---------------------------------------------------------------------------
# Step 1: Use Foundry Local SDK to ensure the whisper model is downloaded
# ---------------------------------------------------------------------------
print(f"Initialising Foundry Local with model: {model_alias}...")
manager = FoundryLocalManager(model_alias)
model_info = manager.get_model_info(model_alias)
cache_location = manager.get_cache_location()

# Build the path to the cached ONNX model files
model_dir = os.path.join(
    cache_location, "Microsoft",
    model_info.id.replace(":", "-"),
    "cpu-fp32"
)

if not os.path.isdir(model_dir):
    print(f"Model directory not found: {model_dir}")
    print("Ensure the whisper model has been downloaded by the SDK.")
    sys.exit(1)

print(f"Model ready: {model_info.id}")
print(f"Model path: {model_dir}")

# ---------------------------------------------------------------------------
# Step 2: Load the encoder and decoder ONNX sessions
# ---------------------------------------------------------------------------
print("\nLoading ONNX encoder and decoder...")
encoder_session = ort.InferenceSession(
    os.path.join(model_dir, "whisper-medium_encoder_fp32.onnx"),
    providers=["CPUExecutionProvider"],
)
decoder_session = ort.InferenceSession(
    os.path.join(model_dir, "whisper-medium_decoder_fp32.onnx"),
    providers=["CPUExecutionProvider"],
)

# Load the feature extractor and tokeniser from the model directory
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_dir)
tokenizer = WhisperTokenizer.from_pretrained(model_dir)

# Whisper decoder dimensions (medium model)
NUM_LAYERS = 24
NUM_HEADS = 16
HEAD_SIZE = 64

# Build the initial decoder token sequence:
# <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|>
sot = tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
notimestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
forced_ids = tokenizer.get_decoder_prompt_ids(language="en", task="transcribe")
INITIAL_TOKENS = [sot] + [tid for _, tid in forced_ids] + [notimestamps]

print("Models loaded and ready.\n")

# Whisper processes exactly 30 seconds of audio per encoder pass.
# For longer files we split into 30-second chunks, transcribe each, and
# concatenate the results.  See https://github.com/microsoft/Foundry-Local/issues/517
CHUNK_SECONDS = 30
SAMPLE_RATE = 16000
CHUNK_SAMPLES = CHUNK_SECONDS * SAMPLE_RATE


def _transcribe_chunk(audio_chunk):
    """Run encoder + decoder on a single ≤30-second audio chunk."""
    features = feature_extractor(audio_chunk, sampling_rate=SAMPLE_RATE, return_tensors="np")
    audio_features = features["input_features"].astype(np.float32)

    encoder_outputs = encoder_session.run(None, {"audio_features": audio_features})
    cross_kv_list = encoder_outputs[1:]

    cross_kv = {}
    for i in range(NUM_LAYERS):
        cross_kv[f"past_key_cross_{i}"] = cross_kv_list[i * 2]
        cross_kv[f"past_value_cross_{i}"] = cross_kv_list[i * 2 + 1]

    self_kv = {}
    for i in range(NUM_LAYERS):
        self_kv[f"past_key_self_{i}"] = np.zeros((1, NUM_HEADS, 0, HEAD_SIZE), dtype=np.float32)
        self_kv[f"past_value_self_{i}"] = np.zeros((1, NUM_HEADS, 0, HEAD_SIZE), dtype=np.float32)

    input_ids = np.array([INITIAL_TOKENS], dtype=np.int32)
    generated = []

    for _ in range(448):
        feeds = {"input_ids": input_ids}
        feeds.update(cross_kv)
        feeds.update(self_kv)

        outputs = decoder_session.run(None, feeds)
        logits = outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))

        if next_token == eot:
            break

        generated.append(next_token)

        for i in range(NUM_LAYERS):
            self_kv[f"past_key_self_{i}"] = outputs[1 + i * 2]
            self_kv[f"past_value_self_{i}"] = outputs[2 + i * 2]

        input_ids = np.array([[next_token]], dtype=np.int32)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def transcribe(audio_path):
    """Transcribe a WAV file of any length by chunking into 30-second segments."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    duration = len(audio) / SAMPLE_RATE

    # Short audio (≤30 s) — single pass
    if len(audio) <= CHUNK_SAMPLES:
        return _transcribe_chunk(audio)

    # Long audio — split into 30-second chunks
    num_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    print(f"  Audio is {duration:.1f}s — splitting into {num_chunks} chunks of {CHUNK_SECONDS}s")
    parts = []
    for idx in range(num_chunks):
        start = idx * CHUNK_SAMPLES
        end = min(start + CHUNK_SAMPLES, len(audio))
        chunk = audio[start:end]
        chunk_dur = len(chunk) / SAMPLE_RATE
        print(f"  Chunk {idx + 1}/{num_chunks} ({chunk_dur:.1f}s)...", end=" ", flush=True)
        text = _transcribe_chunk(chunk)
        print("done")
        if text:
            parts.append(text)

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Step 3: Transcribe each audio file
# ---------------------------------------------------------------------------
for audio_path in audio_files:
    filename = os.path.basename(audio_path)
    print(f"{'=' * 60}")
    print(f"File: {filename}")
    print("=" * 60)

    t0 = time.time()
    text = transcribe(audio_path)
    elapsed = time.time() - t0

    print(text)
    print(f"({elapsed:.1f}s)\n")

print(f"Done — transcribed {len(audio_files)} file(s).")

# Cleanup: unload the model to release resources
manager.unload_model(model_alias)
