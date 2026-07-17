# Foundry Local Integration Lab 🚀

This repository contains a local LLM (Large Language Model) integration setup using the **Foundry Local SDK** and **OpenAI API** standards. The project establishes a secure, cost-effective, and fully offline AI inference pipeline running directly on local hardware.

## ✨ Features
- **Dynamic Port Discovery:** Utilizes `FoundryLocalManager` to automatically detect locally hosted AI endpoints.
- **Hardware-Optimal Variant Selection:** Automatically resolves and connects to the best hardware variant (GPU/NPU/CPU) via `WebGpuExecutionProvider`.
- **Standardized API Client:** Interfaces with local models using official OpenAI client architecture.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **SDK & Tools:** Foundry Local SDK, OpenAI Python SDK
- **Tested Models:** `qwen2.5-0.5b` (GPU), `phi-3.5-mini` (GPU)

## 🏃‍♂️ How to Run
1. Activate the environment:
   ```bash
   source foundry-env/bin/activate
