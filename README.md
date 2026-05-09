# EU Battery Regulation LLM

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-birol91%2Feu--battery--regulation--llm-yellow?logo=huggingface)
![Version](https://img.shields.io/badge/Version-v1.0-orange)

**A fine-tuned Mistral 7B model with RAG pipeline for answering questions about EU Battery Regulation, ECE R100, UN 38.3, and related standards.**

---

## Why This Project Exists

EU Battery Regulation (2023/1542) and its surrounding standards — ECE R100, ECE R155, UN 38.3, Battery Passport requirements — are dense, cross-referenced, and frequently amended. Practitioners in automotive, energy storage, and compliance roles spend significant time manually searching articles, annexes, and amendment texts.

This project fine-tunes a 7B model on regulatory Q&A pairs so it learns the domain's vocabulary, citation style, and reasoning patterns — then augments it with a retrieval pipeline that grounds every answer in the actual source documents. The result: an expert assistant that scores **4.36 / 5.00** on a blind 7-question evaluation, within range of frontier closed models.

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                     FastAPI Backend                      │
│                                                          │
│  Query → BAAI/bge-base-en-v1.5 Embedding                │
│       → FAISS Index (EU regulatory corpus)               │
│       → Top-20 candidate chunks                          │
│       → CrossEncoder Reranker → Top-5 chunks             │
│       → Context Assembly + Citation Tags                 │
│       → Mistral 7B LoRA Q8 (llama-cpp-python)           │
│       → Answer with (Source, Heading) citations          │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Chat UI (index.html) at http://localhost:7860
```

### Component Breakdown

| Component | Detail |
|---|---|
| Base Model | Mistral 7B Instruct v0.3 |
| Fine-tuning | LoRA (r=16, alpha=32, target: q_proj / v_proj) |
| Quantization | Q8 GGUF — 7.2 GB, no quality loss vs FP16 |
| Embedding | BAAI/bge-base-en-v1.5 (768-dim, EN-optimized) |
| Vector Store | FAISS flat-L2 index |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Retrieval | Top-20 FAISS → Top-5 after reranking |
| Inference | llama-cpp-python (Metal on Apple Silicon / CUDA on NVIDIA) |
| Backend | FastAPI |
| Frontend | Vanilla HTML/JS chat UI |

### How It Works

1. **LoRA fine-tuning** teaches the model EU regulatory jargon, article numbering conventions, and citation style (`*(Regulation, Article X)*`) from 2,593 synthetic Q&A pairs.
2. **RAG pipeline** retrieves the most relevant chunks from the regulatory corpus at inference time — the model never relies on memorized facts alone.
3. **CrossEncoder reranking** re-scores retrieved chunks with a dedicated relevance model before injecting them into the prompt, cutting noise from FAISS approximate-match results.
4. **Every answer sentence** carries a `(Source, Heading)` citation so the user can verify the claim in the original document.

---

## 📊 Evaluation Results

Blind evaluation on 7 unseen regulatory questions. Scored by a panel of Claude Opus 4.7, human reviewer, and automated rubric (accuracy, citation quality, completeness, no hallucination).

| Model | Score / 5.00 | Notes |
|---|---|---|
| Claude Opus 4.7 | **5.00** | Anchor-high (closed, no domain FT) |
| Gemini Pro 3.1 | 4.72 | Closed model |
| ChatGPT 5.5 | 4.64 | Closed model |
| **Mistral 7B LoRA Q8 + RAG v3** | **4.36** | **This project** |
| Mistral 7B LoRA Q8 (no RAG) | 1.93 | Shows RAG impact |
| Mistral 7B Base Q4 | 1.32 | Anchor-low (no FT, no RAG) |

Key takeaway: RAG lifts a 7B local model from **1.93 → 4.36**, closing 85% of the gap to frontier models. Without RAG, even fine-tuned weights are insufficient for citation-accurate regulatory answers.

---

## 📋 Covered Regulations

- **EU 2023/1542** — EU Battery Regulation (full text + annexes)
- **ECE R100 Part II** — Electric Vehicle Safety
- **ECE R155** — Cybersecurity Management System
- **UN 38.3** — Transport of Lithium Batteries
- **Battery Passport** documentation (Digital Product Passport spec)
- **2025 amendments** to EU 2023/1542

---

## 🚀 Quick Start

### Requirements

- Python 3.10+
- 8 GB+ VRAM (Apple Silicon M-series or NVIDIA GPU)
- 10 GB free disk space
- Git

> **Note on VRAM:** The Q8 GGUF model is 7.2 GB. Apple M-series uses unified memory — 16 GB RAM is sufficient. On NVIDIA, 8 GB VRAM (RTX 3080 / A10 / etc.) works with llama-cpp-python CUDA build.

### Installation

```bash
git clone https://github.com/birol91/eu-battery-regulation-llm
cd eu-battery-regulation-llm
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open your browser at [http://localhost:7860](http://localhost:7860).

On first launch, the app automatically downloads from HuggingFace:
- `birol91/eu-battery-regulation-llm` — Q8 GGUF weights (~7.2 GB)
- FAISS index + regulatory corpus chunks (~0.5 GB)

Total download: ~8 GB. Subsequent launches load from local cache instantly.

### Environment Variables (optional)

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | — | HuggingFace token (only needed if repo is private) |
| `PORT` | `7860` | Server port |
| `N_GPU_LAYERS` | `-1` | GPU layers for llama-cpp (`-1` = all) |

---

## 📁 Project Structure

```
eu-battery-regulation-llm/
├── app.py                  # FastAPI entry point
├── src/
│   └── data/               # Corpus preprocessing utilities
├── phase_4/                # LoRA fine-tuning scripts & checkpoints
├── phase_5/                # RAG pipeline (inference.py, reranker)
├── phase_6/                # Evaluation reports & scoring
├── Dataset/                # Raw regulatory PDFs
├── requirements.txt
└── index.html              # Chat UI served by FastAPI
```

---

## 🤗 HuggingFace Model

Model weights, GGUF quantization, and dataset are published at:

**[huggingface.co/birol91/eu-battery-regulation-llm](https://huggingface.co/birol91/eu-battery-regulation-llm)**

The repo contains:
- `mistral-7b-eu-battery-lora-q8.gguf` — production model
- `faiss_index/` — prebuilt FAISS index for the regulatory corpus
- Training dataset (2,593 EN-only Q&A pairs, ChatML format)

---

## License

MIT — see [LICENSE](LICENSE).

The underlying Mistral 7B model is released under the Apache 2.0 license by Mistral AI.  
Regulatory source documents (EU 2023/1542, ECE standards) are public domain EU/UN publications.
