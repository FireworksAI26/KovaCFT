# KOVA CFT — Conversational Fine-Tuning

**Train AI models by chatting with them.** Score responses, correct mistakes, and the model learns in real-time. No datasets. No ML expertise. Just a conversation.

> **The goal: make open-source models better than closed models.** Not by training bigger models with more data — by giving everyone the tools to improve the models they already have. CFT is how we get there.
> 

---

## What is CFT?

CFT (Conversational Fine-Tuning) is a new approach to model training where **you are the reward function.** Instead of curating datasets or writing reward models, you simply:

1. Chat with the model
2. Score its responses (1-10)
3. Correct its mistakes
4. The model updates its weights after every scored exchange

Every conversation makes the model better. Type `stop` when you're happy, upload to HuggingFace, done.

## How It Works

```
You: "Build a React navbar with Tailwind"
Model: \[generates code\]
You: 6/10
→ LoRA weights update (reward = 0.11)
→ Next response will be slightly better

You: "The menu doesn't collapse on mobile"
→ Model learns from your correction
→ Training step complete
```

Under the hood: **LoRA adapters** on any HuggingFace model (default: `Qwen/Qwen3.5-4B`), trained with reward-scaled loss. Scores map to rewards: `1/10` → `-1.0`, `5/10` → `0.0`, `10/10` → `+1.0`.

## Features

- **Manual Chat** — You chat, you score, you train
- **Auto Chat** — Connect an AI judge via MCP to train automatically
- **Code Execution** — Run Python/JS/Bash on the server to test generated code
- **One-Click Deploy** — RunPod template, no manual commands
- **Any Model** — Works with Qwen, Llama, Mistral, or any HuggingFace model
- **Upload to HuggingFace** — One tap after training

## Quick Start

### Option 1: RunPod (one-click)

1. Create a RunPod template with this start command:

```bash
bash -c 'pip install -q fastapi uvicorn peft transformers accelerate bitsandbytes huggingface_hub sentencepiece safetensors numpy "mcp\[server\]" sse-starlette && curl -fsSL https://raw.githubusercontent.com/FireworksAI26/kovacft/refs/heads/main/cft_server.py -o /cft_server.py && python /cft_server.py'
```

2. Set image to `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`

3. Expose port `7860`

4. Deploy → grab your URL → paste into the CFT website

### Option 2: Lightning AI / Colab

```bash
pip install -q fastapi uvicorn peft transformers accelerate bitsandbytes huggingface_hub sentencepiece safetensors numpy "mcp\[server\]" sse-starlette
```

```bash
curl -fsSL https://raw.githubusercontent.com/FireworksAI26/kovacft/refs/heads/main/cft_server.py -o cft_server.py
```

```bash
python cft_server.py
```

### Option 3: Any GPU machine

```bash
git clone https://github.com/FireworksAI26/kovacft.git
cd kovacft
pip install -r requirements.txt
python cft_server.py
```

## API Endpoints

| **Method** | **Endpoint** | **Description** |
| --- | --- | --- |
| `POST` | `/chat` | Send a message, get model response |
| `POST` | `/score` | Score last response 1-10, triggers training |
| `POST` | `/correct` | Provide correct answer, trains on it |
| `POST` | `/run_code` | Execute Python/JS/Bash code |
| `GET` | `/status` | Training stats (step, avg score, etc.) |
| `POST` | `/stop` | Stop training, save adapter |
| `POST` | `/upload` | Upload adapter to HuggingFace |
| `GET` | `/history` | Last 20 conversation exchanges |
| `GET` | `/mcp/sse` | MCP server for AI judge integration |

## Environment Variables

| **Variable** | **Default** | **Description** |
| --- | --- | --- |
| `BASE_MODEL_ID` | `Qwen/Qwen3.5-4B` | HuggingFace model to fine-tune |
| `CFT_PORT` | `7860` | Server port |
| `CFT_LR` | `1e-5` | Learning rate |
| `HF_TOKEN` | — | HuggingFace token (for upload) |
| `CFT_CHECKPOINT` | `/root/cft-checkpoints` | Where to save adapters |

## CFT vs Other Methods

| **Method** | **Who gives feedback** | **When** | **Domains** | **Requires** |
| --- | --- | --- | --- | --- |
| **SFT** | Pre-written dataset | Before training | Fixed | GPU + datasets |
| **RLHF** | Paid human labelers | Offline | General | GPU + reward model + $$$ |
| **UVR** | Automated verifiers | During training | Code + math | GPU + compilers |
| **CFT** | **You, by chatting** | **Real-time** | **Anything** | **Just a GPU** |

## Part of Kova

Kova is building a suite of open-source training techniques to make open-source models better than closed models. Together, these tools will let anyone — not just big companies — improve AI.

**🟢 CFT** — Conversational Fine-Tuning (this repo) — **available now**

**🔜 UVR** — Unified Verifiable Reinforcement (automated code training) — coming soon

**🔜 UVR for Phones** — On-device training on iPhones and Android — coming soon

CFT is the first technique available. The rest are in development. The mission is simple: **open-source models trained by millions of people on their own devices will collectively outperform any single company's closed model.**

## Cost

~$0.57 per 2-hour training session on RunPod (RTX 3090 Ti at $0.27/hr). That's 100+ scored messages and real model improvement.

## License

MIT

---

*If you can chat, you can train AI.*
