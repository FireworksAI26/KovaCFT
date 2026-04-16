# cft_server.py — CFT Backend Server
# Runs on RunPod/Lightning AI, exposes API for the CFT website

import json, os, re, time
from pathlib import Path
from threading import Lock

import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
import uvicorn

# ── Config ──
BASE_MODEL = os.environ.get('BASE_MODEL_ID', 'Qwen/Qwen3.5-4B')
CHECKPOINT_DIR = os.environ.get('CFT_CHECKPOINT', '/root/cft-checkpoints')
PORT = int(os.environ.get('CFT_PORT', '7860'))
LR = float(os.environ.get('CFT_LR', '1e-5'))

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# ── App ──
app = FastAPI(title='CFT Server')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

# ── State ──
model = None
tokenizer = None
optimizer = None
model_lock = Lock()
training_active = True
step = 0
scores_history = []
conversation = []  # list of {"role": "user"|"assistant", "content": str}

# ── Models ──
class ChatRequest(BaseModel):
    message: str

class ScoreRequest(BaseModel):
    score: int  # 1-10
    message_index: int = -1

class CorrectRequest(BaseModel):
    correction: str
    message_index: int = -1

class UploadRequest(BaseModel):
    repo_name: str

# ── Load model on startup ──
@app.on_event('startup')
async def load_model():
    global model, tokenizer, optimizer
    print(f'Loading {BASE_MODEL}...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )
    
    # Add LoRA
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    model = get_peft_model(model, lora_config)
    model.train()
    
    # Optimizer for LoRA params only
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LR)
    
    print(f'Model loaded. LoRA params: {sum(p.numel() for p in lora_params):,}')
    print(f'CFT server ready on port {PORT}')

# ── Chat endpoint ──
@app.post('/chat')
async def chat(req: ChatRequest):
    global step
    with model_lock:
        # Add user message to conversation
        conversation.append({'role': 'user', 'content': req.message})
        
        # Check if message is a score (e.g. "7/10")
        score_match = re.match(r'^(\d+)/10$', req.message.strip())
        if score_match:
            score_val = int(score_match.group(1))
            reward = _compute_reward(score_val)
            _train_step(reward)
            return {'response': f'Score {score_val}/10 recorded ✓ (reward: {reward:.2f}). Training step {step} complete. Keep chatting!', 'step': step}
        
        # Build prompt from conversation
        prompt = _build_prompt(conversation)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=True,
                temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        conversation.append({'role': 'assistant', 'content': response})
        
        return {'response': response, 'step': step}

# ── Score endpoint ──
@app.post('/score')
async def score(req: ScoreRequest):
    global step
    if not training_active:
        raise HTTPException(400, 'Training is stopped')
    if req.score < 1 or req.score > 10:
        raise HTTPException(400, 'Score must be 1-10')
    
    with model_lock:
        reward = _compute_reward(req.score)
        _train_step(reward)
        scores_history.append({'score': req.score, 'reward': reward, 'step': step})
        return {'status': 'ok', 'reward': reward, 'step': step}

# ── Correct endpoint ──
@app.post('/correct')
async def correct(req: CorrectRequest):
    global step
    if not training_active:
        raise HTTPException(400, 'Training is stopped')
    
    with model_lock:
        # Train on the correction as a positive example
        _train_on_correction(req.correction)
        scores_history.append({'score': 0, 'reward': -1.0, 'step': step, 'corrected': True})
        return {'status': 'ok', 'step': step}

# ── Status endpoint ──
@app.get('/status')
async def status():
    avg = np.mean([s['score'] for s in scores_history[-20:]]) if scores_history else 0
    return {
        'step': step,
        'total_scores': len(scores_history),
        'avg_score': round(float(avg), 1),
        'training': training_active,
        'total_messages': len(conversation),
        'model': BASE_MODEL
    }

# ── Stop endpoint ──
@app.post('/stop')
async def stop_training():
    global training_active
    training_active = False
    # Save adapter
    save_path = os.path.join(CHECKPOINT_DIR, 'cft-adapter')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # Save conversation log
    with open(os.path.join(CHECKPOINT_DIR, 'conversation.json'), 'w') as f:
        json.dump({'conversation': conversation, 'scores': scores_history}, f, indent=2)
    return {'status': 'stopped', 'checkpoint_path': save_path, 'step': step}

# ── Resume endpoint ──
@app.post('/resume')
async def resume_training():
    global training_active
    training_active = True
    return {'status': 'resumed'}

# ── Upload endpoint ──
@app.post('/upload')
async def upload(req: UploadRequest):
    save_path = os.path.join(CHECKPOINT_DIR, 'cft-adapter')
    if not os.path.exists(save_path):
        raise HTTPException(400, 'No checkpoint found. Stop training first.')
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get('HF_TOKEN'))
        api.upload_folder(folder_path=save_path, repo_id=req.repo_name, repo_type='model')
        return {'status': 'uploaded', 'url': f'https://huggingface.co/{req.repo_name}'}
    except Exception as e:
        raise HTTPException(500, str(e))

# ── Internal helpers ──
def _compute_reward(score_1_to_10: int) -> float:
    """Convert 1-10 score to -1.0 to 1.0 reward."""
    return (score_1_to_10 - 5.5) / 4.5  # 1→-1.0, 5→-0.11, 6→0.11, 10→1.0

def _build_prompt(conv: list) -> str:
    """Build a chat prompt from conversation history."""
    parts = []
    for msg in conv[-10:]:  # Last 10 messages for context
        if msg['role'] == 'user':
            parts.append(f"User: {msg['content']}")
        else:
            parts.append(f"Assistant: {msg['content']}")
    parts.append('Assistant:')
    return '\n'.join(parts)

def _train_step(reward: float):
    """Run one LoRA training step using the last exchange."""
    global step
    if not training_active or len(conversation) < 2:
        return
    
    # Get the last assistant response
    last_assistant = None
    last_user = None
    for msg in reversed(conversation):
        if msg['role'] == 'assistant' and last_assistant is None:
            last_assistant = msg['content']
        elif msg['role'] == 'user' and last_user is None:
            last_user = msg['content']
        if last_assistant and last_user:
            break
    
    if not last_assistant or not last_user:
        return
    
    # Build training text
    train_text = f"User: {last_user}\nAssistant: {last_assistant}"
    inputs = tokenizer(train_text, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    
    # Forward pass
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss * reward  # Scale loss by reward (negative reward = push away)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    step += 1

def _train_on_correction(correction: str):
    """Train on a correction — the correction is the right answer."""
    global step
    if not training_active or len(conversation) < 1:
        return
    
    # Find the last user prompt
    last_user = None
    for msg in reversed(conversation):
        if msg['role'] == 'user':
            last_user = msg['content']
            break
    
    if not last_user:
        return
    
    # Train on correction as positive example
    train_text = f"User: {last_user}\nAssistant: {correction}"
    inputs = tokenizer(train_text, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss  # Positive example — minimize loss directly
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    step += 1

# ── Run ──
if __name__ == '__main__':
    pod_id = os.environ.get('RUNPOD_POD_ID', None)
    colab = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ
    print(f'\n{"="*50}')
    print(f'CFT Server Starting!')
    if pod_id:
        print(f'\n  Your Pod URL: https://{pod_id}-{PORT}.proxy.runpod.net')
        print(f'  Paste this URL into the CFT website to connect.')
    elif colab:
        print(f'\n  Running on Colab — use ngrok or localtunnel to expose port {PORT}.')
        print(f'  Or run: !npx localtunnel --port {PORT}')
    else:
        print(f'\n  Server running on: http://localhost:{PORT}')
    print(f'{"="*50}\n')
    uvicorn.run(app, host='0.0.0.0', port=PORT)
