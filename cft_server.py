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

class RunCodeRequest(BaseModel):
    language: str  # python, javascript, bash
    code: str
    timeout: int = 30

class AutoTrainRequest(BaseModel):
    focus: str = 'general'  # react, python, javascript, etc.
    rounds: int = 20
    method: str = 'code_execution'  # code_execution, ai_judge, both

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

# ── Run Code endpoint (for Auto Chat) ──
@app.post('/run_code')
async def run_code(req: RunCodeRequest):
    import subprocess, tempfile
    try:
        if req.language == 'python':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(req.code)
                f.flush()
                result = subprocess.run(['python', f.name], capture_output=True, text=True, timeout=req.timeout)
        elif req.language in ('javascript', 'typescript'):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(req.code)
                f.flush()
                result = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=req.timeout)
        elif req.language == 'bash':
            result = subprocess.run(['bash', '-c', req.code], capture_output=True, text=True, timeout=req.timeout)
        else:
            return {'stdout': '', 'stderr': f'Unsupported language: {req.language}', 'exit_code': 1, 'error': 'unsupported_language'}
        return {'stdout': result.stdout[:5000], 'stderr': result.stderr[:5000], 'exit_code': result.returncode, 'error': None}
    except subprocess.TimeoutExpired:
        return {'stdout': '', 'stderr': 'Execution timed out', 'exit_code': 1, 'error': 'timeout'}
    except Exception as e:
        return {'stdout': '', 'stderr': str(e), 'exit_code': 1, 'error': str(e)}

# ── Auto Train state ──
auto_train_active = False
auto_train_paused = False
auto_round = 0
auto_total_rounds = 0
auto_pass_count = 0
auto_exchanges = []
auto_focus = 'general'
auto_method = 'code_execution'

PROMPT_TEMPLATES = {
    'react': [
        'Build a React component for a responsive navbar with mobile hamburger menu using Tailwind CSS.',
        'Create a React hook called useDebounce that debounces a value with a configurable delay.',
        'Build a React todo app with add, delete, and toggle complete. Use useState and Tailwind.',
        'Create a React modal component that closes on overlay click and Escape key press.',
        'Build a React form with email and password validation, showing inline error messages.',
        'Create a React dark mode toggle that persists the preference in localStorage.',
        'Build a React accordion component where only one section can be open at a time.',
        'Create a responsive React pricing page with 3 tiers using Tailwind CSS grid.',
        'Build a React infinite scroll component that loads more items when reaching the bottom.',
        'Create a React search bar with live filtering of a list of items.',
    ],
    'python': [
        'Write a Python function that finds all prime numbers up to n using the Sieve of Eratosthenes.',
        'Create a Python class for a binary search tree with insert, search, and delete methods.',
        'Write a Python script that reads a CSV file and outputs statistics (mean, median, mode) for each numeric column.',
        'Implement a Python LRU cache from scratch without using functools.',
        'Write a Python async web scraper that fetches 5 URLs concurrently using aiohttp.',
        'Create a Python decorator that retries a function up to 3 times with exponential backoff.',
        'Write a Python function to flatten a deeply nested dictionary into dot-notation keys.',
        'Implement merge sort in Python with type hints.',
        'Write a Python context manager for timing code execution.',
        'Create a Python dataclass for a REST API response with validation.',
    ],
    'javascript': [
        'Write a debounce function in JavaScript that handles both leading and trailing edge calls.',
        'Implement a simple Promise.all from scratch in JavaScript.',
        'Write a JavaScript function that deep clones an object, handling circular references.',
        'Create a JavaScript event emitter class with on, off, and emit methods.',
        'Write a JavaScript function to convert a nested object to a flat object with dot-notation keys.',
        'Implement a basic pub/sub system in JavaScript.',
        'Write a JavaScript function to detect if two rectangles overlap.',
        'Create a throttle function in JavaScript.',
        'Write a JavaScript function that groups an array of objects by a key.',
        'Implement a basic router in vanilla JavaScript for single-page apps.',
    ],
    'general': [
        'Write a Python function to check if a string is a valid palindrome, ignoring non-alphanumeric characters.',
        'Build a simple React counter component with increment, decrement, and reset buttons.',
        'Write a JavaScript function that converts a number to Roman numerals.',
        'Create a Python script that generates a random password with configurable length and character types.',
        'Write a bash script that monitors a directory for new files and logs their names.',
        'Build a React component that fetches data from an API and displays it in a table.',
        'Write a Python function to find the longest common subsequence of two strings.',
        'Create a JavaScript class for a simple linked list with add, remove, and find methods.',
        'Write a Python function that validates an email address using regex.',
        'Build a React form that calculates BMI from height and weight inputs.',
    ],
}

def _get_prompts(focus: str) -> list:
    return PROMPT_TEMPLATES.get(focus, PROMPT_TEMPLATES['general'])

def _extract_code(response: str) -> tuple:
    """Extract code block and language from model response."""
    import re as _re
    # Match ```language\ncode\n```
    match = _re.search(r'```(\w+)?\n(.*?)```', response, _re.DOTALL)
    if match:
        lang = match.group(1) or 'python'
        code = match.group(2).strip()
        # Map common language names
        lang_map = {'tsx': 'javascript', 'ts': 'javascript', 'jsx': 'javascript', 'js': 'javascript', 'py': 'python', 'sh': 'bash'}
        lang = lang_map.get(lang, lang)
        return lang, code
    return None, None

def _auto_score_code(stdout: str, stderr: str, exit_code: int) -> int:
    """Auto-score based on code execution result."""
    if exit_code == 0 and not stderr:
        return 8  # Clean execution
    elif exit_code == 0 and stderr:
        return 6  # Runs but warnings
    elif 'SyntaxError' in stderr or 'TypeError' in stderr:
        return 2  # Basic errors
    else:
        return 3  # Runtime errors

# ── Auto Train endpoints ──
@app.post('/auto_train')
async def auto_train(req: AutoTrainRequest):
    global auto_train_active, auto_train_paused, auto_round, auto_total_rounds
    global auto_pass_count, auto_exchanges, auto_focus, auto_method
    if auto_train_active:
        raise HTTPException(400, 'Auto training already running')
    auto_train_active = True
    auto_train_paused = False
    auto_round = 0
    auto_total_rounds = req.rounds
    auto_pass_count = 0
    auto_exchanges = []
    auto_focus = req.focus
    auto_method = req.method
    # Start auto training in background
    import asyncio
    asyncio.create_task(_run_auto_train())
    return {'status': 'started', 'focus': req.focus, 'rounds': req.rounds}

async def _run_auto_train():
    global auto_train_active, auto_train_paused, auto_round, auto_pass_count, auto_exchanges, step
    import asyncio, random
    prompts = _get_prompts(auto_focus)
    for i in range(auto_total_rounds):
        if not auto_train_active:
            break
        while auto_train_paused:
            await asyncio.sleep(1)
            if not auto_train_active:
                return
        auto_round = i + 1
        prompt = prompts[i % len(prompts)] if i < len(prompts) else random.choice(prompts)
        # Get model response
        with model_lock:
            conversation.append({'role': 'user', 'content': prompt})
            p = _build_prompt(conversation)
            inputs = tokenizer(p, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id)
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            conversation.append({'role': 'assistant', 'content': response})
        # Try to run code
        lang, code = _extract_code(response)
        exec_result = None
        auto_score = 5
        reason = 'No code found in response'
        if lang and code:
            try:
                import subprocess, tempfile
                if lang == 'python':
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                        f.write(code); f.flush()
                        result = subprocess.run(['python', f.name], capture_output=True, text=True, timeout=30)
                elif lang in ('javascript',):
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                        f.write(code); f.flush()
                        result = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=30)
                elif lang == 'bash':
                    result = subprocess.run(['bash', '-c', code], capture_output=True, text=True, timeout=30)
                else:
                    result = None
                if result:
                    exec_result = {'stdout': result.stdout[:2000], 'stderr': result.stderr[:2000], 'exit_code': result.returncode}
                    auto_score = _auto_score_code(result.stdout, result.stderr, result.returncode)
                    if result.returncode == 0:
                        auto_pass_count += 1
                        reason = 'Code executed successfully'
                    else:
                        reason = f'Runtime error: {result.stderr[:200]}'
            except subprocess.TimeoutExpired:
                exec_result = {'stdout': '', 'stderr': 'Timeout', 'exit_code': 1}
                auto_score = 3
                reason = 'Code execution timed out'
            except Exception as e:
                exec_result = {'stdout': '', 'stderr': str(e), 'exit_code': 1}
                auto_score = 4
                reason = str(e)[:200]
        # Train on the score
        with model_lock:
            reward = _compute_reward(auto_score)
            _train_step(reward)
            scores_history.append({'score': auto_score, 'reward': reward, 'step': step})
        exchange = {
            'round': auto_round,
            'prompt': prompt[:200],
            'response': response[:500],
            'score': auto_score,
            'reason': reason,
            'code_executed': exec_result is not None,
            'passed': exec_result['exit_code'] == 0 if exec_result else False,
        }
        auto_exchanges.append(exchange)
        # Keep only last 50 exchanges in memory
        if len(auto_exchanges) > 50:
            auto_exchanges = auto_exchanges[-50:]
        await asyncio.sleep(0.5)  # Small delay between rounds
    auto_train_active = False

@app.get('/auto_status')
async def auto_status():
    avg = np.mean([s['score'] for s in scores_history[-20:]]) if scores_history else 0
    pass_rate = auto_pass_count / auto_round if auto_round > 0 else 0
    return {
        'running': auto_train_active,
        'paused': auto_train_paused,
        'round': auto_round,
        'total_rounds': auto_total_rounds,
        'avg_score': round(float(avg), 1),
        'pass_rate': round(pass_rate, 2),
        'focus': auto_focus,
        'method': auto_method,
        'exchanges': auto_exchanges[-5:],
        'step': step,
    }

@app.post('/auto_pause')
async def auto_pause():
    global auto_train_paused
    auto_train_paused = True
    return {'status': 'paused'}

@app.post('/auto_resume')
async def auto_resume_training():
    global auto_train_paused
    auto_train_paused = False
    return {'status': 'resumed'}

@app.post('/auto_stop')
async def auto_stop():
    global auto_train_active
    auto_train_active = False
    return {'status': 'stopped', 'rounds_completed': auto_round, 'step': step}

# ── History endpoint (for Auto Chat live feed) ──
@app.get('/history')
async def history():
    exchanges = []
    i = 0
    score_idx = 0
    while i < len(conversation) - 1:
        if conversation[i]['role'] == 'user' and i + 1 < len(conversation) and conversation[i + 1]['role'] == 'assistant':
            ex = {
                'prompt': conversation[i]['content'][:300],
                'response': conversation[i + 1]['content'][:500],
                'score': scores_history[score_idx]['score'] if score_idx < len(scores_history) else None,
                'step': scores_history[score_idx].get('step') if score_idx < len(scores_history) else None,
            }
            exchanges.append(ex)
            score_idx += 1
            i += 2
        else:
            i += 1
    return {'exchanges': exchanges[-20:]}

# ── MCP Server (for Auto Chat — AI judge connects via MCP protocol) ──
try:
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP('CFT-Judge-Server')

    @mcp.tool()
    async def cft_chat(message: str) -> str:
        """Send a message to the training model and get its response."""
        result = await chat(ChatRequest(message=message))
        return json.dumps(result)

    @mcp.tool()
    async def cft_run_code(language: str, code: str) -> str:
        """Execute python, javascript, or bash code and return the output."""
        result = await run_code(RunCodeRequest(language=language, code=code))
        return f"STDOUT: {result['stdout']}\nSTDERR: {result['stderr']}\nEXIT_CODE: {result['exit_code']}"

    @mcp.tool()
    async def cft_score(score_value: int) -> str:
        """Score the last model response 1-10. Triggers a LoRA training step."""
        result = await score(ScoreRequest(score=score_value))
        return json.dumps(result)

    @mcp.tool()
    async def cft_correct(correction: str) -> str:
        """Provide the correct answer. Trains the model on it as a positive example."""
        result = await correct(CorrectRequest(correction=correction))
        return json.dumps(result)

    @mcp.tool()
    async def cft_status() -> str:
        """Get training stats: step count, avg score, total messages."""
        result = await status()
        return json.dumps(result)

    @mcp.tool()
    async def cft_stop() -> str:
        """Stop training and save the LoRA adapter checkpoint."""
        result = await stop_training()
        return json.dumps(result)

    @mcp.tool()
    async def cft_upload(repo_name: str) -> str:
        """Upload the saved adapter to HuggingFace."""
        result = await upload(UploadRequest(repo_name=repo_name))
        return json.dumps(result)

    # Mount MCP onto FastAPI — creates /mcp/sse and /mcp/messages/ automatically
    mcp.mount_to(app)
    print('MCP server enabled (FastMCP mounted at /mcp/sse)')
except ImportError:
    print('MCP SDK not installed — MCP endpoints disabled. Install with: pip install "mcp[server]"')

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
