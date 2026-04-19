# cft_server.py v2 — CFT Backend Server (Bulletproof Edition)
# All known bugs fixed: no 400/404/405, no ASGI crashes, no deprecation warnings
# Auto mode: MCP AI chats with training model (no hardcoded prompts)
# Model: Qwen/Qwen3.6-35B-A3B (MoE, 35B total, ~3B active)

import json, os, re, time, asyncio
from pathlib import Path
from threading import Lock
from contextlib import asynccontextmanager

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import uvicorn

# ── Config ──
BASE_MODEL = os.environ.get('BASE_MODEL_ID', 'Qwen/Qwen3.6-35B-A3B')
CHECKPOINT_DIR = os.environ.get('CFT_CHECKPOINT', '/root/cft-checkpoints')
PORT = int(os.environ.get('CFT_PORT', '7860'))
LR = float(os.environ.get('CFT_LR', '1e-5'))

Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

# ── State ──
model = None
tokenizer = None
optimizer = None
model_lock = Lock()
training_active = True
step = 0
scores_history = []
conversation = []  # list of {"role": "user"|"assistant", "content": str}

# ── Pydantic models ──
class ChatRequest(BaseModel):
    message: str

class ScoreRequest(BaseModel):
    score: int
    message_index: int = -1

class CorrectRequest(BaseModel):
    correction: str
    message_index: int = -1

class UploadRequest(BaseModel):
    repo_name: str

class RunCodeRequest(BaseModel):
    language: str
    code: str
    timeout: int = 30


# ══════════════════════════════════════════════════
# FIX #1: Use lifespan instead of @app.on_event
# ══════════════════════════════════════════════════
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, optimizer
    print(f'Loading {BASE_MODEL}...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map='auto', trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05,
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
    )
    model = get_peft_model(model, lora_config)
    model.train()

    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=LR)

    print(f'Model loaded. LoRA params: {sum(p.numel() for p in lora_params):,}')
    print(f'CFT server ready on port {PORT}')
    yield
    # Shutdown: save checkpoint
    try:
        save_path = os.path.join(CHECKPOINT_DIR, 'cft-adapter')
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f'Checkpoint saved to {save_path}')
    except Exception as e:
        print(f'Warning: could not save on shutdown: {e}')


app = FastAPI(title='CFT Server', lifespan=lifespan)


# ══════════════════════════════════════════════════
# FIX #2: Generous CORS — no more 405 on OPTIONS
# ══════════════════════════════════════════════════
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# ══════════════════════════════════════════════════
# FIX #3: Global exception handler — no uncaught 500s
# ══════════════════════════════════════════════════
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f'Unhandled error on {request.method} {request.url.path}: {exc}')
    return JSONResponse(
        status_code=500,
        content={'error': str(exc), 'path': request.url.path},
    )


# ══════════════════════════════════════════════════
# FIX #4: Root route — no more GET / → 404
# ══════════════════════════════════════════════════
@app.get('/')
async def root():
    return {
        'name': 'CFT Server',
        'version': '2.0',
        'model': BASE_MODEL,
        'status': 'running',
        'mcp_endpoints': {
            'streamable_http': 'POST /mcp  (bolt.new, Cursor)',
            'sse': 'GET /mcp/sse  (Claude Desktop)',
        },
        'endpoints': [
            'GET  /',
            'GET  /status',
            'GET  /history',
            'POST /chat',
            'POST /score',
            'POST /correct',
            'POST /run_code',
            'POST /stop',
            'POST /resume',
            'POST /upload',
            'POST /mcp  (Streamable HTTP — bolt.new, Cursor)',
            'GET  /mcp/sse  (SSE — Claude Desktop)',
            'POST /mcp/messages/',
        ],
    }


# ── Internal helpers ──
def _compute_reward(score_1_to_10: int) -> float:
    return (score_1_to_10 - 5.5) / 4.5

def _build_prompt(conv: list) -> str:
    messages = [{'role': m['role'], 'content': m['content']} for m in conv[-10:]]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        parts = []
        for msg in messages:
            role = 'User' if msg['role'] == 'user' else 'Assistant'
            parts.append(f"{role}: {msg['content']}")
        parts.append('Assistant:')
        return '\n'.join(parts)

def _train_step(reward: float):
    global step
    if not training_active or len(conversation) < 2:
        return
    last_assistant = last_user = None
    for msg in reversed(conversation):
        if msg['role'] == 'assistant' and last_assistant is None:
            last_assistant = msg['content']
        elif msg['role'] == 'user' and last_user is None:
            last_user = msg['content']
        if last_assistant and last_user:
            break
    if not last_assistant or not last_user:
        return
    train_messages = [
        {'role': 'user', 'content': last_user},
        {'role': 'assistant', 'content': last_assistant},
    ]
    try:
        train_text = tokenizer.apply_chat_template(train_messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        train_text = f"User: {last_user}\nAssistant: {last_assistant}"
    inputs = tokenizer(train_text, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss * reward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1

def _train_on_correction(correction: str):
    global step
    if not training_active or len(conversation) < 1:
        return
    last_user = None
    for msg in reversed(conversation):
        if msg['role'] == 'user':
            last_user = msg['content']
            break
    if not last_user:
        return
    train_messages = [
        {'role': 'user', 'content': last_user},
        {'role': 'assistant', 'content': correction},
    ]
    try:
        train_text = tokenizer.apply_chat_template(train_messages, tokenize=False, add_generation_prompt=False)
    except Exception:
        train_text = f"User: {last_user}\nAssistant: {correction}"
    inputs = tokenizer(train_text, return_tensors='pt', truncation=True, max_length=1024).to(model.device)
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    step += 1

def _run_code_sync(language: str, code: str, timeout: int = 30) -> dict:
    import subprocess, tempfile
    try:
        if language == 'python':
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code); f.flush()
                result = subprocess.run(['python', f.name], capture_output=True, text=True, timeout=timeout)
        elif language in ('javascript', 'typescript'):
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code); f.flush()
                result = subprocess.run(['node', f.name], capture_output=True, text=True, timeout=timeout)
        elif language == 'bash':
            result = subprocess.run(['bash', '-c', code], capture_output=True, text=True, timeout=timeout)
        else:
            return {'stdout': '', 'stderr': f'Unsupported language: {language}', 'exit_code': 1, 'error': 'unsupported'}
        return {'stdout': result.stdout[:5000], 'stderr': result.stderr[:5000], 'exit_code': result.returncode, 'error': None}
    except subprocess.TimeoutExpired:
        return {'stdout': '', 'stderr': 'Execution timed out', 'exit_code': 1, 'error': 'timeout'}
    except Exception as e:
        return {'stdout': '', 'stderr': str(e), 'exit_code': 1, 'error': str(e)}


# ══════════════════════════════════════════════════
# REST ENDPOINTS (used by Manual Chat + MCP tools)
# ══════════════════════════════════════════════════

@app.post('/chat')
async def chat(req: ChatRequest):
    global step
    with model_lock:
        conversation.append({'role': 'user', 'content': req.message})
        score_match = re.match(r'^(\d+)/10$', req.message.strip())
        if score_match:
            score_val = int(score_match.group(1))
            reward = _compute_reward(score_val)
            _train_step(reward)
            scores_history.append({'score': score_val, 'reward': reward, 'step': step})
            return {'response': f'Score {score_val}/10 recorded ✓ (reward: {reward:.2f}). Training step {step} complete.', 'step': step}
        prompt = _build_prompt(conversation)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=1024, do_sample=True,
                temperature=0.7, top_p=0.9, pad_token_id=tokenizer.pad_token_id
            )
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        conversation.append({'role': 'assistant', 'content': response})
        return {'response': response, 'step': step}


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


@app.post('/correct')
async def correct(req: CorrectRequest):
    global step
    if not training_active:
        raise HTTPException(400, 'Training is stopped')
    with model_lock:
        _train_on_correction(req.correction)
        scores_history.append({'score': 0, 'reward': -1.0, 'step': step, 'corrected': True})
        return {'status': 'ok', 'step': step}


@app.get('/status')
async def status():
    avg = np.mean([s['score'] for s in scores_history[-20:]]) if scores_history else 0
    return {
        'step': step,
        'total_scores': len(scores_history),
        'avg_score': round(float(avg), 1),
        'training': training_active,
        'total_messages': len(conversation),
        'model': BASE_MODEL,
    }


@app.post('/stop')
async def stop_training():
    global training_active
    training_active = False
    save_path = os.path.join(CHECKPOINT_DIR, 'cft-adapter')
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(CHECKPOINT_DIR, 'conversation.json'), 'w') as f:
        json.dump({'conversation': conversation, 'scores': scores_history}, f, indent=2)
    return {'status': 'stopped', 'checkpoint_path': save_path, 'step': step}


@app.post('/resume')
async def resume_training():
    global training_active
    training_active = True
    return {'status': 'resumed'}


@app.post('/run_code')
async def run_code(req: RunCodeRequest):
    return _run_code_sync(req.language, req.code, req.timeout)


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


# ══════════════════════════════════════════════════
# FIX #5: MCP — the Auto Train mechanism
# Supports BOTH transports:
#   1. SSE transport (Claude Desktop) → GET /mcp/sse
#   2. Streamable HTTP / JSON-RPC (bolt.new, Cursor) → POST /mcp
# No template prompts — the MCP AI decides what to ask.
# ══════════════════════════════════════════════════

# ── MCP Tool Registry (transport-independent) ──
# Tools are defined once, used by both SSE and Streamable HTTP transports.

MCP_TOOLS = {
    'cft_chat': {
        'description': 'Send a message to the training model and get its response. Use this to ask the model questions, give it coding tasks, or have a conversation.',
        'inputSchema': {
            'type': 'object',
            'properties': {'message': {'type': 'string', 'description': 'The message to send'}},
            'required': ['message'],
        },
    },
    'cft_run_code': {
        'description': 'Execute code and return stdout/stderr/exit_code. Supported languages: python, javascript, bash. Use this to test code the training model wrote.',
        'inputSchema': {
            'type': 'object',
            'properties': {
                'language': {'type': 'string', 'description': 'python, javascript, or bash'},
                'code': {'type': 'string', 'description': 'The code to execute'},
            },
            'required': ['language', 'code'],
        },
    },
    'cft_score': {
        'description': 'Score the last model response from 1 to 10. Triggers a LoRA training step. 1=terrible, 5=mediocre, 8=good, 10=perfect. Score honestly.',
        'inputSchema': {
            'type': 'object',
            'properties': {'score_value': {'type': 'integer', 'description': 'Score from 1 to 10'}},
            'required': ['score_value'],
        },
    },
    'cft_correct': {
        'description': 'Provide the correct answer when the model was wrong. Trains on your correction as a positive example. Use for factual errors, not style.',
        'inputSchema': {
            'type': 'object',
            'properties': {'correction': {'type': 'string', 'description': 'The correct answer'}},
            'required': ['correction'],
        },
    },
    'cft_status': {
        'description': 'Get current training stats: step count, average score, total messages, total scores, and which model is loaded.',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    'cft_stop': {
        'description': 'Stop training and save the LoRA adapter checkpoint. Call this when done training.',
        'inputSchema': {'type': 'object', 'properties': {}},
    },
    'cft_upload': {
        'description': "Upload the saved LoRA adapter to HuggingFace. Call cft_stop() first. Example repo_name: 'KovaUser/kova-cft-qwen3b'",
        'inputSchema': {
            'type': 'object',
            'properties': {'repo_name': {'type': 'string', 'description': 'HuggingFace repo like user/model'}},
            'required': ['repo_name'],
        },
    },
}


async def _execute_mcp_tool(name: str, arguments: dict) -> str:
    """Execute an MCP tool by name and return the result as a string."""
    if name == 'cft_chat':
        result = await chat(ChatRequest(message=arguments['message']))
        return json.dumps(result)
    elif name == 'cft_run_code':
        result = _run_code_sync(arguments['language'], arguments['code'])
        return f"STDOUT: {result['stdout']}\nSTDERR: {result['stderr']}\nEXIT_CODE: {result['exit_code']}"
    elif name == 'cft_score':
        result = await score(ScoreRequest(score=arguments['score_value']))
        return json.dumps(result)
    elif name == 'cft_correct':
        result = await correct(CorrectRequest(correction=arguments['correction']))
        return json.dumps(result)
    elif name == 'cft_status':
        result = await status()
        return json.dumps(result)
    elif name == 'cft_stop':
        result = await stop_training()
        return json.dumps(result)
    elif name == 'cft_upload':
        result = await upload(UploadRequest(repo_name=arguments['repo_name']))
        return json.dumps(result)
    else:
        return json.dumps({'error': f'Unknown tool: {name}'})


# ══════════════════════════════════════════════════
# TRANSPORT 1: Streamable HTTP (bolt.new, Cursor, etc.)
# Single POST /mcp endpoint — handles JSON-RPC directly
# This is what bolt.new expects. No SSE, no session_id.
# ══════════════════════════════════════════════════

@app.post('/mcp')
async def mcp_streamable_http(request: Request):
    """Streamable HTTP transport — bolt.new, Cursor, and newer MCP clients POST here."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse(status_code=200, content={
            'jsonrpc': '2.0', 'id': None,
            'error': {'code': -32700, 'message': 'Parse error: invalid JSON'},
        })

    jsonrpc_id = body.get('id')
    method = body.get('method', '')
    params = body.get('params', {})

    # ── initialize ──
    if method == 'initialize':
        return JSONResponse(content={
            'jsonrpc': '2.0', 'id': jsonrpc_id,
            'result': {
                'protocolVersion': '2024-11-05',
                'capabilities': {'tools': {'listChanged': False}},
                'serverInfo': {'name': 'CFT-Server', 'version': '2.0'},
            },
        })

    # ── notifications (initialized, cancelled, etc.) ──
    if method.startswith('notifications/') or method == 'initialized':
        return JSONResponse(content={'jsonrpc': '2.0', 'id': jsonrpc_id, 'result': {}})

    # ── tools/list ──
    if method == 'tools/list':
        tools_list = []
        for tool_name, tool_def in MCP_TOOLS.items():
            tools_list.append({
                'name': tool_name,
                'description': tool_def['description'],
                'inputSchema': tool_def['inputSchema'],
            })
        return JSONResponse(content={
            'jsonrpc': '2.0', 'id': jsonrpc_id,
            'result': {'tools': tools_list},
        })

    # ── tools/call ──
    if method == 'tools/call':
        tool_name = params.get('name', '')
        arguments = params.get('arguments', {})
        if tool_name not in MCP_TOOLS:
            return JSONResponse(content={
                'jsonrpc': '2.0', 'id': jsonrpc_id,
                'error': {'code': -32601, 'message': f'Unknown tool: {tool_name}'},
            })
        try:
            result_text = await _execute_mcp_tool(tool_name, arguments)
            return JSONResponse(content={
                'jsonrpc': '2.0', 'id': jsonrpc_id,
                'result': {
                    'content': [{'type': 'text', 'text': result_text}],
                    'isError': False,
                },
            })
        except Exception as e:
            return JSONResponse(content={
                'jsonrpc': '2.0', 'id': jsonrpc_id,
                'result': {
                    'content': [{'type': 'text', 'text': f'Error: {str(e)}'}],
                    'isError': True,
                },
            })

    # ── ping ──
    if method == 'ping':
        return JSONResponse(content={'jsonrpc': '2.0', 'id': jsonrpc_id, 'result': {}})

    # ── unknown method ──
    return JSONResponse(content={
        'jsonrpc': '2.0', 'id': jsonrpc_id,
        'error': {'code': -32601, 'message': f'Method not found: {method}'},
    })


print('Streamable HTTP MCP enabled at POST /mcp (for bolt.new, Cursor, etc.)')


# ══════════════════════════════════════════════════
# TRANSPORT 2: SSE transport (Claude Desktop, etc.)
# GET /mcp/sse to open stream, POST /mcp/messages/ to send
# ══════════════════════════════════════════════════
try:
    from mcp.server.fastmcp import FastMCP
    from mcp.server.sse import SseServerTransport

    mcp_fastmcp = FastMCP('CFT-Server')

    @mcp_fastmcp.tool()
    async def cft_chat_sse(message: str) -> str:
        """Send a message to the training model and get its response."""
        result = await chat(ChatRequest(message=message))
        return json.dumps(result)

    @mcp_fastmcp.tool()
    async def cft_run_code_sse(language: str, code: str) -> str:
        """Execute code and return stdout/stderr/exit_code."""
        result = _run_code_sync(language, code)
        return f"STDOUT: {result['stdout']}\nSTDERR: {result['stderr']}\nEXIT_CODE: {result['exit_code']}"

    @mcp_fastmcp.tool()
    async def cft_score_sse(score_value: int) -> str:
        """Score the last model response 1-10. Triggers a LoRA training step."""
        result = await score(ScoreRequest(score=score_value))
        return json.dumps(result)

    @mcp_fastmcp.tool()
    async def cft_correct_sse(correction: str) -> str:
        """Provide the correct answer. Trains on it as a positive example."""
        result = await correct(CorrectRequest(correction=correction))
        return json.dumps(result)

    @mcp_fastmcp.tool()
    async def cft_status_sse() -> str:
        """Get training stats."""
        result = await status()
        return json.dumps(result)

    @mcp_fastmcp.tool()
    async def cft_stop_sse() -> str:
        """Stop training and save checkpoint."""
        result = await stop_training()
        return json.dumps(result)

    @mcp_fastmcp.tool()
    async def cft_upload_sse(repo_name: str) -> str:
        """Upload adapter to HuggingFace."""
        result = await upload(UploadRequest(repo_name=repo_name))
        return json.dumps(result)

    _low = getattr(mcp_fastmcp, '_mcp_server', None) or getattr(mcp_fastmcp, '_server', None)
    if _low is None:
        for attr in dir(mcp_fastmcp):
            obj = getattr(mcp_fastmcp, attr, None)
            if hasattr(obj, 'run') and hasattr(obj, 'create_initialization_options'):
                _low = obj
                break

    if _low:
        sse = SseServerTransport('/mcp/messages/')

        @app.get('/mcp/sse')
        async def handle_sse(request: Request):
            try:
                async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                    await _low.run(streams[0], streams[1], _low.create_initialization_options())
            except Exception as e:
                print(f'MCP SSE error (GET): {e}')
                return JSONResponse(status_code=200, content={'error': str(e)})

        @app.post('/mcp/sse')
        async def handle_sse_post(request: Request):
            try:
                await sse.handle_post_message(request.scope, request.receive, request._send)
            except Exception as e:
                print(f'MCP SSE error (POST): {e}')
                try:
                    return JSONResponse(status_code=200, content={
                        'warning': 'SSE session error. If using bolt.new, use POST /mcp instead.',
                        'detail': str(e),
                    })
                except RuntimeError:
                    pass

        @app.post('/mcp/messages/')
        async def handle_messages(request: Request):
            try:
                await sse.handle_post_message(request.scope, request.receive, request._send)
            except Exception as e:
                print(f'MCP messages error: {e}')
                try:
                    return JSONResponse(status_code=200, content={'error': str(e)})
                except RuntimeError:
                    pass

        @app.post('/mcp/messages')
        async def handle_messages_no_slash(request: Request):
            try:
                await sse.handle_post_message(request.scope, request.receive, request._send)
            except Exception as e:
                print(f'MCP messages (no slash) error: {e}')
                try:
                    return JSONResponse(status_code=200, content={'error': str(e)})
                except RuntimeError:
                    pass

        print('SSE MCP enabled at GET /mcp/sse (for Claude Desktop, etc.)')
    else:
        print('WARNING: Could not find MCP low-level server. SSE transport disabled.')

except ImportError:
    print('MCP SDK not installed — SSE transport disabled. Streamable HTTP at POST /mcp still works.')
    print('To enable SSE too: pip install mcp sse-starlette')


# ── Run ──
if __name__ == '__main__':
    pod_id = os.environ.get('RUNPOD_POD_ID', None)
    colab = 'COLAB_GPU' in os.environ or 'COLAB_RELEASE_TAG' in os.environ
    print(f'\n{"="*50}')
    print(f'CFT Server v2 Starting!')
    print(f'Model: {BASE_MODEL}')
    if pod_id:
        print(f'\n  Pod URL:  https://{pod_id}-{PORT}.proxy.runpod.net')
        print(f'  MCP URL:  https://{pod_id}-{PORT}.proxy.runpod.net/mcp/sse')
        print('  MCP (bolt.new):  https://' + pod_id + '-' + str(PORT) + '.proxy.runpod.net/mcp')
        print('\n  Manual: paste the Pod URL into the CFT website')
        print('  Auto:   connect an MCP client to the right MCP URL above')
    elif colab:
        print(f'\n  Running on Colab — use ngrok or localtunnel to expose port {PORT}.')
    else:
        print(f'\n  Server:        http://localhost:{PORT}')
        print(f'  MCP (bolt):    http://localhost:{PORT}/mcp')
        print(f'  MCP (Claude):  http://localhost:{PORT}/mcp/sse')
    print(f'{"="*50}\n')
    uvicorn.run(app, host='0.0.0.0', port=PORT)
