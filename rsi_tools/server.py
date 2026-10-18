"""Loopback-only Transformers adapter for V100. No vLLM, BF16 or FlashAttention2."""
from __future__ import annotations
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import pathlib
import os
import threading
import time
import uuid
from .common import MODEL, REVISION, append, environment, new_run_dir, now, read, sha, write_new

class HFBackend:
    def __init__(self, model_dir, args):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        self.torch = torch
        self.parser = JsonSchemaParser
        self.prefix_fn = build_transformers_prefix_allowed_tokens_fn
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; CPU/offload fallback is intentionally disabled')
        receipt = read(pathlib.Path(model_dir) / 'rsi-model-manifest.json')
        if receipt['repo_id'] != MODEL or receipt['revision'] != REVISION:
            raise RuntimeError('Unexpected model revision')
        for name, expected in receipt['files'].items():
            if sha(pathlib.Path(model_dir) / name) != expected:
                raise RuntimeError('Changed model file: ' + name)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.float16, attn_implementation='sdpa', low_cpu_mem_usage=True)
        self.model.to('cuda:0').eval()
        self.meta = {'model': MODEL, 'revision': REVISION, 'served_name': 'Qwen3-4B',
                     'dtype': 'float16', 'attention': 'sdpa', 'enable_thinking': False,
                     'json_mode': 'lm-format-enforcer JSON-object grammar; truncated output may be invalid',
                     'max_context': args.max_context, 'max_new_tokens': args.max_new_tokens,
                     'max_request_seconds': args.max_request_seconds,
                     'model_manifest_sha256': sha(pathlib.Path(model_dir) / 'rsi-model-manifest.json'),
                     'environment': environment()}

    def generate(self, body, remaining_tokens):
        torch = self.torch
        messages = body.get('messages')
        if not isinstance(messages, list) or not messages or any(
                not isinstance(m, dict) or not isinstance(m.get('content'), str)
                or m.get('role') not in ['system', 'user', 'assistant'] for m in messages):
            raise ValueError('Only nonempty text chat messages are supported')
        if body.get('stream'):
            raise ValueError('Streaming is not supported by this pilot adapter')
        if body.get('temperature', 0.0) != 0.0:
            raise ValueError('This frozen pilot only supports temperature=0')
        if body.get('chat_template_kwargs', {}).get('enable_thinking', False):
            raise ValueError('Frozen pilot requires enable_thinking=False')
        if body.get('model') not in ['Qwen3-4B', MODEL]:
            raise ValueError('Unexpected served model name')
        n = int(body.get('max_tokens', self.meta['max_new_tokens']))
        if not 1 <= n <= self.meta['max_new_tokens']:
            raise ValueError('max_tokens exceeds the frozen limit')
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        prompt_tokens = inputs['input_ids'].shape[1]
        if prompt_tokens + n > self.meta['max_context']:
            raise ValueError('maximum context length exceeded; input is NOT silently truncated')
        if prompt_tokens + n > remaining_tokens:
            raise ValueError('Total server token budget would be exceeded')
        kwargs = {'max_new_tokens': n, 'do_sample': False,
                  'pad_token_id': self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                  'max_time': self.meta['max_request_seconds'], 'use_cache': True}
        fmt = body.get('response_format', {}).get('type', 'text')
        if fmt == 'json_object':
            parser = self.parser({'type': 'object'})
            kwargs['prefix_allowed_tokens_fn'] = self.prefix_fn(self.tokenizer, parser)
        elif fmt != 'text':
            raise ValueError('Only text and json_object response formats are supported')
        inputs = {k: v.to('cuda:0') for k, v in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        token_ids = output[0, prompt_tokens:].tolist()
        content = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        elapsed = time.monotonic() - started
        eos = self.model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [eos]
        cause = 'stop' if token_ids and token_ids[-1] in eos_ids else ('length' if len(token_ids) >= n else 'time_limit')
        finish = 'stop' if cause == 'stop' else 'length'
        usage = {'prompt_tokens': prompt_tokens, 'completion_tokens': len(token_ids), 'total_tokens': prompt_tokens + len(token_ids)}
        extras = {'elapsed_seconds': elapsed, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                  'json_parse_ok': None, 'finish_reason': finish, 'stopping_cause': cause}
        if fmt == 'json_object':
            try:
                extras['json_parse_ok'] = isinstance(json.loads(content), dict)
            except json.JSONDecodeError:
                extras['json_parse_ok'] = False
        return content, finish, usage, extras

def handler_factory(backend, out, limits):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in ['/health', '/rsi/meta']:
                self.send_json(200, {'status': 'ready', 'meta': backend.meta,
                                     'used_requests': limits['used_requests'], 'used_tokens': limits['used_tokens']})
            elif self.path == '/v1/models':
                self.send_json(200, {'object': 'list', 'data': [{'id': 'Qwen3-4B', 'object': 'model', 'owned_by': 'local'}]})
            else:
                self.send_json(404, {'error': {'message': 'Unknown endpoint'}})

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.send_json(404, {'error': {'message': 'Unknown endpoint'}})
            request_id = 'chatcmpl-' + uuid.uuid4().hex
            event = {'request_id': request_id, 'time': now()}
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 1024 * 1024:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(size))
                if time.monotonic() > limits['deadline'] or limits['used_requests'] >= limits['max_requests']:
                    return self.send_json(429, {'error': {'message': 'Server pilot budget exhausted'}})
                limits['used_requests'] += 1
                event['request'] = body
                content, finish, usage, extra = backend.generate(body, limits['max_tokens'] - limits['used_tokens'])
                limits['used_tokens'] += usage['total_tokens']
                response = {'id': request_id, 'object': 'chat.completion', 'created': int(time.time()),
                            'model': 'Qwen3-4B', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': finish}],
                            'usage': usage, 'rsi_diagnostic': extra}
                event.update(response=response, diagnostic=extra)
                append(out / 'server-calls.jsonl', event)
                self.send_json(200, response)
            except ValueError as exc:
                event['error'] = str(exc)
                append(out / 'server-calls.jsonl', event)
                self.send_json(400, {'error': {'message': str(exc), 'type': 'invalid_request_error'}})
            except Exception as exc:
                event['error_type'] = type(exc).__name__
                event['error'] = str(exc)[:1000]
                append(out / 'server-calls.jsonl', event)
                self.send_json(500, {'error': {'message': event['error'], 'type': type(exc).__name__}})
                # Stop on OOM/runtime errors rather than altering dtype/context behind the user's back.
                threading.Thread(target=self.server.shutdown, daemon=True).start()
    return Handler

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--port', type=int, default=8000)
    p.add_argument('--max-context', type=int, default=8192)
    p.add_argument('--max-new-tokens', type=int, default=512)
    p.add_argument('--max-request-seconds', type=float, default=120)
    p.add_argument('--max-requests', type=int, default=180)
    p.add_argument('--max-total-tokens', type=int, default=160000)
    p.add_argument('--max-minutes', type=float, default=30)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if not 1 <= a.max_context <= 8192 or not 1 <= a.max_new_tokens <= 512 or a.max_minutes <= 0:
        p.error('Pilot limits: context<=8192, output<=512, positive runtime')
    out = new_run_dir(a.out)
    write_new(out / 'requested-settings.json', vars(a))
    deadline = time.monotonic() + a.max_minutes * 60
    def hard_stop():
        append(out / 'server-events.jsonl', {'time': now(), 'event': 'hard_wall_limit', 'max_minutes': a.max_minutes})
        os._exit(124)
    # Includes model loading and generation; HF max_time alone is not a hard wall limit.
    watchdog = threading.Timer(a.max_minutes * 60, hard_stop)
    watchdog.daemon = True
    watchdog.start()
    backend = HFBackend(a.model_dir, a)
    write_new(out / 'server-metadata.json', backend.meta)
    limits = {'deadline': deadline, 'max_requests': a.max_requests, 'max_tokens': a.max_total_tokens,
              'used_requests': 0, 'used_tokens': 0}
    server = HTTPServer(('127.0.0.1', a.port), handler_factory(backend, out, limits))
    timer = threading.Timer(max(0.01, deadline - time.monotonic()), server.shutdown)
    timer.daemon = True
    timer.start()
    print('READY http://127.0.0.1:' + str(a.port), flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        watchdog.cancel()
        timer.cancel()
        server.server_close()
        write_new(out / 'server-summary.json', {k: v for k, v in limits.items() if k != 'deadline'})

if __name__ == '__main__':
    main()
