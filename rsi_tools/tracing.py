"""Runtime-only instrumentation around author client/cache; no source edits."""
from __future__ import annotations
import dataclasses
import inspect
import json
import time
from types import SimpleNamespace
import urllib.parse
import urllib.request
from .common import append, digest, now

class BudgetStop(RuntimeError):
    pass

def local_meta(base_url):
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme != 'http' or parsed.hostname not in ['127.0.0.1', 'localhost'] or parsed.username or parsed.password:
        raise ValueError('This pilot only calls a loopback HTTP model server')
    url = urllib.parse.urlunparse((parsed.scheme, parsed.netloc, '/rsi/meta', '', '', ''))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=10) as f:
        return json.load(f)

def module_role():
    for f in inspect.stack(context=0):
        if f.function in ['run_planner', 'run_selector', 'run_caller', 'run_synthesizer', 'blame', 'mutate', 'monolithic_mutate']:
            return f.function
    return 'unknown'

def make_client(cfg, out, limits, *, create_override=None, server_meta=None):
    from src.llm.client import LLMClient
    class TraceClient(LLMClient):
        def __init__(self):
            self.config = cfg
            self.total_tokens = 0
            self.call_records = []
            self.calls = 0
            self.used_tokens = 0
            self.deadline = time.monotonic() + limits['max_minutes'] * 60
            self.context = {}
            self.server_meta = server_meta or local_meta(cfg.base_url)
            self.context_limit = int(self.server_meta['meta']['max_context'])
            if create_override is None:
                from openai import OpenAI
                import httpx
                sdk = OpenAI(base_url=cfg.base_url, api_key='EMPTY', max_retries=0,
                             timeout=limits.get('request_timeout_seconds', 150),
                             http_client=httpx.Client(trust_env=False))
                create = sdk.chat.completions.create
            else:
                create = create_override
            def recorded_create(**kwargs):
                if (time.monotonic() >= self.deadline or self.calls >= limits['max_calls']
                        or self.used_tokens + self.context_limit > limits['max_total_tokens']):
                    raise BudgetStop('Pilot call/time/token budget reached; partial run retained')
                self.calls += 1
                started = time.monotonic()
                event = {'event': 'model_call', 'call_index': self.calls, 'time': now(),
                         'role': module_role(), 'episode_context': dict(self.context),
                         'request': kwargs, 'request_sha256': digest(kwargs),
                         'model': cfg.model_name, 'server_revision': self.server_meta['meta'].get('revision')}
                try:
                    response = create(**kwargs)
                    data = response.model_dump(mode='json')
                    usage = data.get('usage')
                    if usage is None or any(usage.get(k) is None for k in ['prompt_tokens', 'completion_tokens', 'total_tokens']):
                        raise RuntimeError('Missing actual token usage; cannot enforce budget')
                    self.used_tokens += int(usage['total_tokens'])
                    event['response'] = data
                    event['usage'] = usage
                    event['output_truncated'] = any(c.get('finish_reason') in ['length', 'time_limit'] for c in data.get('choices', []))
                    return response
                except Exception as exc:
                    event['error_type'] = type(exc).__name__
                    event['error'] = str(exc)[:1000]
                    event['usage'] = None
                    raise
                finally:
                    event['elapsed_seconds'] = time.monotonic() - started
                    self.call_records.append(event)
                    append(out / 'model-calls.jsonl', event)
            self.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=recorded_create)))
    return TraceClient()

class EpisodeCache(dict):
    """Retains original per-run memoization; logs cache hits without new inference."""
    def __init__(self, client, out, max_steps, run_kind):
        super().__init__()
        self.client, self.out, self.max_steps, self.run_kind = client, out, max_steps, run_kind
        self.start = 0
        self.policy_id = None
        self.episode_count = 0
        self.cache_hits = 0

    def __contains__(self, key):
        hit = super().__contains__(key)
        if not hit:
            self.start = self.client.calls
            self.client.context = {'task_id': str(key[1]), 'policy_key': str(key[0])}
            for frame in inspect.stack(context=0):
                policy = frame.frame.f_locals.get('policy')
                if policy is not None and hasattr(policy, 'policy_id'):
                    self.policy_id = policy.policy_id
                    break
        return hit

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.cache_hits += 1
        append(self.out / 'episodes.jsonl', {'event': 'episode_cache_hit', 'task_id': str(key[1]),
                                           'policy_key': str(key[0]), 'max_steps': self.max_steps})
        return value

    def __setitem__(self, key, value):
        record = dataclasses.asdict(value)
        calls = self.client.call_records[self.start:self.client.calls]
        record.update(event='episode_complete', policy_id=self.policy_id,
                      policy_key=str(key[0]), max_steps=self.max_steps,
                      call_indices=[x['call_index'] for x in calls],
                      h1_generation_truncated=any(x.get('output_truncated') for x in calls),
                      h1_context_overflow_attempt=any('context length' in x.get('error', '') for x in calls),
                      run_kind=self.run_kind)
        append(self.out / 'episodes.jsonl', record)
        self.episode_count += 1
        super().__setitem__(key, value)

class GenerationLog:
    def __init__(self, out):
        self.out = out

    def log_generation(self, **kwargs):
        def encode(x):
            if dataclasses.is_dataclass(x):
                return dataclasses.asdict(x)
            if isinstance(x, dict):
                return {k: encode(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [encode(v) for v in x]
            return x
        append(self.out / 'generations.jsonl', encode(kwargs))
