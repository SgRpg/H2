"""Bounded CPU contracts. Synthetic outputs never count as research results."""
import contextlib
import copy
from http.server import HTTPServer
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from rsi_tools.common import activate_vendor, read, sha
activate_vendor()
from rsi_tools.analysis import matched_gold_calls
from rsi_tools.data import developer_subset, validate_split
from rsi_tools.server import handler_factory
from rsi_tools.tracing import BudgetStop, local_meta, make_client
from src.config import LLMConfig
from src.eval_bfcl_ast import bfcl_ast_success
from src.metrics import progress
from pilot import execute

META = {'meta': {'max_context': 8192, 'revision': '1cfa9a7208912126459214e8b04321603b3df60c',
                 'dtype': 'float16', 'enable_thinking': False}}

class Response:
    def __init__(self, content):
        from types import SimpleNamespace
        self.content = content
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self.usage = SimpleNamespace(total_tokens=30)

    def model_dump(self, **kwargs):
        return {'choices': [{'message': {'content': self.content}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 20, 'completion_tokens': 10, 'total_tokens': 30}}

def scripted_create(**kwargs):
    s = kwargs['messages'][0]['content']
    if 'diagnostic judge' in s:
        value = {'primary': 'caller', 'diagnosis': 'Synthetic CPU contract output'}
    elif 'targeted prompt editor' in s:
        value = {'target_module': 'caller', 'revised_spec': 'Synthetic CPU contract caller, not a learned research result'}
    elif '"plan"' in s:
        value = {'plan': ['Synthetic abstract subgoal']}
    elif '"tool"' in s:
        value = {'tool': 'NONE'}
    elif '"arguments"' in s:
        value = {'arguments': {}}
    else:
        return Response('Synthetic CPU final answer')
    return Response(json.dumps(value))

class Contracts(unittest.TestCase):
    def test_original_files(self):
        from rsi_tools.common import verify_vendor
        self.assertEqual(verify_vendor()['commit'], '6a82719ea6e22cf23f963078e2527094e50b47f6')

    def test_split_and_sealed_test(self):
        split = validate_split()
        train, sel = developer_subset(3, 2)
        self.assertEqual([x['id'] for x in train], split['train_ids'][:3])
        self.assertFalse({x['id'] for x in train + sel} & set(split['sealed_test_ids']))
        with self.assertRaises(ValueError):
            developer_subset(31, 2)

    def test_synthetic_h2_reward_alias(self):
        instance = {'id': 'synthetic-only', 'gold_match': [{'f': {'x': [1]}}, {'g': {'y': [2]}}],
                    'gold_plan': [{'tool': 'f'}, {'tool': 'g'}]}
        partial = [{'tool': 'f', 'args': {'x': 1}, 'status': 'no_execution'},
                   {'tool': 'g', 'args': {'y': 999}, 'status': 'no_execution'}]
        wrong = copy.deepcopy(partial)
        wrong[0]['args'] = {'x': 999}
        self.assertFalse(bfcl_ast_success(instance, partial))
        self.assertFalse(bfcl_ast_success(instance, wrong))
        self.assertEqual(progress(partial, instance), progress(wrong, instance))
        self.assertEqual(matched_gold_calls(instance, partial), 1)
        self.assertEqual(matched_gold_calls(instance, wrong), 0)

    def test_strict_types_and_duplicate_calls(self):
        instance = {'gold_match': [{'f': {'x': [1]}}, {'f': {'x': [2]}}]}
        self.assertEqual(matched_gold_calls(instance, [{'tool': 'f', 'args': {'x': 1}}] * 2), 1)
        self.assertEqual(matched_gold_calls(instance, [{'tool': 'f', 'args': {'x': True}}]), 0)
        self.assertEqual(matched_gold_calls(instance, [{'tool': 'f', 'args': {'x': '1'}}]), 0)
        self.assertEqual(matched_gold_calls(instance, [{'tool': 'f', 'args': {'x': 1, 'extra': 5}}]), 0)

    def test_author_developer_gold_plans(self):
        from rsi_tools.data import normalized_data
        split = validate_split()
        raw = normalized_data()
        for key in split['train_ids'] + split['selection_ids']:
            self.assertTrue(bfcl_ast_success(raw[key], raw[key]['gold_plan']), key)

    def test_local_endpoint_only(self):
        with self.assertRaises(ValueError):
            local_meta('https://example.com/v1')

    def test_budget_stops_before_second_call(self):
        with tempfile.TemporaryDirectory(prefix='rsi-cpu-contract-') as d:
            client = make_client(LLMConfig(model_name='Qwen3-4B', base_url='http://127.0.0.1:8000/v1'),
                                 pathlib.Path(d), {'max_calls': 1, 'max_total_tokens': 20000, 'max_minutes': 1},
                                 create_override=scripted_create, server_meta=META)
            client.generate([{'role': 'user', 'content': 'CPU test'}])
            with self.assertRaises(BudgetStop):
                client.generate([{'role': 'user', 'content': 'CPU test2'}])
            self.assertEqual(client.calls, 1)
            self.assertEqual(client.used_tokens, 30)

    def test_native_context_retry_is_logged(self):
        def limited(**kwargs):
            if sum(len(m['content']) for m in kwargs['messages']) > 25000:
                raise RuntimeError('maximum context length exceeded')
            return Response('{}')
        with tempfile.TemporaryDirectory(prefix='rsi-cpu-contract-') as d:
            client = make_client(LLMConfig(), pathlib.Path(d),
                                 {'max_calls': 3, 'max_total_tokens': 30000, 'max_minutes': 1},
                                 create_override=limited, server_meta=META)
            self.assertEqual(client.generate_json([{'role': 'user', 'content': 'x' * 30000}]), {})
            self.assertEqual(client.calls, 2)
            self.assertIn('error', client.call_records[0])
            self.assertIn('[truncated to fit context]', client.call_records[1]['request']['messages'][0]['content'])
            self.assertEqual(client.used_tokens, 30)

    def test_collect_synthetic_end_to_end(self):
        with tempfile.TemporaryDirectory(prefix='rsi-cpu-contract-') as d:
            plan = read(ROOT / 'configs/collect-3.json')
            plan['train_count'] = 1
            result = execute(plan, pathlib.Path(d), create_override=scripted_create, server_meta=META,
                             run_kind='synthetic_cpu_contract')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(result['unique_episodes'], 1)
            self.assertFalse(result['test_evaluation'])
            self.assertEqual(result['run_kind'], 'synthetic_cpu_contract')
            calls = [json.loads(s) for s in (pathlib.Path(d) / 'model-calls.jsonl').read_text('utf-8').splitlines()]
            self.assertIn('blame', [x['role'] for x in calls])
            self.assertTrue((pathlib.Path(d) / 'h2-offline-diagnostic.json').exists())

    def test_evolve_synthetic_and_partial_budget(self):
        with tempfile.TemporaryDirectory(prefix='rsi-cpu-contract-') as d:
            plan = read(ROOT / 'configs/evolve-2.json')
            result = execute(plan, pathlib.Path(d), create_override=scripted_create, server_meta=META,
                             run_kind='synthetic_cpu_contract')
            self.assertEqual(result['status'], 'completed')
            self.assertTrue((pathlib.Path(d) / 'generations.jsonl').exists())
            self.assertTrue((pathlib.Path(d) / 'selected-policy.json').exists())
            self.assertGreater(result['cache_hits'], 0)
        with tempfile.TemporaryDirectory(prefix='rsi-cpu-contract-') as d:
            plan['limits']['max_calls'] = 1
            result = execute(plan, pathlib.Path(d), create_override=scripted_create, server_meta=META,
                             run_kind='synthetic_cpu_contract')
            self.assertEqual(result['status'], 'budget_exhausted')
            self.assertTrue((pathlib.Path(d) / 'summary.json').exists())

    def test_accepted_mutation_uses_author_selection(self):
        def toy(key):
            return {'id': key, 'query': 'Echo integerone. Synthetic CPU contract.',
                    'available_tools': [{'name': 'echo', 'description': 'Echo', 'parameters': {'x': 'integer'}}],
                    'gold_plan': [{'tool': 'echo', 'args': {'x': 1}}],
                    'gold_match': [{'echo': {'x': [1]}}], 'gold_answer': None, 'benchmark': 'bfcl',
                    'mock_outputs': {}, 'required_token': None, 'metric': 'bfcl', 'recorded_outputs': [], 'outputs': []}
        def create(**kwargs):
            s = kwargs['messages'][0]['content']
            if '"plan"' in s:
                return Response('{"plan":["Echo integerone"]}')
            if '"tool"' in s:
                return Response('{"tool":"echo"}')
            if '"arguments"' in s:
                return Response('{"arguments":{"x":1}}' if 'CPU_CONTRACT_X1' in s else '{"arguments":{"x":0}}')
            if 'targeted prompt editor' in s:
                return Response('{"target_module":"caller","revised_spec":"CPU_CONTRACT_X1"}')
            return scripted_create(**kwargs)
        with tempfile.TemporaryDirectory(prefix='rsi-cpu-contract-') as d:
            plan = read(ROOT / 'configs/evolve-2.json')
            with patch('rsi_tools.data.developer_subset', return_value=([toy('synthetic-train')], [toy('synthetic-selection')])):
                result = execute(plan, pathlib.Path(d), create_override=create, server_meta=META,
                                 run_kind='synthetic_cpu_contract')
            self.assertEqual(result['status'], 'completed')
            logs = [json.loads(x) for x in (pathlib.Path(d) / 'generations.jsonl').read_text('utf-8').splitlines()]
            self.assertTrue(logs[0]['accepted'])
            selected = read(pathlib.Path(d) / 'selected-policy.json')
            self.assertEqual(selected['caller'], 'CPU_CONTRACT_X1')
            self.assertFalse(result['test_evaluation'])

    def test_loopback_server_sdk_and_errors(self):
        import time
        class FakeBackend:
            meta = META['meta']
            def generate(self, body, remaining_tokens):
                if body.get('bad_request'):
                    raise ValueError('maximum context length exceeded')
                r = scripted_create(messages=body['messages'])
                return r.content, 'stop', r.model_dump()['usage'], {'synthetic': True}
        with tempfile.TemporaryDirectory(prefix='rsi-cpu-contract-') as d:
            limits = {'deadline': time.monotonic() + 30, 'used_requests': 0, 'used_tokens': 0,
                      'max_requests': 4, 'max_tokens': 20000}
            server = HTTPServer(('127.0.0.1', 0), handler_factory(FakeBackend(), pathlib.Path(d), limits))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = 'http://127.0.0.1:' + str(server.server_port) + '/v1'
                client = make_client(LLMConfig(model_name='Qwen3-4B', base_url=base, api_key='EMPTY'),
                                     pathlib.Path(d), {'max_calls': 3, 'max_total_tokens': 20000, 'max_minutes': 1})
                response = client.generate_json([{'role': 'system', 'content': 'Return {"plan":[]}'},
                                                 {'role': 'user', 'content': 'CPU contract'}])
                self.assertIsInstance(response['plan'], list)
                self.assertEqual(client.used_tokens, 30)
                payload = json.dumps({'bad_request': True}).encode()
                with self.assertRaises(urllib.error.HTTPError):
                    urllib.request.urlopen(urllib.request.Request(base + '/chat/completions', data=payload,
                                                                  headers={'Content-Type': 'application/json'}), timeout=5)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

if __name__ == '__main__':
    unittest.main(verbosity=2)
