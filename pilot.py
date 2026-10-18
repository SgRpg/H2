"""Explicit author-baseline adapter entry. Does not train model weights."""
from __future__ import annotations
import argparse
import dataclasses
import json
import pathlib
import os
import sys
from rsi_tools.common import ROOT, MODEL, REVISION, activate_vendor, digest, environment, new_run_dir, now, read, sha, write_new

def execute(plan, out, *, create_override=None, server_meta=None, run_kind='real_model'):
    activate_vendor()
    from src.config import EvoToolConfig
    from src.policy import agent, modules
    from src.evolve.blame import blame
    from src.evolve.diagnostics import extract_diagnostics
    from src.evolve.loop import evolve
    from src.evolve.select import best_policy
    from rsi_tools.data import developer_subset, validate_split
    from rsi_tools.tracing import BudgetStop, EpisodeCache, GenerationLog, make_client
    from rsi_tools.analysis import analyze_file
    if plan['mode'] not in ['collect', 'evolve'] or not 1 <= plan['max_steps'] <= 12:
        raise ValueError('Unknown mode or invalid step budget')
    if plan['mode'] == 'evolve' and (not 1 <= plan['generations'] <= 3 or plan['selection_count'] < 1):
        raise ValueError('Short evolution needs1–3generations and a nonempty developer selection set')
    frozen = validate_split()
    train, sel = developer_subset(plan['train_count'], plan['selection_count'])
    cfg = EvoToolConfig()
    cfg.benchmark = 'bfcl'
    cfg.seed = 42
    cfg.max_steps = plan['max_steps']
    cfg.llm.model_name = 'Qwen3-4B'
    cfg.llm.base_url = plan['base_url']
    cfg.llm.api_key = 'EMPTY'
    cfg.llm.temperature = 0.0
    cfg.llm.max_tokens = plan['max_new_tokens']
    cfg.evolve.epochs = 1
    cfg.evolve.batch_size = 1
    cfg.evolve.generations = plan['generations']
    cfg.evolve.log_test_eval = False
    cfg.evolve.parent_select = 'win'
    cfg.evolve.accept_on = 'batch'
    client = make_client(cfg.llm, out, plan['limits'], create_override=create_override, server_meta=server_meta)
    actual_meta = client.server_meta['meta']
    if (actual_meta.get('revision') != REVISION or actual_meta.get('dtype') != 'float16'
            or actual_meta.get('max_context') != plan['max_context']):
        raise ValueError('Server model/precision/context differs from frozen pilot plan')
    cache = EpisodeCache(client, out, cfg.max_steps, run_kind)
    previous_cache = agent._EPISODE_CACHE
    agent._EPISODE_CACHE = cache
    manifest = {'started_at': now(), 'plan': plan, 'plan_sha256': digest(plan),
                'run_kind': run_kind, 'environment': environment(), 'server': client.server_meta,
                'baseline': read(ROOT / 'baseline-manifest.json'),
                'task_ids': {'train': [x['id'] for x in train], 'selection': [x['id'] for x in sel]},
                'split_sha256': digest(frozen), 'test_evaluation': False,
                'source_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in ROOT.glob('*.py')},
                'adapter_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'rsi_tools').glob('*.py')}}
    write_new(out / 'manifest.json', manifest)
    completed, status, error = [], 'completed', None
    try:
        if plan['mode'] == 'collect':
            policy = modules.initial_policy()
            write_new(out / 'initial-policy.json', dataclasses.asdict(policy))
            for instance in train:
                episode = agent.run_episode(client, policy, instance, cfg.max_steps)
                diagnostic = extract_diagnostics(episode, instance)
                reason = {}
                target = blame(client, episode, diagnostic, rationale_out=reason) if plan['collect_blame'] else None
                completed.append(str(instance['id']))
                with (out / 'blame.jsonl').open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'task_id': instance['id'], 'target': target, 'rationale': reason,
                                        'diagnostics': diagnostic}, ensure_ascii=False) + '\n')
        else:
            population = evolve(client, cfg, train, sel, test=None, runlog=GenerationLog(out), log=print)
            best = best_policy(client, cfg, population, sel)
            write_new(out / 'retained-policies.json', [dataclasses.asdict(x) for x in population])
            write_new(out / 'selected-policy.json', dataclasses.asdict(best))
    except BudgetStop as exc:
        status, error = 'budget_exhausted', str(exc)
    except Exception as exc:
        status, error = 'runtime_error', type(exc).__name__ + ': ' + str(exc)[:1500]
    finally:
        agent._EPISODE_CACHE = previous_cache
    if (out / 'episodes.jsonl').is_file():
        write_new(out / 'h2-offline-diagnostic.json', analyze_file(out / 'episodes.jsonl'))
    accounting_complete = all('error' not in row for row in client.call_records)
    json_validation = {'known_valid': 0, 'known_invalid': 0, 'unknown': 0}
    for row in client.call_records:
        if row.get('request', {}).get('response_format', {}).get('type') != 'json_object':
            continue
        valid = row.get('response', {}).get('rsi_diagnostic', {}).get('json_parse_ok')
        json_validation['known_valid' if valid is True else 'known_invalid' if valid is False else 'unknown'] += 1
    summary = {'finished_at': now(), 'status': status, 'error': error, 'run_kind': run_kind,
               'completed_collect_task_ids': completed, 'unique_episodes': cache.episode_count,
               'cache_hits': cache.cache_hits, 'attempted_model_calls': client.calls,
               'actual_total_tokens_all_roles': client.used_tokens if accounting_complete else None,
               'known_total_tokens_all_roles': client.used_tokens,
               'token_accounting_complete': accounting_complete,
               'json_object_validation': json_validation,
               'generation_truncation_count': sum(bool(x.get('output_truncated')) for x in client.call_records),
               'author_client_token_counter': client.total_tokens,
               'test_evaluation': False, 'weights_updated': False}
    write_new(out / 'summary.json', summary)
    hashes = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    write_new(out / 'output-hashes.json', hashes)
    return summary

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--execute', action='store_true', help='Run only after the project GPU-plan approval process')
    a = p.parse_args()
    plan = read(a.plan)
    if not a.execute:
        print(json.dumps({'preparation_only': True, 'plan': plan, 'plan_sha256': digest(plan),
                          'next': 'Review/register the formalGPUplan before execution; no model call was made.'}, ensure_ascii=False, indent=2))
        return 0
    if not os.environ.get('ARAG_RUN_DIR'):
        raise SystemExit('Realmodel collection must be launched by the approved researchrunner; no modelcall made.')
    out = new_run_dir(a.out)
    summary = execute(plan, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary['status'] == 'completed' else 2

if __name__ == '__main__':
    sys.exit(main())
