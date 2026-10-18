"""Offline diagnostics only. Never changes the author's reward or success check."""
from __future__ import annotations
import argparse
import collections
import json
import pathlib
from .common import activate_vendor, read, write_new

def matched_gold_calls(instance, predicted):
    from src.eval_bfcl_ast import bfcl_ast_success
    gold = instance.get('gold_match') or []
    edges = [[j for j, pred in enumerate(predicted)
              if bfcl_ast_success({'gold_match': [g]}, [pred])] for g in gold]
    owner = {}
    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owner or augment(owner[j], seen):
                owner[j] = i
                return True
        return False
    return sum(augment(i, set()) for i in range(len(gold)))

def episode_diagnostic(episode):
    from src.metrics import progress
    from src.eval_bfcl_ast import bfcl_ast_success
    instance = episode['instance']
    predicted = episode.get('predicted_plan', [])
    return {'task_id': str(instance['id']), 'policy_id': episode.get('policy_id'),
            'author_progress': progress(predicted, instance),
            'author_success': bfcl_ast_success(instance, predicted),
            'matched_gold_call_count': matched_gold_calls(instance, predicted),
            'gold_call_count': len(instance.get('gold_match') or []),
            'predicted_call_count': len(predicted),
            'observation_modes': dict(collections.Counter(x.get('status', 'unknown') for x in predicted)),
            'h1_plan_cut_by_step_cap': len(episode.get('plan', [])) > episode['max_steps'],
            'run_kind': episode.get('run_kind', 'unknown'),
            'interpretation': 'Diagnostic signal granularity only; not proof of optimization harm or improvement.'}

def analyze_file(path):
    activate_vendor()
    items = [json.loads(line) for line in pathlib.Path(path).read_text('utf-8').splitlines() if line.strip()]
    episodes = [x for x in items if x.get('event') == 'episode_complete']
    rows = [episode_diagnostic(x) for x in episodes]
    by_task = collections.defaultdict(list)
    for row in rows:
        by_task[row['task_id']].append(row)
    collisions = []
    for task, candidates in by_task.items():
        for i, a in enumerate(candidates):
            for b in candidates[i + 1:]:
                if (abs(a['author_progress'] - b['author_progress']) < 1e-12
                        and a['author_success'] == b['author_success']
                        and a['matched_gold_call_count'] != b['matched_gold_call_count']):
                    collisions.append({'task_id': task, 'policies': [a['policy_id'], b['policy_id']],
                                       'author_progress': a['author_progress'],
                                       'matched_gold_call_counts': [a['matched_gold_call_count'], b['matched_gold_call_count']]})
    return {'evidence_kind': 'offline diagnostic of supplied traces', 'unique_tasks': len(by_task),
            'unique_episode_records': len(rows), 'cache_hit_records_excluded': len(items) - len(episodes),
            'rows': rows, 'within_task_reward_collisions': collisions,
            'warning': 'No cross-task collision is causal evidence. Empty collision list does not refute H2 with few/no alternative policies.'}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--episodes', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = analyze_file(a.episodes)
    write_new(a.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
