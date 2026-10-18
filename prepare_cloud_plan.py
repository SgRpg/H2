"""Write a concrete draft after cloud setup/download. Never approves or runs it."""
import argparse
import pathlib
import sys
from rsi_tools.common import ROOT, MODEL, REVISION, read, sha, verify_vendor, write_new

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', default='configs/collect-3.json')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--id', default='rsi-evotool-collect3-v1')
    p.add_argument('--out', default='configs/cloud-experiment-draft.json')
    a = p.parse_args()
    profile_path = pathlib.Path(a.profile).resolve()
    model_dir = pathlib.Path(a.model_dir).resolve()
    if not profile_path.is_relative_to(ROOT) or not model_dir.is_relative_to(ROOT):
        raise ValueError('Profile/model must be inside the uploaded project for workflow input hashing')
    receipt = read(model_dir / 'rsi-model-manifest.json')
    if receipt['repo_id'] != MODEL or receipt['revision'] != REVISION:
        raise ValueError('Unexpected model receipt')
    verify_vendor()
    profile = read(profile_path)
    inputs = [p.relative_to(ROOT).as_posix() for p in sorted(ROOT.rglob('*')) if p.is_file()
              and '.git' not in p.parts and '__pycache__' not in p.parts
              and p.suffix in ['.py', '.json', '.txt', '.safetensors', '.yaml', '.yml', '.sql']
              and not p.is_relative_to(ROOT / 'outputs') and not p.is_relative_to(ROOT / '.arag')
              and not p.is_relative_to(ROOT / 'experiments') and p.resolve() != pathlib.Path(a.out).resolve()]
    plan = {'id': a.id, 'idea_id': 'rsi-evotool-feedback-audit-v1', 'stage': 'minimal',
            'command': [sys.executable, 'cloud_job.py', '--plan', profile_path.relative_to(ROOT).as_posix(),
                        '--model-dir', model_dir.relative_to(ROOT).as_posix(), '--execute'],
            'seeds': [42], 'inputs': inputs,
            'hardware': {'gpu_count': 1, 'gpu': 'TeslaV100-SXM2-16GB', 'dtype': 'float16'},
            'max_minutes': profile['limits']['max_minutes'] + 5,
            'budget': {'max_model_calls': profile['limits']['max_calls'], 'max_model_tokens': profile['limits']['max_total_tokens'],
                       'external_api_calls': 0, 'max_gpu_hours': (profile['limits']['max_minutes'] + 5) / 60,
                       'currency_cost': 'Unknown; operator must check currentDSWbilling/freequota before approval'},
            'metrics': ['runtime completion', 'strict authorBFCLsuccess', 'reward-signal collisions', 'all-role tokens', 'budget/truncation flags'],
            'dataset': {'name': 'authorBFCL150demo', 'split': 'fixed90/30/30; firstdeveloper tasks only; testsealed',
                        'sha256': sha(ROOT / 'vendor/evotool/data/bfcl/samples.json'), 'licensing': 'upstream provenance review required before formal release'},
            'model': {'id': MODEL, 'revision': REVISION, 'manifest_sha256': sha(model_dir / 'rsi-model-manifest.json')},
            'hypothesis': 'Diagnostic collection only; no claim of innovation or method effectiveness; novelty/feasibility reviews remain pending.'}
    write_new(a.out, plan)
    print('WroteDRAFT:', a.out)
    print('Notregistered, notapproved, notexecuted. Reviewtheprojectgatesbeforeexperiment-plan.')

if __name__ == '__main__':
    main()
