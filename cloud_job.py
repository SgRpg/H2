"""Single approved cloud job: supervise local server + author-baseline pilot."""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from rsi_tools.common import ROOT, digest, new_run_dir, now, read, write_new

def stop_process(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=8)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True)
    p.add_argument('--model-dir', required=True)
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    profile_path = pathlib.Path(a.plan).resolve()
    profile = read(profile_path)
    maximum = profile['limits']['max_minutes'] + 5
    if not a.execute:
        print(json.dumps({'dry_run': True, 'profile_sha256': digest(profile),
                          'max_gpu_minutes_including_startup': maximum, 'model_dir': a.model_dir,
                          'approval': 'Requires registered plan and interactive experiment-approve in the research workflow.'}, ensure_ascii=False, indent=2))
        return 0
    if sys.platform != 'linux' or not os.environ.get('ARAG_RUN_DIR'):
        raise SystemExit('Use the approved research experiment-run entry onLinux; ARAG_RUN_DIR is absent. NoGPUprocess started.')
    out = new_run_dir(pathlib.Path(os.environ['ARAG_RUN_DIR']) / 'cloud-artifacts')
    write_new(out / 'cloud-job.json', {'started_at': now(), 'profile': profile,
                                      'maximum_gpu_minutes': maximum, 'model_dir': str(pathlib.Path(a.model_dir).resolve())})
    # Refuse to borrow an already running service from a different configuration.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open('http://127.0.0.1:8000/health', timeout=1):
            raise RuntimeError('Port8000already serves a model; choose a clean approved job')
    except urllib.error.URLError:
        pass
    server = collector = None
    deadline = time.monotonic() + maximum * 60
    previous = signal.getsignal(signal.SIGTERM)
    def on_terminate(signum, frame):
        raise TimeoutError('Formalrunner terminated cloudjob')
    signal.signal(signal.SIGTERM, on_terminate)
    status, returncode, error = 'runtime_error', 2, None
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['HF_HUB_OFFLINE'] = '1'
    env['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    try:
        with (out / 'server.log').open('x', encoding='utf-8') as serverlog, (out / 'pilot.log').open('x', encoding='utf-8') as pilotlog:
            server = subprocess.Popen([sys.executable, '-B', '-m', 'rsi_tools.server',
                '--model-dir', str(pathlib.Path(a.model_dir).resolve()), '--out', str(out / 'model-server'),
                '--max-context', str(profile['max_context']), '--max-new-tokens', str(profile['max_new_tokens']),
                '--max-minutes', str(maximum), '--max-requests', str(profile['limits']['max_calls']),
                '--max-total-tokens', str(profile['limits']['max_total_tokens'])],
                cwd=ROOT, env=env, stdout=serverlog, stderr=subprocess.STDOUT, start_new_session=True)
            # Boundedstartup check; no network download or automatic dtype/context fallback.
            while True:
                if server.poll() is not None:
                    raise RuntimeError('Modelserver exited; inspect server.log')
                if time.monotonic() >= deadline:
                    raise TimeoutError('Cloudjob total wall budget exhausted during startup')
                try:
                    with opener.open('http://127.0.0.1:8000/health', timeout=1) as f:
                        if json.load(f).get('status') == 'ready':
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(0.5)
            collector = subprocess.Popen([sys.executable, '-B', 'pilot.py', '--plan', str(profile_path),
                '--out', str(out / 'pilot'), '--execute'], cwd=ROOT, env=env,
                stdout=pilotlog, stderr=subprocess.STDOUT, start_new_session=True)
            returncode = collector.wait(timeout=max(0.1, deadline - time.monotonic()))
            status = 'completed' if returncode == 0 else 'incomplete_or_error'
    except (Exception, KeyboardInterrupt) as exc:
        error = type(exc).__name__ + ': ' + str(exc)[:1000]
        returncode = 2
    finally:
        stop_process(collector)
        stop_process(server)
        signal.signal(signal.SIGTERM, previous)
        write_new(out / 'cloud-summary.json', {'finished_at': now(), 'status': status,
                                              'returncode': returncode, 'error': error})
    return returncode

if __name__ == '__main__':
    sys.exit(main())
