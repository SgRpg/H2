"""Explicit cloud download; never called by imports, preparation or CPU checks."""
import argparse
import json
import pathlib
from rsi_tools.common import MODEL, REVISION, now, read, sha, write_new

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dest', required=True)
    a = p.parse_args()
    from huggingface_hub import snapshot_download
    dest = pathlib.Path(a.dest).resolve()
    receipt = dest / 'rsi-download-request.json'
    request = {'repo_id': MODEL, 'revision': REVISION, 'token': False}
    if dest.exists() and any(dest.iterdir()):
        if not receipt.is_file() or read(receipt)['request'] != request:
            raise ValueError('Use an empty model directory or resume the same frozen download')
    dest.mkdir(parents=True, exist_ok=True)
    if not receipt.exists():
        write_new(receipt, {'request': request, 'started_at': now()})
    snapshot_download(repo_id=MODEL, revision=REVISION, token=False, local_dir=str(dest),
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'LICENSE', 'README.md'])
    weights = list(dest.glob('*.safetensors'))
    if not weights or not (dest / 'tokenizer.json').is_file():
        raise RuntimeError('Model weights/tokenizer incomplete')
    files = {x.name: sha(x) for x in dest.iterdir() if x.is_file() and x.name != 'rsi-model-manifest.json'}
    result = {'repo_id': MODEL, 'revision': REVISION, 'finished_at': now(), 'files': files,
              'license': 'Apache-2.0; retain downloaded LICENSE', 'weights_downloaded': True}
    final = dest / 'rsi-model-manifest.json'
    if final.exists():
        old = read(final)
        if old['repo_id'] != MODEL or old['revision'] != REVISION or old['files'] != files:
            raise RuntimeError('Existing model receipt differs')
    else:
        write_new(final, result)
    print(json.dumps({'model': MODEL, 'revision': REVISION, 'weight_files': len(weights),
                      'verified_manifest': str(final)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
