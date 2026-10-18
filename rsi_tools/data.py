"""Freeze the author's 90/30/30 split BEFORE subsampling developer tasks."""
from __future__ import annotations
import json
import random
from .common import ROOT, VENDOR, read, sha, write_new

DATA = VENDOR / 'data/bfcl/samples.json'
SPLIT = ROOT / 'configs/bfcl-split.json'

def make_split():
    raw = read(DATA)
    ids = [str(x['id']) for x in raw]
    if len(raw) != 150 or len(set(ids)) != 150:
        raise ValueError('Expected author reference data with150 unique examples')
    shuffled = ids[:]
    random.Random(42).shuffle(shuffled)
    return {'source': 'author BFCL reference demo, not full official benchmark',
            'source_commit': '6a82719ea6e22cf23f963078e2527094e50b47f6',
            'data_sha256': sha(DATA), 'seed': 42,
            'train_ids': shuffled[:90], 'selection_ids': shuffled[90:120],
            'sealed_test_ids': shuffled[120:],
            'note': 'All inspected/diagnosed tasks remain developer data; no test scoring in this package.'}

def validate_split():
    frozen = read(SPLIT)
    if frozen != make_split():
        raise ValueError('Data or frozen90/30/30 split changed')
    sets = [set(frozen[k]) for k in ['train_ids', 'selection_ids', 'sealed_test_ids']]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Split overlap')
    return frozen

def normalized_data():
    from src.benchmarks import load_instances
    return {str(x['id']): x for x in load_instances('bfcl', str(VENDOR / 'data'))}

def developer_subset(train_count, selection_count=0):
    frozen = validate_split()
    if not 1 <= train_count <= 30 or not 0 <= selection_count <= 10:
        raise ValueError('Preparation supports1–30train and0–10selection tasks only')
    by_id = normalized_data()
    return ([by_id[k] for k in frozen['train_ids'][:train_count]],
            [by_id[k] for k in frozen['selection_ids'][:selection_count]])
