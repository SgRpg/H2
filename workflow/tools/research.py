#!/usr/bin/env python3
"""Agentic RAG research ledger. Python 3.10+, standard library only."""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

VERSION = '1.0.0'
PACKAGE = Path(__file__).resolve().parent.parent

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2) + '\n')

def safe_id(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,95}', value):
        raise ValueError('ID must contain 1-96 letters, digits, dots, underscores or hyphens')
    return value

def inside(root, relative, exists=True):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Path must be a file or directory inside the project')
    if exists and not path.exists():
        raise ValueError(f'Missing path: {path}')
    return path

def git(path, *args, required=True):
    proc = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True, encoding='utf-8', timeout=30)
    if required and proc.returncode:
        raise ValueError('Git operation failed: ' + proc.stderr.strip()[:300])
    return proc.stdout.strip() if proc.returncode == 0 else None

def tracked_snapshot(path):
    """Tracked working-tree bytes, including local deletions, never edits the repository."""
    raw = subprocess.run(['git','-C',str(path),'ls-files','-z'], capture_output=True, check=True, timeout=30).stdout
    entries = {}
    for name in raw.decode('utf-8').split('\0'):
        if name:
            file = Path(path) / name
            entries[name] = file_hash(file) if file.is_file() else 'MISSING'
    return entries

def audit(db, action, object_id, details=None):
    db.execute('INSERT INTO audit(action,object_id,details_json,created_at) VALUES(?,?,?,?)',
               (action, object_id, canonical(details or {}), now()))

@contextlib.contextmanager
def connect(project):
    marker = project / '.arag-project.json'
    if not marker.is_file():
        raise ValueError('Project not initialized; use init first')
    if read_json(marker).get('schema_version') != 1:
        raise ValueError('Unsupported project schema; do not overwrite it')
    path = project / '.arag' / 'research.sqlite'
    if not path.is_file():
        raise ValueError('Missing database; restore backup or run init')
    db = sqlite3.connect(path, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    try:
        if db.execute('SELECT version FROM schema_version').fetchone()[0] != 1:
            raise ValueError('Unsupported database schema')
        yield db
        db.commit()
    except BaseException:
        db.rollback()
        raise
    finally:
        db.close()

def init(project):
    project.mkdir(parents=True, exist_ok=True)
    marker = project / '.arag-project.json'
    if marker.exists() and read_json(marker).get('schema_version') != 1:
        raise ValueError('Existing project marker is incompatible')
    (project / '.arag').mkdir(exist_ok=True)
    database = project / '.arag' / 'research.sqlite'
    if database.exists():
        with connect(project) as db:
            return {'status':'already_initialized','database':str(database)}
    schema = (PACKAGE / 'schema' / 'schema.sql').read_text(encoding='utf-8')
    with contextlib.closing(sqlite3.connect(database)) as db:
        with db:
            db.executescript(schema)
    if not marker.exists():
        write_new(marker, {'schema_version':1,'workflow_version':VERSION,'domain':'Agentic RAG',
                          'target':'CCF C; stretch B; venue rank must be verified',
                          'max_explorations':2,'created_at':now()})
    return {'status':'initialized','database':str(database)}

def ingest(project, filename):
    obj = read_json(filename)
    allowed = {'sources','entities','claims','edges'}
    if not set(obj).issubset(allowed):
        raise ValueError('Ingest keys must be sources, entities, claims, edges')
    fields = {
        'sources':['id','title','kind','uri','retrieved_at','sha256','metadata_json'],
        'entities':['id','kind','label','status','details_json'],
        'claims':['id','entity_id','source_id','locator','text','epistemic','verification'],
        'edges':['id','source_entity','target_entity','relation','source_id','rationale']}
    counts = {}
    with connect(project) as db:
        for table in ['sources','entities','claims','edges']:
            count = 0
            for original in obj.get(table, []):
                row = dict(original)
                safe_id(row['id'])
                if table == 'sources':
                    for field in ['title','kind','uri','retrieved_at']:
                        if not str(row.get(field, '')).strip():
                            raise ValueError(f'Source requires {field}')
                    timestamp = dt.datetime.fromisoformat(row['retrieved_at'].replace('Z','+00:00'))
                    if timestamp.tzinfo is None:
                        raise ValueError('retrieved_at must include timezone')
                    row.setdefault('sha256', None)
                    if row['sha256'] is not None and not re.fullmatch('[a-fA-F0-9]{64}', row['sha256']):
                        raise ValueError('Invalid SHA256')
                    row.setdefault('metadata_json', {})
                if table == 'entities':
                    row.setdefault('status','unverified')
                    row.setdefault('details_json',{})
                if table == 'claims':
                    row.setdefault('verification','unverified')
                for field in fields[table]:
                    if field not in row:
                        raise ValueError(f'{table} missing field: {field}')
                    if field.endswith('_json'):
                        if isinstance(row[field], str):
                            row[field] = json.loads(row[field])
                        row[field] = canonical(row[field])
                values = [row[f] for f in fields[table]]
                prior = db.execute(f'SELECT * FROM {table} WHERE id=?',(row['id'],)).fetchone()
                if prior:
                    if any(prior[f] != row[f] for f in fields[table]):
                        raise ValueError(f'Immutable record {row["id"]} differs; use a new version ID')
                    continue
                db.execute(f'INSERT INTO {table}({",".join(fields[table])}) VALUES({",".join("?" for _ in values)})', values)
                count += 1
            counts[table] = count
        audit(db,'ingest',file_hash(filename),counts)
    return counts

def knowledge_export(project):
    with connect(project) as db:
        tables = {t:[dict(r) for r in db.execute(f'SELECT * FROM {t} ORDER BY id')]
                  for t in ['sources','entities','claims','edges','ideas','reviews']}
    key = digest(tables)[:12]
    out = project / 'knowledge' / 'exports' / key
    if out.exists():
        return {'status':'already_exported','path':str(out)}
    lines = ['# Agentic RAG 知识网络','', '数据库记录的只读快照；引用支持状态与研究假设分开列出。','']
    for entity in tables['entities']:
        lines += [f'## {entity["id"]}: {entity["label"]}',f'类型：{entity["kind"]}；状态：{entity["status"]}','']
        for c in tables['claims']:
            if c['entity_id'] == entity['id']:
                lines += [f'- [{c["epistemic"]}/{c["verification"]}] {c["text"]}（{c["source_id"]}, {c["locator"]}）']
        for e in tables['edges']:
            if e['source_entity'] == entity['id']:
                lines += [f'- → {e["relation"]} → {e["target_entity"]}：{e["rationale"]}（{e["source_id"]}）']
        lines += ['']
    lines += ['## 来源','']
    for s in tables['sources']:
        lines += [f'- {s["id"]}: {s["title"]} | {s["uri"]} | accessed {s["retrieved_at"]} | SHA256 {s["sha256"] or "not captured"}']
    write_new(out / 'knowledge.md','\n'.join(lines)+'\n')
    write_new(out / 'graph.json',tables)
    return {'status':'exported','path':str(out),'records':sum(len(v) for v in tables.values())}

def idea_add(project, filename):
    idea = read_json(filename)
    for key in ['id','hypothesis','lane','falsifier','nearest_work','minimum_test','budget']:
        if not idea.get(key):
            raise ValueError(f'Idea requires {key}')
    safe_id(idea['id'])
    with connect(project) as db:
        limit = int(read_json(project / '.arag-project.json').get('max_explorations',2))
        if idea['lane'] == 'exploration' and db.execute("SELECT count(*) FROM ideas WHERE lane='exploration' AND status NOT IN ('archived','rejected')").fetchone()[0] >= limit:
            raise ValueError('Exploration limit reached; archive one idea or explicitly revise project limit')
        db.execute('INSERT INTO ideas(id,hypothesis,lane,details_json,created_at) VALUES(?,?,?,?,?)',
                   (idea['id'],idea['hypothesis'],idea['lane'],canonical(idea),now()))
        audit(db,'idea-add',idea['id'])
    return {'id':idea['id'],'status':'proposed'}

def idea_review(project, args):
    stages = ['novelty','feasibility','minimal']
    if len(args.evidence.strip()) < 20:
        raise ValueError('Supply concrete evidence, links/record IDs and decision rationale (at least 20 characters)')
    with connect(project) as db:
        if not db.execute('SELECT 1 FROM ideas WHERE id=?',(args.id,)).fetchone():
            raise ValueError('Unknown idea')
        for stage in stages[:stages.index(args.stage)]:
            row = db.execute('SELECT decision FROM reviews WHERE idea_id=? AND stage=? ORDER BY id DESC LIMIT 1',(args.id,stage)).fetchone()
            if not row or row[0] != 'pass':
                raise ValueError(f'Previous gate has not passed: {stage}')
        db.execute('INSERT INTO reviews(idea_id,stage,decision,evidence,created_at) VALUES(?,?,?,?,?)',
                   (args.id,args.stage,args.decision,args.evidence,now()))
        status = args.stage + ('_passed' if args.decision == 'pass' else '_failed')
        db.execute('UPDATE ideas SET status=? WHERE id=?',(status,args.id))
        audit(db,'idea-review',args.id,{'stage':args.stage,'decision':args.decision})
    return {'id':args.id,'status':status}

def baseline_register(project, args):
    path = Path(args.path).resolve()
    if not path.is_dir():
        raise ValueError('Baseline directory missing')
    commit = git(path,'rev-parse','HEAD')
    status = git(path,'status','--porcelain')
    if status:
        raise ValueError('Baseline has local changes/untracked files; register a clean isolated clone')
    tree = digest(tracked_snapshot(path))
    name = safe_id(args.id or path.name)
    with connect(project) as db:
        db.execute('INSERT INTO baselines VALUES(?,?,?,?,?,?,?)',
                   (name,str(path),args.url,commit,tree,args.license,now()))
        audit(db,'baseline-register',name,{'commit':commit})
    return {'id':name,'commit':commit,'tree_sha256':tree}

def input_hashes(project, paths):
    hashes = {}
    for relative in paths:
        p = inside(project, relative)
        if not p.is_file():
            raise ValueError('Tracked inputs must be files: '+str(relative))
        hashes[str(relative)] = file_hash(p)
    if not hashes:
        raise ValueError('At least one input file is required')
    return hashes

def experiment_plan(project, filename):
    plan = read_json(filename)
    required = ['id','idea_id','stage','command','seeds','inputs','hardware','max_minutes','budget','metrics','dataset','model']
    for key in required:
        if key not in plan or plan[key] in (None,'',[],{}):
            raise ValueError('Plan requires '+key)
    safe_id(plan['id'])
    if plan['stage'] not in ['minimal','pilot','full','ablation','robustness']:
        raise ValueError('Unknown experiment stage')
    if not isinstance(plan['command'],list) or any(not isinstance(x,str) for x in plan['command']):
        raise ValueError('command must be a JSON array of executable and arguments, never shell text')
    if not isinstance(plan['seeds'],list) or any(type(x) is not int for x in plan['seeds']) or len(set(plan['seeds'])) != len(plan['seeds']):
        raise ValueError('seeds must be unique integers')
    if not 0 < float(plan['max_minutes']) <= 7*24*60:
        raise ValueError('max_minutes must be 0..10080')
    for label,keys in [('dataset',['name','revision','split','license']),('model',['name','revision'])]:
        if not isinstance(plan[label],dict) or any(not plan[label].get(k) for k in keys):
            raise ValueError(label+' must include '+', '.join(keys))
    if len(plan['seeds']) > 1 and not any('{seed}' in x for x in plan['command']):
        raise ValueError('Multiple seeds require {seed} in command arguments')
    forbidden = re.compile(r'(?i)(api[_-]?key|bearer\s|password=|token=|secret=)')
    if any(forbidden.search(x) for x in plan['command']):
        raise ValueError('Credentials must not appear in command arguments')
    plan['input_hashes'] = input_hashes(project,plan['inputs'])
    executable=shutil.which(plan['command'][0])
    if not executable:
        candidate=inside(project,plan['command'][0])
        if not candidate.is_file():
            raise ValueError('Command executable not found')
        executable=str(candidate)
    executable=Path(executable).resolve()
    plan['executable']={'path':str(executable),'sha256':file_hash(executable)}
    plan['code_hashes'] = {str(p.relative_to(PACKAGE)):file_hash(p) for p in (PACKAGE/'tools').glob('*.py')}
    plan['workflow_version'] = VERSION
    plan['project_commit'] = git(project,'rev-parse','HEAD',required=False) if shutil.which('git') else None
    plan['project_git_status'] = git(project,'status','--porcelain',required=False) if shutil.which('git') else None
    if plan['stage'] in ['full','ablation','robustness'] and (not plan['project_commit'] or plan['project_git_status']):
        raise ValueError('Full/ablation/robustness runs require a clean committed project; commit explicit source files first')
    with connect(project) as db:
        if not db.execute('SELECT 1 FROM ideas WHERE id=?',(plan['idea_id'],)).fetchone():
            raise ValueError('Register an idea before planning experiments')
        for stage in ['novelty','feasibility'] + ([] if plan['stage']=='minimal' else ['minimal']):
            review = db.execute('SELECT decision FROM reviews WHERE idea_id=? AND stage=? ORDER BY id DESC LIMIT 1',(plan['idea_id'],stage)).fetchone()
            if not review or review[0] != 'pass':
                raise ValueError('Idea gate not passed: '+stage)
        plan['baselines'] = [dict(r) for r in db.execute('SELECT * FROM baselines')]
        sha = digest(plan)
        db.execute('INSERT INTO experiment_plans VALUES(?,?,?,?)',(plan['id'],canonical(plan),sha,now()))
        audit(db,'experiment-plan',plan['id'],{'digest':sha})
    write_new(project/'experiments'/'plans'/(plan['id']+'.json'),{'digest':sha,'plan':plan})
    return {'id':plan['id'],'digest':sha,'approval_required':True}

def get_plan(db, name):
    row = db.execute('SELECT * FROM experiment_plans WHERE id=?',(name,)).fetchone()
    if not row:
        raise ValueError('Unknown experiment plan')
    plan = json.loads(row['plan_json'])
    if digest(plan) != row['digest']:
        raise ValueError('Plan integrity failure')
    return plan,row['digest']

def approve(project, args):
    if not sys.stdin.isatty():
        raise ValueError('Human approval requires an interactive terminal; Codex must not approve on your behalf')
    with connect(project) as db:
        plan,sha = get_plan(db,args.id)
        print(json.dumps({'plan':plan,'digest':sha},ensure_ascii=False,indent=2))
        phrase = f'APPROVE {args.id} {sha[:12]}'
        print('Check dataset, command, costs, hardware and inputs. One execution only.')
        if input(f'Type {phrase}: ').strip() != phrase:
            raise ValueError('Approval was not granted')
        human = args.human or input('Your name/identifier: ').strip()
        if not human:
            raise ValueError('Human identifier required')
        db.execute('INSERT INTO approvals(plan_id,digest,human,approved_at) VALUES(?,?,?,?)',(args.id,sha,human,now()))
        audit(db,'experiment-approve',args.id,{'digest':sha,'human':human})
    return {'approved':args.id,'digest':sha}

def check_plan_files(project,plan):
    if file_hash(plan['executable']['path']) != plan['executable']['sha256']:
        raise ValueError('Command executable changed; make a new plan')
    if input_hashes(project,plan['inputs']) != plan['input_hashes']:
        raise ValueError('Experiment inputs changed; make and approve a new plan')
    for name,sha in plan['code_hashes'].items():
        if file_hash(PACKAGE/name) != sha:
            raise ValueError('Workflow code changed; make a new plan')
    if plan.get('project_commit') and (git(project,'rev-parse','HEAD') != plan['project_commit'] or git(project,'status','--porcelain') != plan['project_git_status']):
        raise ValueError('Project Git state changed; make a new plan')
    for b in plan['baselines']:
        if git(b['path'],'rev-parse','HEAD') != b['commit_sha'] or git(b['path'],'status','--porcelain') or digest(tracked_snapshot(b['path'])) != b['tree_sha256']:
            raise ValueError('Baseline changed: '+b['id'])

def environment():
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'workflow_version':VERSION}

def experiment_run(project,args):
    run_id = args.id + '-' + uuid.uuid4().hex[:10]
    with connect(project) as db:
        # Lock approval consumption so two processes cannot launch the same approval.
        db.execute('BEGIN IMMEDIATE')
        plan,sha = get_plan(db,args.id)
        check_plan_files(project,plan)
        row = db.execute('SELECT id FROM approvals WHERE plan_id=? AND digest=? AND consumed_at IS NULL ORDER BY id DESC LIMIT 1',(args.id,sha)).fetchone()
        if not row:
            raise ValueError('No unused human approval for this exact plan')
        db.execute('UPDATE approvals SET consumed_at=? WHERE id=?',(now(),row[0]))
        manifest = {'id':run_id,'plan_digest':sha,'plan':plan,'environment':environment(),'started_at':now(),'seeds':[]}
        db.execute('INSERT INTO runs VALUES(?,?,?,?,?,?)',(run_id,args.id,'formal','running',canonical(manifest),now()))
        audit(db,'experiment-run-start',run_id)
    out = project/'experiments'/'runs'/run_id
    out.mkdir(parents=True)
    status='completed'
    deadline=time.monotonic()+float(plan['max_minutes'])*60
    try:
        for seed in plan['seeds']:
            remaining=deadline-time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Total experiment time budget exhausted')
            command=[x.replace('{seed}',str(seed)).replace('{run_dir}',str(out)).replace('{project}',str(project)) for x in plan['command']]
            command[0]=plan['executable']['path']
            env=os.environ.copy()
            env['ARAG_RUN_DIR']=str(out)
            env['ARAG_SEED']=str(seed)
            with (out/f'seed-{seed}.log').open('x',encoding='utf-8') as log:
                proc=subprocess.Popen(command,cwd=project,stdout=log,stderr=subprocess.STDOUT,env=env,shell=False)
                try:
                    rc=proc.wait(timeout=remaining)
                except BaseException:
                    if os.name=='nt':
                        subprocess.run(['taskkill','/PID',str(proc.pid),'/T','/F'],capture_output=True,timeout=15)
                    else:
                        proc.kill()
                    proc.wait(timeout=15)
                    raise
            manifest['seeds'].append({'seed':seed,'command':command,'returncode':rc})
            if rc:
                raise ValueError(f'Seed {seed} failed with exit code {rc}')
        check_plan_files(project,plan)
    except BaseException as exc:
        status='failed'
        manifest['error']=type(exc).__name__+': '+str(exc)
    manifest.update(status=status,finished_at=now())
    manifest['artifacts']={str(p.relative_to(out)):file_hash(p) for p in out.rglob('*') if p.is_file()}
    write_new(out/'manifest.json',manifest)
    with connect(project) as db:
        db.execute('UPDATE runs SET status=?,manifest_json=? WHERE id=?',(status,canonical(manifest),run_id))
        audit(db,'experiment-run-finish',run_id,{'status':status})
    if status!='completed':
        raise ValueError(f'Run failed; inspect {out}/manifest.json')
    return {'run_id':run_id,'status':status,'path':str(out)}

def status(project):
    with connect(project) as db:
        return {'counts':{table:db.execute(f'SELECT count(*) FROM {table}').fetchone()[0] for table in
                          ['sources','entities','claims','edges','ideas','reviews','baselines','experiment_plans','runs']},
                'ideas':[dict(r) for r in db.execute('SELECT id,lane,status,hypothesis FROM ideas')],
                'pending_experiments':[r[0] for r in db.execute('SELECT id FROM experiment_plans WHERE id NOT IN (SELECT plan_id FROM runs WHERE plan_id IS NOT NULL)')]}

def doctor(project):
    with connect(project) as db:
        integrity=db.execute('PRAGMA integrity_check').fetchone()[0]
        fk=[list(r) for r in db.execute('PRAGMA foreign_key_check')]
        baselines=[]
        for b in db.execute('SELECT * FROM baselines'):
            ok=git(b['path'],'rev-parse','HEAD',required=False)==b['commit_sha'] and digest(tracked_snapshot(b['path']))==b['tree_sha256']
            baselines.append({'id':b['id'],'unchanged':ok})
    result={'ok':integrity=='ok' and not fk and all(b['unchanged'] for b in baselines),
            'sqlite':sqlite3.sqlite_version,'integrity':integrity,'foreign_key_errors':fk,'baselines':baselines,
            'environment':environment(),'optional':{name:bool(shutil.which(name)) for name in ['git','pdftotext','mineru','nvidia-smi']},
            'gpu_experiments':'require explicit human approval','zotero_write':'requires separate staged approval'}
    if not result['ok']:
        raise ValueError(canonical(result))
    return result

def backup(project):
    target=project/'backups'/('knowledge-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]+'.sqlite')
    target.parent.mkdir(parents=True,exist_ok=True)
    with connect(project) as source, contextlib.closing(sqlite3.connect(target)) as destination:
        source.backup(destination)
    write_new(target.with_suffix('.manifest.json'),{'created_at':now(),'sha256':file_hash(target),'schema_version':1})
    return {'path':str(target),'sha256':file_hash(target)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    sub=p.add_subparsers(dest='command',required=True)
    for name in ['init','doctor','status','export','demo','backup']:
        sub.add_parser(name)
    for name in ['ingest','idea-add','experiment-plan']:
        sp=sub.add_parser(name);sp.add_argument('--file',required=True)
    sp=sub.add_parser('search');sp.add_argument('query')
    sp=sub.add_parser('idea-review');sp.add_argument('--id',required=True);sp.add_argument('--stage',choices=['novelty','feasibility','minimal'],required=True);sp.add_argument('--decision',choices=['pass','fail'],required=True);sp.add_argument('--evidence',required=True)
    sp=sub.add_parser('idea-archive');sp.add_argument('--id',required=True)
    sp=sub.add_parser('baseline-register');sp.add_argument('--id');sp.add_argument('--path',required=True);sp.add_argument('--url',required=True);sp.add_argument('--license',required=True)
    sp=sub.add_parser('experiment-approve');sp.add_argument('--id',required=True);sp.add_argument('--human')
    sp=sub.add_parser('experiment-run');sp.add_argument('--id',required=True)
    args=p.parse_args();project=Path(args.project).resolve()
    try:
        if args.command=='init':result=init(project)
        elif args.command=='doctor':result=doctor(project)
        elif args.command=='status':result=status(project)
        elif args.command=='ingest':result=ingest(project,args.file)
        elif args.command=='export':result=knowledge_export(project)
        elif args.command=='backup':result=backup(project)
        elif args.command=='idea-add':result=idea_add(project,args.file)
        elif args.command=='idea-review':result=idea_review(project,args)
        elif args.command=='idea-archive':
            with connect(project) as db:
                cur=db.execute("UPDATE ideas SET status='archived' WHERE id=?",(args.id,))
                if not cur.rowcount:raise ValueError('Unknown idea')
                audit(db,'idea-archive',args.id)
            result={'archived':args.id}
        elif args.command=='baseline-register':result=baseline_register(project,args)
        elif args.command=='experiment-plan':result=experiment_plan(project,args.file)
        elif args.command=='experiment-approve':result=approve(project,args)
        elif args.command=='experiment-run':result=experiment_run(project,args)
        elif args.command=='demo':
            from demo_runner import run_demo
            result=run_demo(project)
        elif args.command=='search':
            with connect(project) as db:
                q='%'+args.query.replace('\\','\\\\').replace('%','\\%').replace('_','\\_')+'%'
                result={'entities':[dict(r) for r in db.execute("SELECT * FROM entities WHERE label LIKE ? ESCAPE '\\' OR details_json LIKE ? ESCAPE '\\'",(q,q))],
                        'claims':[dict(r) for r in db.execute("SELECT * FROM claims WHERE text LIKE ? ESCAPE '\\'",(q,))]}
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError,sqlite3.Error,subprocess.SubprocessError,KeyError,TypeError) as exc:
        print(json.dumps({'error':str(exc)},ensure_ascii=False),file=sys.stderr)
        return 2

if __name__=='__main__':
    raise SystemExit(main())
