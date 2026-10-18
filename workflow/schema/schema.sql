PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS schema_version(version INTEGER PRIMARY KEY CHECK(version=1));
INSERT OR IGNORE INTO schema_version VALUES(1);
CREATE TABLE IF NOT EXISTS sources(
 id TEXT PRIMARY KEY, title TEXT NOT NULL, kind TEXT NOT NULL,
 uri TEXT NOT NULL, retrieved_at TEXT NOT NULL, sha256 TEXT,
 metadata_json TEXT NOT NULL DEFAULT '{}', CHECK(sha256 IS NULL OR length(sha256)=64));
CREATE TABLE IF NOT EXISTS entities(
 id TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('paper','concept','question','method','dataset','idea','result')),
 label TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'unverified', details_json TEXT NOT NULL DEFAULT '{}');
CREATE TABLE IF NOT EXISTS claims(
 id TEXT PRIMARY KEY, entity_id TEXT NOT NULL REFERENCES entities(id),
 source_id TEXT NOT NULL REFERENCES sources(id), locator TEXT NOT NULL CHECK(length(locator)>0),
 text TEXT NOT NULL, epistemic TEXT NOT NULL CHECK(epistemic IN ('quoted','reported','inferred','hypothesis','synthetic')),
 verification TEXT NOT NULL DEFAULT 'unverified' CHECK(verification IN ('unverified','checked','contradicted')));
CREATE TABLE IF NOT EXISTS edges(
 id TEXT PRIMARY KEY, source_entity TEXT NOT NULL REFERENCES entities(id),
 target_entity TEXT NOT NULL REFERENCES entities(id), relation TEXT NOT NULL,
 source_id TEXT NOT NULL REFERENCES sources(id), rationale TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS ideas(
 id TEXT PRIMARY KEY, hypothesis TEXT NOT NULL, lane TEXT NOT NULL CHECK(lane IN ('mainline','exploration')),
 status TEXT NOT NULL DEFAULT 'proposed', details_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS one_mainline ON ideas(lane) WHERE lane='mainline' AND status NOT IN ('archived','rejected');
CREATE TABLE IF NOT EXISTS reviews(
 id INTEGER PRIMARY KEY, idea_id TEXT NOT NULL REFERENCES ideas(id),
 stage TEXT NOT NULL CHECK(stage IN ('novelty','feasibility','minimal')),
 decision TEXT NOT NULL CHECK(decision IN ('pass','fail')), evidence TEXT NOT NULL,
 created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS baselines(
 id TEXT PRIMARY KEY, path TEXT NOT NULL, url TEXT NOT NULL, commit_sha TEXT NOT NULL,
 tree_sha256 TEXT NOT NULL, license TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS experiment_plans(
 id TEXT PRIMARY KEY, plan_json TEXT NOT NULL, digest TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS approvals(
 id INTEGER PRIMARY KEY, plan_id TEXT NOT NULL REFERENCES experiment_plans(id),
 digest TEXT NOT NULL, human TEXT NOT NULL, approved_at TEXT NOT NULL, consumed_at TEXT);
CREATE TABLE IF NOT EXISTS runs(
 id TEXT PRIMARY KEY, plan_id TEXT REFERENCES experiment_plans(id),
 kind TEXT NOT NULL CHECK(kind IN ('synthetic-demo','formal')), status TEXT NOT NULL,
 manifest_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS metrics(
 run_id TEXT NOT NULL REFERENCES runs(id), name TEXT NOT NULL, value REAL NOT NULL,
 PRIMARY KEY(run_id,name));
CREATE TABLE IF NOT EXISTS audit(
 id INTEGER PRIMARY KEY, action TEXT NOT NULL, object_id TEXT NOT NULL,
 details_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS claims_entity ON claims(entity_id);
CREATE INDEX IF NOT EXISTS edges_source ON edges(source_entity);
CREATE INDEX IF NOT EXISTS edges_target ON edges(target_entity);
