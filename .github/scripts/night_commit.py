"""Validate the night agent's files and commit them. Runs in its own job.

The agent's job has a read-only token and cannot push. This job has write access and
trusts nothing the agent produced. Two grades of failure:

  REFUSE  the record could have been rewritten, or something secret-shaped is in it:
          nothing but an empty marker commit is made and the run goes red.
  PROBLEM a file is unusable but the record is intact: everything else is committed, the
          bad file is left out, and the run goes red so the morning check sees it.

Losing ledger rows is itself a harm (it undercounts what was tested), so only tampering
throws a whole night away.

  python .github/scripts/night_commit.py <dir with the agent's files> <night YYYY-MM-DD> <label>
"""
import csv, difflib, importlib.util, io, os, pathlib, re, subprocess, sys, time

src, night, label = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
ROOT = pathlib.Path.cwd()
ALLOWED = [r"research/batches/[A-Za-z0-9_.-]+\.json", r"research/out/lab_[A-Za-z0-9_.-]+\.(md|log)",
           r"research/lab_ledger\.csv", r"research/NIGHT_LOG\.md"]
MAX_BYTES = 5_000_000
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
SECRET = re.compile(r"sk-ant-|gh[pousr]_[A-Za-z0-9]{20}|github_pat_|x-access-token|AKIA[0-9A-Z]{16}|-----BEGIN")
LONGRUN = re.compile(r"[A-Za-z0-9_\-]{40,}")
problems = []
summary = os.environ.get("GITHUB_STEP_SUMMARY")


def say(msg):
    print(msg)
    if summary:
        with open(summary, "a") as fh: fh.write(msg + "\n\n")


def refuse(msg, secret=False):
    say(f"REFUSED, nothing committed: {msg}")
    if not secret and tonight_rows_raw:
        say("Tonight's ledger rows, not committed:\n```\n" + tonight_rows_raw + "\n```")
    if secret and os.environ.get("RUNNER_TEMP"):
        open(os.path.join(os.environ["RUNNER_TEMP"], "delete_artifact"), "w").close()
    marker("refused"); sys.exit(1)


tonight_rows_raw = ""


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def secret_shaped(text):
    if SECRET.search(text): return True
    for m in LONGRUN.finditer(text):          # long, upper AND lower AND digits: a key, not an id or a ruler
        t = m.group(0)
        if re.search(r"[A-Z]", t) and re.search(r"[a-z]", t) and re.search(r"\d", t) and len(set(t)) > 12:
            return True
    return False


def marker(kind):
    # The gate treats any 'night lab <night>' subject as "tonight is done", so a refused or empty
    # night still stops later triggers from starting another session.
    sh("git", "config", "user.name", "us-equity-bot")
    sh("git", "config", "user.email", "214131671+COffenberg@users.noreply.github.com")
    sh("git", "commit", "-q", "--allow-empty", "-m", f"night lab {label} ({kind})")
    for _ in range(3):
        if sh("git", "pull", "--rebase", "-q", "origin", "main", check=False).returncode == 0 \
                and sh("git", "push", "-q", check=False).returncode == 0:
            return
        sh("git", "rebase", "--abort", check=False); time.sleep(15)


# A queued run that started after tonight's session already committed has nothing to add.
if any(s == f"night lab {label}" or s.startswith(f"night lab {label} (") for s in
       sh("git", "log", "--since=20 hours ago", "--format=%s").stdout.splitlines()):
    say(f"night {label}: already committed by an earlier run; nothing to do"); sys.exit(0)

files = sorted(p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file() and not p.is_symlink()) \
    if src.exists() else []
cand = {}
for f in files:
    if not any(re.fullmatch(a, f) for a in ALLOWED):
        problems.append(f"not on the allowlist, left out: {f}"); continue
    new = (src/f).read_bytes()
    if len(new) > MAX_BYTES: refuse(f"{f} is {len(new):,} bytes")
    try: text = new.decode("utf-8")
    except UnicodeDecodeError: refuse(f"{f} is not UTF-8 text")
    old = (ROOT/f).read_bytes() if (ROOT/f).exists() else None
    if old == new: continue
    fresh = text[len(old.decode()):] if old is not None and new.startswith(old) else text
    if f == "research/NIGHT_LOG.md" and old is not None:      # prepended: scan only what was added
        fresh = "\n".join(l[2:] for l in difflib.ndiff(old.decode().splitlines(), text.splitlines()) if l.startswith("+ "))
    if EMAIL.search(fresh): refuse(f"{f} contains an email address; the repo is public", secret=True)
    if secret_shaped(fresh): refuse(f"{f} contains a secret-shaped string; the repo is public", secret=True)
    cand[f] = (old, new)

# --- the ledger: append-only, and parse what was added tonight
tonight_rows = []
if "research/lab_ledger.csv" in cand:
    old, new = cand["research/lab_ledger.csv"]
    if old is not None and not new.startswith(old): refuse("the ledger was not append-only")
    head = (old or new).decode().splitlines()[0]
    added = new[len(old):].decode() if old is not None else new.decode().split("\n", 1)[1]
    tonight_rows_raw = added.strip()
    tonight_rows = list(csv.DictReader(io.StringIO(head + "\n" + added)))
tonight_ids = sorted({r["batch_id"] for r in tonight_rows})

# --- batch specs: old ones frozen; each one ledgered tonight needs exactly one matching spec
_spec = importlib.util.spec_from_file_location("lab", ROOT/"research/lab.py")
lab = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lab)
specs = {}
for f, (old, new) in list(cand.items()):
    if not f.startswith("research/batches/"): continue
    if old is not None:
        problems.append(f"{f} already existed and was changed; registered batches are frozen, the change is left out")
        del cand[f]; continue
    tmp = ROOT/".night_spec_check.json"; tmp.write_bytes(new)
    try:
        B = lab.load_spec(tmp)
    except lab.LabError as e:
        say(f"{f}: not a valid batch ({e}); left out"); del cand[f]; continue
    finally:
        tmp.unlink()
    if B["batch_id"] != pathlib.Path(f).stem:
        problems.append(f"{f}: batch_id {B['batch_id']!r} does not match the file name")
    if B["batch_id"] not in tonight_ids:
        say(f"{f}: written but never run; left out"); del cand[f]; continue
    hashes = {r["spec_hash"] for r in tonight_rows if r["batch_id"] == B["batch_id"]}
    if hashes != {B["spec_hash"]}:
        problems.append(f"{f}: edited after it ran (ledger spec {sorted(hashes)}, file {B['spec_hash']}); "
                        f"left out, the registered text survives in research/out/lab_{B['batch_id']}.md")
        del cand[f]; continue
    specs.setdefault(B["batch_id"], []).append(f)
for bid in tonight_ids:
    if len(specs.get(bid, [])) != 1:
        problems.append(f"batch {bid!r} was ledgered tonight with {len(specs.get(bid, []))} matching spec files (need 1)")
# The lab enforces the night budget and one run per batch; check it again from the record.
prior_hashes = set()
if (ROOT/"research/lab_ledger.csv").exists():
    prior_hashes = {r["param_hash"] for r in csv.DictReader(open(ROOT/"research/lab_ledger.csv"))}
first_tonight = {}
for r in tonight_rows:
    if r["param_hash"] not in prior_hashes: first_tonight.setdefault(r["param_hash"], set()).add(r["batch_id"])
budget = int(os.environ.get("ARM_BUDGET", "12"))
if len(first_tonight) > budget:
    problems.append(f"{len(first_tonight)} new parameter sets tonight, over the budget of {budget}")
dup = sorted(h for h, b in first_tonight.items() if len(b) > 1)
if dup: problems.append(f"new parameter sets registered in more than one batch tonight: {dup}")
if len({(r['batch_id'], r['timestamp']) for r in tonight_rows}) != len(tonight_ids):
    problems.append("a batch was run more than once tonight")

# --- scoreboards: only for batches ledgered tonight, never an unledgered peek
for f, (old, new) in list(cand.items()):
    if not f.startswith("research/out/"): continue
    bid = re.fullmatch(r"research/out/lab_(.+)\.(md|log)", f).group(1)
    if bid not in tonight_ids:      # e.g. a run the lab stopped on coverage: by design, not a problem
        say(f"{f}: no ledger rows tonight for {bid!r}; left out"); del cand[f]; continue
    if f.endswith(".md") and "NOT LEDGERED" in new.decode():
        problems.append(f"{f}: an unledgered run; left out"); del cand[f]

# --- the notebook: newest first, so only additions; every ledgered batch must be in it
log_added = ""
if "research/NIGHT_LOG.md" in cand:
    old, new = cand["research/NIGHT_LOG.md"]
    diff = list(difflib.ndiff((old or b"").decode().splitlines(), new.decode().splitlines()))
    log_added = "\n".join(l[2:] for l in diff if l.startswith("+ "))
    if any(l.startswith("- ") for l in diff):
        # Keep what the agent added (it may hold a lead's registration); drop its edits to old lines.
        problems.append("research/NIGHT_LOG.md: existing lines were changed; kept only the added lines")
        cand["research/NIGHT_LOG.md"] = (old, (log_added + "\n\n" + (old or b"").decode()).encode())
missing = [b for b in tonight_ids if not re.search(rf"(?<![\w-]){re.escape(b)}(?![\w-])", log_added)]
if missing:
    problems.append(f"ledgered tonight but not in the agent's log entry: {', '.join(missing)}")

if not cand:
    say(f"night {label}: nothing new to commit"); marker("empty")
else:
    for f, (_, new) in cand.items():
        (ROOT/f).parent.mkdir(parents=True, exist_ok=True); (ROOT/f).write_bytes(new)
    if missing:   # make the omission visible where the morning check reads
        p = ROOT/"research/NIGHT_LOG.md"; cur = p.read_text() if p.exists() else ""
        p.write_text(f"> (auto, {night}) ledgered but not logged by the agent: {', '.join(missing)}\n\n" + cur)
        cand.setdefault("research/NIGHT_LOG.md", (None, b""))
    sh("git", "config", "user.name", "us-equity-bot")
    sh("git", "config", "user.email", "214131671+COffenberg@users.noreply.github.com")
    sh("git", "add", "--", *cand)
    sh("git", "commit", "-q", "-m", f"night lab {label}")
    for i in range(3):
        if sh("git", "pull", "--rebase", "-q", "origin", "main", check=False).returncode == 0 \
                and sh("git", "push", "-q", check=False).returncode == 0:
            break
        sh("git", "rebase", "--abort", check=False); time.sleep(15)
    else:
        refuse("push failed 3 times")
    say(f"night {label}: committed {len(cand)} files\n" + "\n".join(f"- {f}" for f in cand))

if tonight_rows:
    say("| batch | variant | role | d %/mo | d t | D pp | clears |\n|---|---|---|---|---|---|---|\n" + "\n".join(
        f"| {r['batch_id']} | {r['variant']} | {r['role']} | {r['d_mo']} | {r['d_t']} | {r['D_pp']} | {r['clears_bar']} |"
        for r in tonight_rows))

# The agent failing is a problem unless it was simply stopped at the deadline.
if os.environ.get("AGENT_OUTCOME") == "failure" and time.time() < int(os.environ.get("HARD_END", "0")) - 12*60:
    problems.append("the agent step failed before its deadline (token, usage limit, turn limit or error)")
if os.environ.get("LAB_RESULT") == "failure":
    problems.append("the lab job failed (setup, cache, harness check or collect)")
if problems:
    say("PROBLEMS:\n" + "\n".join(f"- {p}" for p in problems)); sys.exit(1)
