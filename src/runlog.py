"""Every run gets its own folder acuc/runs/<YYYY-MM-DD_HHMM>_<name>/ with: the exact command, the code version
(git HEAD + hash of the uncommitted diff + a copy of the script), all outputs saved via save(), and the log.
One line per run is appended to acuc/RESULTS.md by add_result()."""
import os, sys, time, json, shutil, hashlib, subprocess, pickle

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def start(name):
    d = os.path.join(ROOT, "acuc", "runs", time.strftime("%Y-%m-%d_%H%M") + "_" + name); os.makedirs(d, exist_ok=True)
    head = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", ROOT, "diff"], capture_output=True, text=True).stdout
    json.dump(dict(cmd=" ".join(sys.argv), git_head=head, diff_sha1=hashlib.sha1(diff.encode()).hexdigest()[:12],
                   started=time.strftime("%Y-%m-%d %H:%M:%S")), open(os.path.join(d, "run.json"), "w"), indent=1)
    shutil.copy(os.path.abspath(sys.argv[0]), d)
    return d


def save(d, name, obj):
    pickle.dump(obj, open(os.path.join(d, name), "wb"))


def add_result(d, line):
    with open(os.path.join(ROOT, "acuc", "RESULTS.md"), "a") as f:
        f.write(f"| {os.path.basename(d)} | {line} |\n")
