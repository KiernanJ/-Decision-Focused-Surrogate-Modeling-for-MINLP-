"""Persist the reference timings incrementally, and report convergence.

30_reftime.py writes its JSON only after the last solve, so a kill at 14/15
loses every measurement -- and on case300 that is over an hour of solves. The
per-solve times ARE printed, so this tails the log and checkpoints them after
each one. Nothing needs re-running if the job stops.

It also reports whether the median has CONVERGED: the running median is
recomputed as samples arrive, and what matters is that it stops moving. A
4-sample median gave 120.4s and 345.1s for the same quantity; this shows
directly when adding samples has stopped changing the answer.
"""
import json, os, re, sys, time
import numpy as np

CASE = sys.argv[1] if len(sys.argv) > 1 else "case300"
LOG = f"logs/reftime{CASE.replace('case','')}.log"
OUT = f"results/reftime_{CASE}_partial.json"
pat = re.compile(r"^\s+(\d+)/(\d+)\s+([\d.]+)s")
seen = 0
while True:
    if os.path.exists(LOG):
        T = [float(m.group(3)) for m in
             (pat.match(l) for l in open(LOG, errors="ignore")) if m]
        if len(T) > seen:
            seen = len(T)
            a = np.array(T)
            rec = dict(case=CASE, n=len(T), times=list(map(float, a)),
                       median=float(np.median(a)), mean=float(a.mean()),
                       q25=float(np.percentile(a, 25)),
                       q75=float(np.percentile(a, 75)))
            # convergence: how much the running median still moves
            meds = [float(np.median(a[:k])) for k in range(3, len(a)+1)]
            if len(meds) >= 3:
                rec["median_last3"] = meds[-3:]
                rec["drift_pct"] = float(100*abs(meds[-1]-meds[-3])/max(meds[-1], 1e-9))
            json.dump(rec, open(OUT, "w"), indent=1)
            msg = f"[{time.strftime('%H:%M:%S')}] {CASE} n={len(T):2d} " \
                  f"median {rec['median']:7.1f}s  IQR {rec['q25']:.0f}-{rec['q75']:.0f}s"
            if "drift_pct" in rec:
                msg += f"  median drift over last 3 samples: {rec['drift_pct']:5.1f}%"
            print(msg, flush=True)
    if not os.popen("pgrep -f 30_reftime.py").read().strip() and seen > 0:
        print(f"  job finished; {seen} samples checkpointed to {OUT}", flush=True)
        break
    time.sleep(20)
