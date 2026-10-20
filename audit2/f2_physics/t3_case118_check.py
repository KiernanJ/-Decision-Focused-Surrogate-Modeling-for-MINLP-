# F2 task 3b: case118 fresh6 (repro final.pkl): raw-voltage feasibility of strictly priced points + PYPOWER AC-OPF at the same commitments
import sys, os, pickle, time, numpy as np, warnings; warnings.filterwarnings("ignore")
R, case, fin = sys.argv[1], sys.argv[2], sys.argv[3]; sys.path.insert(0, os.path.join(R, "src")); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data2
from pipeline import grid
from evalac import opf_cost, check_point
g, nl = grid(case); INST = data2.all_instances(g, case)
F = pickle.load(open(fin, "rb")); SR, tid = F["SR"], F["tid"]; r0 = F["rows"][0]
out = {}; t0 = time.time()
for i in tid:
    p, q = INST[i]
    J, pg, vr, vi, nrep, uref = SR[i]
    chk = check_point(g, p, q, uref, vr, vi, pg)
    rec = dict(chk=chk, J=J)
    for nm, u, Jm in (("ref", np.asarray(uref), J), ("ours", r0["OP"][i]["u"], r0["OP"][i]["cost"]), ("proxy", r0["PX"][i]["u"], r0["PX"][i]["cost"])):
        ok, Jo, _ = opf_cost(g, nl, p, q, np.asarray(u, float))
        rec[nm] = (Jm, ok, Jo)
    out[i] = rec
    print(f"{i}: chk ref {', '.join(f'{k} {v:.1e}' for k, v in chk.items())} | " + " | ".join(f"{nm} strict {rec[nm][0]:.2f} opf {'ok' if rec[nm][1] else 'FAIL'} {rec[nm][2]:.2f} ({100*(rec[nm][0]-rec[nm][2])/rec[nm][2]:+.4f}%)" for nm in ("ref","ours","proxy")) + f" [{time.time()-t0:.0f}s]", flush=True)
pickle.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"t3_{case}_{os.path.basename(os.path.dirname(fin))[:40]}.pkl"), "wb"))
d = {nm: np.array([100*(out[i][nm][0]-out[i][nm][2])/out[i][nm][2] for i in tid if out[i][nm][1]]) for nm in ("ref","ours","proxy")}
for nm, v in d.items(): print(f"[{nm}] strict minus PYPOWER-OPF (%): n={len(v)}, mean {v.mean():+.4f}, median {np.median(v):+.4f}, min {v.min():+.4f}, max {v.max():+.4f}; OPF failures {sum(not out[i][nm][1] for i in tid)}")
mx = {k: max(out[i]['chk'][k] for i in tid) for k in out[tid[0]]['chk']}; print("[ref priced points] worst raw-voltage checks:", mx)
# signed gaps under both evaluators
for nm in ("ours", "proxy"):
    s1 = np.array([100*(out[i][nm][0]-out[i]['ref'][0])/out[i]['ref'][0] for i in tid])
    ok = [k for k, i in enumerate(tid) if out[i][nm][1] and out[i]['ref'][1]]
    s2 = np.array([100*(out[tid[k]][nm][2]-out[tid[k]]['ref'][2])/out[tid[k]]['ref'][2] for k in ok])
    print(f"[{nm} vs ref] signed gap strict: mean {s1.mean():+.4f}% (|gap| {np.abs(s1).mean():.4f}%) ; PYPOWER-OPF (n={len(ok)}): mean {s2.mean():+.4f}% (|gap| {np.abs(s2).mean():.4f}%); sign agreement {np.mean(np.sign(np.round(s1[ok],4))==np.sign(np.round(s2,4))):.2f}")
