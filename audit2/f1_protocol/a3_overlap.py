import os, sys, json, numpy as np, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src")
import data2
te, va, tr = data2.split()
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    RP = data2.refs(case, R + "/acuc/results/pool2")
    conv = {i for i, r in RP.items() if data2.converged(r)}
    T = {int(k) for k in json.load(open(f"{R}/acuc/results/selfteacher_{case}_bestof_train_nc.json"))}
    trs = set(tr)
    noT = trs - T; noR = trs - conv
    print(f"{case}: train w/o own target {len(noT)}, train w/o converged ref {len(noR)}, both {len(noT & noR)}, own-target-but-no-ref {len(T & noR)}, ref-but-no-own-target {len(conv & trs - T)}")
    # agreement of own targets with references on train where both exist
    TT = {int(k): np.array(v) for k, v in json.load(open(f"{R}/acuc/results/selfteacher_{case}_bestof_train_nc.json")).items()}
    both = sorted(T & conv & trs)
    dis = np.mean([np.mean(TT[i] != np.asarray(RP[i]["u"])) for i in both])
    exact = np.mean([np.all(TT[i] == np.asarray(RP[i]["u"])) for i in both])
    print(f"   own target vs reference commitment on {len(both)} train draws: unit disagreement {100*dis:.2f}%, identical commitments {100*exact:.1f}%")
