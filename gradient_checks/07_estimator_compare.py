# %% EXACT LAYER vs SMOOTHED ESTIMATOR, scored against the same reference.
#
# This is the one-variable comparison the project's own rule demands: both
# estimate d<gl,u_relax>/db for the SAME instance and the SAME loss. Only the
# ESTIMATOR differs. The reference is central finite differences, verified
# converged (h=1e-3 vs 3e-4) before anything is scored against it.
#
# The smoothed estimator is the one in 21_tiered_train.py:
#     g = (1/P) sum_p [L(b + s E_p) - L(b - s E_p)] / (2 s) * E_p
# so it costs 2P solves. The exact layer costs ONE solve plus one sparse LU.
import os, sys, time, json, warnings
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
CASE=os.environ.setdefault("QCAC_CASE","case118"); os.environ["OMP_NUM_THREADS"]="1"
warnings.filterwarnings("ignore")
import numpy as np
from pipeline import grid, sample
from socp import Socp
from conelayer import ConeLayer

N_INST=int(os.environ.get("N_INST","6"))
N_REP =int(os.environ.get("N_REP","8"))        # independent draws of E, for variance
SELFC_MAX=0.10
PAIRS=(1,2,4,8,16,32)
SIGMAS=(0.05,0.15,0.5,1.0)
REG={"case118":1e-8,"case300":1e-9}[CASE]      # chosen on the validation split in 06
ConeLayer.REG=REG
g,nl=grid(CASE); K=8; S=Socp(g,nl,n_cuts=K)
V1,V0=np.ones(g.n_bus),np.zeros(g.n_bus)
rng=np.random.default_rng(1)
A0=0.05*rng.normal(size=(K,2*g.n_gen))
print(f"[{CASE}] {g.n_bus} bus / {g.n_gen} gen | eps={REG:.0e} | "
      f"{N_INST} instances x {N_REP} repeats", flush=True)

rows=[]
for i,(pd_,qd_) in enumerate(sample(g,N_INST,seed=0)):
    r0=S.solve(pd_,qd_,V1,V0,rho=1e4,A=A0,b=np.full(K,1e3),cut_cap=0.0)
    if r0 is None: continue
    b0=A0@np.concatenate([r0["pg"],r0["u"]])-0.05
    r=S.solve(pd_,qd_,V1,V0,rho=1e4,A=A0,b=b0,cut_cap=0.0)
    if r is None: continue
    u_ref=r["u"].copy(); gl=rng.normal(size=g.n_gen)
    def L(bb):
        rr=S.solve(pd_,qd_,V1,V0,rho=1e4,A=A0,b=bb,cut_cap=0.0)
        return np.nan if rr is None else float(gl@rr["u"])
    def fdv(h):
        return np.array([(L(np.where(np.arange(K)==j,b0+h,b0))
                          -L(np.where(np.arange(K)==j,b0-h,b0)))/(2*h) for j in range(K)])
    f1,f2=fdv(1e-3),fdv(3e-4)
    if not (np.all(np.isfinite(f1)) and np.all(np.isfinite(f2))): continue
    ref=0.5*(f1+f2); selfc=np.abs(f1-f2).max()/max(1e-12,np.abs(ref).max())
    if selfc>SELFC_MAX:
        print(f"inst{i}: reference not converged (selfc {selfc:.3f}) -- skipped",flush=True)
        continue
    cos=lambda v: float(v@ref/max(1e-30,np.linalg.norm(v))/np.linalg.norm(ref))

    # ---- exact layer: ONE solve + one LU ----
    t0=time.time()
    lay=ConeLayer(S.prob,[S.p_b],[S.v["u"]]); lay.build_structure([b0])
    x,st=lay.forward([b0],tol=1e-12)
    uidx=[int(np.argmin(np.abs(x-v))) for v in u_ref]
    dL=np.zeros(x.size); dL[uidx]=gl
    g_ex=lay.vjp(st,dL)[0]; t_ex=time.time()-t0
    c_ex=cos(g_ex)

    # ---- smoothed estimator: 2P solves per draw ----
    sm={}
    for s_ in SIGMAS:
        for P in PAIRS:
            cs=[]
            for rep in range(N_REP):
                rr=np.random.default_rng(1000*i+10*rep+int(s_*100))
                acc=np.zeros(K); good=0
                for p in range(P):
                    E=rr.standard_normal(K)
                    lp,lm=L(b0+s_*E),L(b0-s_*E)
                    if np.isfinite(lp) and np.isfinite(lm):
                        acc+=(lp-lm)/(2*s_)*E; good+=1
                if good: cs.append(cos(acc/good))
            if cs: sm[(s_,P)]=(float(np.mean(cs)),float(np.std(cs)))
    rows.append(dict(i=i,selfc=selfc,exact=c_ex,t_exact=t_ex,sm=sm))
    print(f"inst{i:2d} selfc {selfc:.3f}  EXACT cos {c_ex:+.4f} ({t_ex:.2f}s)",flush=True)

print(f"\n{'='*78}\n{CASE}: mean cosine vs the converged reference, over {len(rows)} instances")
print(f"\nEXACT LAYER   (1 solve + 1 LU):  cosine {np.mean([r['exact'] for r in rows]):+.4f}"
      f"   min {np.min([r['exact'] for r in rows]):+.4f}")
print(f"\nSMOOTHED ESTIMATOR (2P solves), mean +- sd over {N_REP} independent draws of E:")
print(f"{'sigma':>7s}" + "".join(f"{'P='+str(P):>16s}" for P in PAIRS))
print(f"{'':>7s}" + "".join(f"{'('+str(2*P)+' solves)':>16s}" for P in PAIRS))
best_sm=-9
for s_ in SIGMAS:
    cells=[]
    for P in PAIRS:
        v=[r["sm"].get((s_,P)) for r in rows if (s_,P) in r["sm"]]
        if not v: cells.append(f"{'-':>16s}"); continue
        m=float(np.mean([a[0] for a in v])); sd=float(np.mean([a[1] for a in v]))
        best_sm=max(best_sm,m)
        cells.append(f"{m:+.3f}+-{sd:.3f}".rjust(16))
    print(f"{s_:7.2f}" + "".join(cells))
print(f"\nbest smoothed cosine anywhere in the sweep: {best_sm:+.4f}"
      f"   (exact layer: {np.mean([r['exact'] for r in rows]):+.4f})")
json.dump([{k:(v if k!="sm" else {f"{a}_{b}":c for (a,b),c in v.items()})
            for k,v in r.items()} for r in rows],
          open(f"results/estimator_{CASE}.json","w"),indent=1)
print(f"wrote results/estimator_{CASE}.json")
