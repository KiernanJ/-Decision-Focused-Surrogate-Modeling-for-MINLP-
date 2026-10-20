# A: how much did layer training change the per-unit fixings? (warm start = exact BCE re-creation)
import sys, numpy as np, torch
from common import *
case = sys.argv[1]
D = setup(case); data2 = D["data2"]; G = D["G"]
cw, TG, ids = warm_cutnet(D, case); cw.eval()
cd, UC = deployed_cutnet(D, case); c = UC["c"]
sw, sd = cw.state_dict(), UC["state"]
mx = max(float((sw[k]-sd[k]).abs().max()) for k in sw); rel = {k: round(float((sw[k]-sd[k]).norm()/sd[k].norm()), 4) for k in sw}
nsteps = (UC["epoch"]+1)*int(np.ceil(len(ids)/24))
print(f"[{case}] c={c} deployed epoch {UC['epoch']} (~{nsteps} Adam steps at lr 1e-4 => max |dtheta| bound ~{nsteps*1e-4:.4f})")
print(f"  warm(re-created) vs deployed: max |dtheta| {mx:.5f}; rel norm diff per tensor {rel}")
print(f"  label-free val: warm (v_cur) {UC['v_cur']:.6f}  deployed {UC['val']:.6f}  -> deployed {'WORSE' if UC['val'] > UC['v_cur'] else 'better'} by {UC['val']-UC['v_cur']:+.6f}")
sets = {"train(own targets)": ids, "validation": D["va"], "fresh5": data2.fresh5_ids()}
if case == "case118": sets["fresh6"] = data2.fresh6_ids()
with torch.no_grad():
    Mw = torch.sigmoid(cw(D["Xn"])).numpy(); Md = torch.sigmoid(cd(D["Xn"])).numpy()
for nm, S in sets.items():
    S = list(S)
    onw, offw = Mw[S] > c, Mw[S] < 1-c; ond, offd = Md[S] > c, Md[S] < 1-c
    fixw = onw | offw; fixd = ond | offd
    chg = (onw != ond) | (offw != offd)
    inst_chg = chg.any(1)
    extra = ""
    if nm.startswith("train"):
        Y = np.stack([TG[i] for i in S])
        wrongw = (onw & (Y < .5)) | (offw & (Y > .5)); wrongd = (ond & (Y < .5)) | (offd & (Y > .5))
        extra = f" | fixings contradicting own target: warm {wrongw.sum()} deployed {wrongd.sum()}"
    print(f"  {nm:20s} n={len(S):3d}: units fixed warm {fixw.sum(1).mean():.2f}/{G}, deployed {fixd.sum(1).mean():.2f}/{G} | (inst,unit) fixings changed {chg.sum()} of {len(S)*G} ({100*chg.mean():.3f}%) on {inst_chg.sum()} instances; "
          f"added {(fixd & ~fixw).sum()}, removed {(fixw & ~fixd).sum()}, flipped on<->off {((onw & offd) | (offw & ond)).sum()} | max |dM| {np.abs(Mw[S]-Md[S]).max():.4f}, mean |dM| {np.abs(Mw[S]-Md[S]).mean():.5f}{extra}")
