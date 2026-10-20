# F2 task 4: reference quality (pool2 + fresh5/fresh6)
import sys, os, pickle, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src"))
import data2
from pipeline import grid
res = os.path.join(R, "acuc", "results")
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    g, nl = grid(case); INST = data2.all_instances(g, case); base = g.pd.sum()
    sc = lambda i: INST[i][0].sum()/base
    te, va, tr = data2.split()
    P = data2.refs(case, os.path.join(res, "pool2"))
    print(f"\n===== {case}: pool2 has {len(P)} refs; split te {len(te)} va {len(va)} tr {len(tr)}")
    for nm, ids in (("test", te), ("val", va), ("train", tr)):
        have = [i for i in ids if i in P and P[i] is not None]
        conv = [i for i in have if P[i].get("converged")]
        print(f"  {nm}: refs {len(have)}/{len(ids)}, converged {len(conv)}; non-converged/missing at load "
              f"{np.round(sorted(sc(i) for i in ids if i not in conv), 3).tolist()[:30]}")
    # ext_grid unit 0 commitment rate in converged refs
    U = np.array([P[i]["u"] for i in P if P[i] is not None and P[i].get("converged")])
    print(f"  converged pool2 refs: {len(U)}; unit-0 (ext_grid) committed in {int(U[:,0].sum())}/{len(U)}; n_on median {np.median(U.sum(1)):.0f} [{U.sum(1).min()}-{U.sum(1).max()}]")
    if case.startswith("pglib"):
        real = g.pmax > 0
        print(f"  PGLib: condensers (pmax=0) committed fraction {U[:, ~real].mean():.3f}; real units committed {U[:, real].sum(1).mean():.1f}/{real.sum()}")
    sl = np.array([P[i].get("slack", np.nan) for i in P if P[i] is not None])
    mg = [P[i].get("mip_gaps", []) for i in P if P[i] is not None and P[i].get("converged")]
    lastg = np.array([m[-1] if len(m) else np.nan for m in mg])
    print(f"  pool2 slack quantiles {np.round(np.nanquantile(sl,[0,.5,.9,1]),5).tolist()}; final-iteration MIP gap of converged refs quantiles {np.round(np.nanquantile(lastg,[0,.5,.9,.99,1]),5).tolist()}")
    secs = np.array([P[i].get("secs", np.nan) for i in P if P[i] is not None and P[i].get("converged")]); its = np.array([P[i].get("iters", -1) for i in P if P[i] is not None and P[i].get("converged")])
    print(f"  secs median {np.nanmedian(secs):.0f} [p90 {np.nanquantile(secs,.9):.0f}], iters median {np.median(its):.0f} [max {its.max()}]")
    for draw in ("fresh5", "fresh6"):
        f = os.path.join(res, f"{draw}_refs_{case}.pkl")
        if not os.path.exists(f): continue
        F = pickle.load(open(f, "rb")); ids = getattr(data2, f"{draw}_ids")()
        conv = [i for i in ids if F.get(i, {}).get("converged")]
        print(f"  --- {draw}: {len(F)} refs, converged {len(conv)}/{len(ids)}; non-converged at load {np.round(sorted(sc(i) for i in ids if i not in conv),3).tolist()}")
        print(f"      statuses: { {s: sum(F[i].get('status', F[i].get('err'))==s for i in ids if i in F) for s in set(F[i].get('status', F[i].get('err')) for i in ids if i in F)} }")
        gb = np.array([F[i].get("gap_before", np.nan) for i in conv]); ga = np.array([F[i].get("gap_after", np.nan) for i in conv])
        ch = np.array([F[i].get("changed", False) for i in conv]); rej = [i for i in conv if "polish_rejected_slack" in F[i]]
        print(f"      MIP gap before polish quantiles {np.round(np.nanquantile(gb,[0,.5,.9,1]),5).tolist()}; after polish {np.round(np.nanquantile(ga,[0,.5,.9,1]),5).tolist()}; polish changed u on {int(ch.sum())}/{len(conv)}; polish rejected (slack) on {len(rej)}")
        # is the final kept solution's certified gap = gap_after only if polish kept? the polished solution's bound applies to the same MIQCQP
        nonimpr = [i for i in conv if not F[i].get("changed", False)]
        print(f"      max per-iteration MIP gap over all iterations (converged refs): median {np.median([np.nanmax(F[i]['mip_gaps'][:-1]) if len(F[i]['mip_gaps'])>1 else np.nan for i in conv]):.4f}, max {np.nanmax([np.nanmax(F[i]['mip_gaps'][:-1]) if len(F[i]['mip_gaps'])>1 else np.nan for i in conv]):.4f}")
        secs = np.array([F[i]["secs"] - F[i].get("secs_polish", 0) for i in conv]); sp = np.array([F[i].get("secs_polish", 0) for i in conv])
        print(f"      secs (excl. polish) median {np.median(secs):.0f} [min {secs.min():.0f}, max {secs.max():.0f}]; polish secs median {np.median(sp):.0f} max {sp.max():.0f}; iters median {np.median([F[i]['iters'] for i in conv]):.0f}")
        U = np.array([F[i]["u"] for i in conv]); print(f"      unit-0 committed {int(U[:,0].sum())}/{len(U)}; slack max {max(F[i]['slack'] for i in conv):.2e}")
