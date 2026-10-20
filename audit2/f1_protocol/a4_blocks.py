import os, sys, json, pickle, numpy as np, warnings
warnings.filterwarnings("ignore")
import os; R = os.environ.get("REPO") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, R + "/src")
import data2
from pipeline import grid
te, va, tr = data2.split()
blocks = dict(test=te, val=va, train=tr, fresh=data2.fresh_ids(), fresh2=data2.fresh2_ids(), fresh3=data2.fresh3_ids(),
              extra=data2.extra_ids(), fresh4=data2.fresh4_ids(), fresh5=data2.fresh5_ids(), band=data2.band_ids(), fresh6=data2.fresh6_ids())
out = {}
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    os.environ["QCAC_CASE"] = case
    g, nl = grid(case)
    INST = data2.all_instances(g, case)
    N = len(INST)
    allidx = sorted(set(sum(blocks.values(), [])))
    print(f"===== {case}: {N} instances in all_instances, max block index {max(allidx)}, union of blocks {len(allidx)}; contiguous {allidx == list(range(N))}")
    # pairwise block overlap of indices
    names = list(blocks)
    for a in range(len(names)):
        for b in range(a+1, len(names)):
            ov = set(blocks[names[a]]) & set(blocks[names[b]])
            if ov: print(f"   INDEX OVERLAP {names[a]} x {names[b]}: {len(ov)}")
    base = float(g.pd.sum())
    X = np.stack([np.r_[p, q]/base for p, q in INST])
    sc = np.array([p.sum()/g.pd.sum() for p, q in INST])
    # duplicates / near duplicates: nearest-neighbour distance across blocks in normalized feature space
    from scipy.spatial import cKDTree
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sd
    tree = cKDTree(Xn)
    d, j = tree.query(Xn, k=2)
    nn = d[:, 1]
    print(f"   nearest-neighbour distance (normalised features, dim {X.shape[1]}): min {nn.min():.3f}, 1% {np.percentile(nn,1):.3f}, median {np.median(nn):.3f}; exact duplicates {(nn < 1e-9).sum()}")
    # for each eval block, min distance to any training/extra draw
    trainish = tr + data2.extra_ids()
    t2 = cKDTree(Xn[trainish])
    for nm in ("val", "test", "fresh4", "fresh5", "fresh6"):
        dd, _ = t2.query(Xn[blocks[nm]], k=1)
        dtt, _ = cKDTree(Xn[[i for i in trainish]]).query(Xn[trainish], k=2)
        print(f"   {nm:7s}: load scale [{sc[blocks[nm]].min():.3f}, {sc[blocks[nm]].max():.3f}] mean {sc[blocks[nm]].mean():.3f}; min/median dist to train+extra {dd.min():.3f}/{np.median(dd):.3f} (train+extra self-NN median {np.median(dtt[:,1]):.3f})")
    for nm in ("train", "extra", "band"):
        print(f"   {nm:7s}: load scale [{sc[blocks[nm]].min():.3f}, {sc[blocks[nm]].max():.3f}] mean {sc[blocks[nm]].mean():.3f}, n={len(blocks[nm])}")
    # KS test extra vs test-like blocks
    from scipy.stats import ks_2samp
    tests = blocks["fresh5"] + blocks["fresh6"]
    print(f"   KS(load scale) extra vs fresh5+fresh6: p={ks_2samp(sc[blocks['extra']], sc[tests]).pvalue:.3f}; train vs extra p={ks_2samp(sc[blocks['train']], sc[blocks['extra']]).pvalue:.3f}")
    out[case] = sc
    if case.startswith("pglib"):
        RP = data2.refs(case, R + "/acuc/results/pool2")
        T = {int(k) for k in json.load(open(f"{R}/acuc/results/selfteacher_{case}_bestof_train_nc.json"))}
        miss = [i for i in tr if i not in T]
        print(f"   PGLib train draws without own target / ref: load scale {np.round(np.sort(sc[miss]),3).tolist()}")
        print(f"   PGLib train draws WITH target: load scale max {sc[[i for i in tr if i in T]].max():.3f}")
    if case == "case300":
        T = {int(k) for k in json.load(open(f"{R}/acuc/results/selfteacher_{case}_bestof_train_nc.json"))}
        miss = [i for i in tr if i not in T]
        print(f"   case300 train draws without own target: load scale {np.round(np.sort(sc[miss]),3).tolist()}")
pickle.dump(out, open("loadscales.pkl", "wb"))
