import sys, pickle, numpy as np
from scipy.stats import wilcoxon, binomtest
CASE = sys.argv[1]; SETS = sys.argv[2].split("+"); PAIRS = [p.split("-") for p in sys.argv[3].split(",")]
ST = [pickle.load(open(f"res_{CASE}_{s}.pkl","rb")) for s in SETS]
def col(arm):
    out = []
    for st in ST:
        SR = st["meta"]["SR"]
        for i in st["meta"]["scored"]:
            r = st["arms"][arm].get(i); J = SR[i][0]
            out.append(np.nan if r is None else 100*abs(r["cost"]-J)/J)
    return np.array(out)
def ids():
    return [(s, i) for s, st in zip(SETS, ST) for i in st["meta"]["scored"]]
for a, b in PAIRS:
    x, y = col(a), col(b); ok = ~np.isnan(x) & ~np.isnan(y); d = x[ok]-y[ok]; dt = np.where(np.abs(d) <= 1e-3, 0, d)
    bt, wo, ti = int((dt<0).sum()), int((dt>0).sum()), int((dt==0).sum())
    pw = wilcoxon(dt, zero_method="wilcox").pvalue if (dt!=0).any() else np.nan
    ps = binomtest(bt, bt+wo).pvalue if bt+wo else np.nan
    rng = np.random.default_rng(0); bs = [rng.choice(d, d.size).mean() for _ in range(10000)]
    # median of paired differences restricted to non-blow-up instances (both < 2%)
    nb = (x[ok] < 2) & (y[ok] < 2)
    print(f"{a:5s} - {b:5s}: mean {d.mean():+.4f} pp [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}], {a} better/tie/worse {bt}/{ti}/{wo}, Wilcoxon p {pw:.3g}, sign p {ps:.3g} | excl. blow-ups (n={nb.sum()}): mean diff {d[nb].mean():+.4f} pp | blow-ups >2%: {a} {int((x>2).sum())}, {b} {int((y>2).sum())}")
print("blow-up instances (>2%):")
I = ids()
for arm in sorted({p for pr in PAIRS for p in pr}):
    x = col(arm); print(f"  {arm}: " + ", ".join(f"{s}#{i}({x[k]:.1f}%)" for k, (s, i) in enumerate(I) if x[k] > 2))
