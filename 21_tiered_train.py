"""Learned TIERED cardinality cuts, with the falsifying control built in.

The previous cut was decorative: soft cuts meant a violated band's right-hand
side dropped out of the optimality conditions, so a dummy band [0,0] reproduced
the learned band bit for bit. Two changes:

  HARD cuts (cut_cap=0). The band now determines the commitment; verified
  respected 8/8 where the soft version was 0/8.

  TIERED counts. Generators are split into K merit-order tiers and the network
  predicts how many run in each, so the cut shapes WHICH units, not just how
  many. A constant cannot imitate it -- tier counts move differently per
  instance. Oracle ceiling at moderate V error: K=3 reaches 0.184% against
  K=1's 0.489% and 0.923% with no cut.

  K=3 is deliberate, not tuned upward: the ceiling keeps improving with K
  (K=8 -> 2.39% discrete) but fragility grows faster. At a realistic 0.5-unit
  prediction error K=3 holds 0.303% while K=5 collapses to 0.802%.

EVERY run ends with the dummy-band control: the same net evaluated with the
band replaced by [0,0] and by a random band. If those match the learned band,
the cut is decorative and the run reports it rather than claiming a result.
"""
# %% imports
import os, sys, time, json, argparse, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src")); sys.path.insert(0, HERE)
from pipeline import CASE, RESERVE, grid, sample, set_cuts, set_down, _init, _dep
from socp import Socp


# %% network: V head + K tier-count heads
class TieredNet(torch.nn.Module):
    def __init__(self, n_bus, n_gen, tier_sizes, hidden=256, v_scale=0.5,
                 init_frac=0.25, learn_thr=True):
        super().__init__()
        self.n_bus, self.n_gen, self.v_scale = n_bus, n_gen, v_scale
        self.K = len(tier_sizes)
        self.register_buffer("tsz", torch.tensor(tier_sizes, dtype=torch.float32))
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(2*n_bus, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU())
        self.head_v = torch.nn.Linear(hidden, 2*n_bus)
        self.head_t = torch.nn.Linear(hidden, self.K)
        # per-generator rounding threshold, initialised at 0.5 so training
        # starts exactly at the current fixed-0.5 behaviour
        self.learn_thr = learn_thr
        self.head_thr = torch.nn.Linear(hidden, n_gen)
        torch.nn.init.zeros_(self.head_thr.weight)
        torch.nn.init.zeros_(self.head_thr.bias)
        torch.nn.init.zeros_(self.head_v.weight); torch.nn.init.zeros_(self.head_v.bias)
        torch.nn.init.normal_(self.head_t.weight, std=1e-3)
        # Start each tier count at a fraction of its size, i.e. BINDING. A cut
        # that starts inactive has zero derivative w.r.t. its own coefficients
        # and never trains -- that mistake cost a 50-epoch run.
        with torch.no_grad():
            self.head_t.bias.fill_(float(np.log(init_frac/(1-init_frac))))

    def forward(self, x):
        h = self.trunk(x)
        dv = self.v_scale*torch.tanh(self.head_v(h))
        Vr = 1.0 + dv[:, :self.n_bus]
        Vi = 0.0 + dv[:, self.n_bus:]
        cnt = self.tsz[None, :]*torch.sigmoid(self.head_t(h))   # per-tier count
        thr = torch.sigmoid(self.head_thr(h))                   # in (0,1), starts 0.5
        return (Vr, Vi, cnt, thr) if self.learn_thr else (Vr, Vi, cnt)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def tier_rows(cnt, TIERS, n_gen, K_rows):
    """Hard equality-style band per tier: sum_{i in T} u_i == cnt_T."""
    A, b = np.zeros((K_rows, 2*n_gen)), np.ones(K_rows)*1e3
    k = 0
    for T, c in zip(TIERS, cnt):
        A[k, n_gen + T] = -1.0; b[k] = -float(c); k += 1
        A[k, n_gen + T] = 1.0;  b[k] = float(c);  k += 1
    return A, b


# %% main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", type=int, default=3)
    # How the fleet is partitioned into tiers. The original code used
    # MERIT[i::K] ("interleaved"), which is NOT a merit-order tier -- each slice
    # spans the whole cost range (3639-9503, 3691-19879, 3706-32200), i.e. three
    # near-identical arbitrary thirds. "block" is the contiguous merit ordering
    # the name implies; "random" is the CONTROL: if a random partition matches
    # the others, the gain is simply K constraints instead of 1 and the
    # partition carries no meaning.
    ap.add_argument("--partition", choices=["interleaved", "block", "random"],
                    default="interleaved")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sigma", type=float, default=0.03)
    ap.add_argument("--sigma-tier", type=float, default=0.6)
    ap.add_argument("--sigma-thr", type=float, default=0.08)  # thresholds are in (0,1)
    ap.add_argument("--learn-thr", type=int, default=1)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--n-test", type=int, default=48)
    ap.add_argument("--down", type=int, default=6)       # downward trials at EVAL
    # Training cost is dominated by restoration, not by the relaxation: each
    # deployment runs a price() loop of up to 25 solves plus `down` more. With
    # 144 instances x 5 perturbations that is ~5-20k SOCP solves an epoch, i.e.
    # 5 minutes. --down-train 0 and a minibatch cut that to seconds; the
    # question being asked here -- does the band vary per instance -- does not
    # need the full restoration in the loop, and EVAL still uses --down.
    ap.add_argument("--down-train", type=int, default=0)
    ap.add_argument("--batch", type=int, default=0)       # 0 = all train
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    # Fraction of each tier's size the counts start at. Must start BINDING but
    # within reach: case118 commits ~27% of its fleet, case300 ~75%, so one
    # value cannot serve both. Too low on case300 means a 35-unit climb; too
    # high means the cut never binds and gets zero gradient.
    ap.add_argument("--init-frac", type=float, default=0.25)
    ap.add_argument("--tag", default="tier")
    a = ap.parse_args()
    import multiprocessing as mp
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    K_ROWS = 2*a.tiers
    set_cuts(K_ROWS); set_down(a.down_train)
    g, nl = grid(); NG = g.n_gen

    POOL = []
    for f in sorted(os.listdir(f"{HERE}/results")):
        m = re.fullmatch(rf"ref_{CASE}_s(\d+)_n(\d+)\.npz", f)
        if not m:
            continue
        REF = list(np.load(f"{HERE}/results/{f}", allow_pickle=True)["ref"])
        inst = sample(g, int(m.group(2)), seed=int(m.group(1)))
        POOL += [(inst[i][0], inst[i][1], REF[i]["u"], REF[i]["cost"])
                 for i in range(int(m.group(2))) if REF[i] is not None]
    perm = rng.permutation(len(POOL))
    te = [int(i) for i in perm[:a.n_test]]; tr = [int(i) for i in perm[a.n_test:]]
    avg = (g.c2*g.pmax**2 + g.c1*g.pmax + nl)/np.maximum(g.pmax, 1e-6)
    MERIT = np.argsort(avg)
    if a.partition == "interleaved":
        TIERS = [MERIT[i::a.tiers] for i in range(a.tiers)]
    elif a.partition == "block":
        cut = np.array_split(MERIT, a.tiers)
        TIERS = [np.asarray(c) for c in cut]
    else:
        shuf = np.random.default_rng(1234).permutation(NG)
        TIERS = [np.asarray(c) for c in np.array_split(shuf, a.tiers)]
    base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]), dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6; Xn = (X - mu)/sg
    print(f"[setup] {CASE}: {len(POOL)} inst, {len(tr)} train / {len(te)} test | "
          f"K={a.tiers} {a.partition} tiers, sizes {[len(T) for T in TIERS]} | "
          f"HARD cuts",
          flush=True)

    S = Socp(g, nl, n_cuts=K_ROWS)
    SCALE = np.zeros(len(POOL))
    for j in range(len(POOL)):
        r = S.solve(POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus), rho=1e6)
        SCALE[j] = float(r["cost"]) if r else 1e5

    # workers must exist before the pool forks
    def jobs(idx, Vr, Vi, CNT):
        out = []
        for k, j in enumerate(idx):
            A, b = tier_rows(CNT[k], TIERS, NG, K_ROWS)
            out.append((k, POOL[j][0], POOL[j][1], Vr[k], Vi[k], A, b))
        return out
    pool = mp.get_context("fork").Pool(a.procs, initializer=_init)

    # Band init WITHOUT a per-case constant. n*/n_min measured 1.97 on case118
    # and 2.24 on case300, so a band starting at 1.6*n_min binds on both while
    # staying reachable. n_min is a sort and a cumsum -- no labels.
    #
    # NOTE the parameterisation: TieredNet computes cnt = tier_size * sigmoid(.),
    # so init_frac is a fraction of TIER SIZE, not of the span above n_min.
    # Deriving it as 0.6*n_min/(NG-n_min) -- correct for a span parameterisation
    # -- started case118's band at 54*0.097 = 5.2 against n* = 14.7. The cut then
    # forced heavy under-commitment, restoration repaired upward every time, and
    # the band's position stopped mattering: the mean-band control matched the
    # learned band to three decimals on BOTH cases.
    _order = np.argsort(-g.pmax); _cum = np.cumsum(g.pmax[_order])
    _nmin = np.mean([float(np.searchsorted(_cum, (1.0+RESERVE)*float(POOL[j][0].sum()))+1)
                     for j in tr])
    _frac = float(np.clip(1.6*_nmin/NG, 0.02, 0.95))
    print(f"[init ] mean n_min {_nmin:.1f} -> band starts at {_frac*NG:.1f} "
          f"of {NG} (init_frac {_frac:.3f}, label-free)", flush=True)
    net = TieredNet(g.n_bus, NG, [len(T) for T in TIERS], init_frac=_frac,
                    learn_thr=bool(a.learn_thr))
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print(f"[net ] {net.n_params():,} params | outputs {2*g.n_bus + a.tiers}",
          flush=True)

    def evaluate(idx, cnt_of):
        J = []
        for k, j in enumerate(idx):
            out = cnt_of(k, j)
            vr_, vi_, c_ = out[0], out[1], out[2]
            th_ = out[3] if len(out) > 3 else None
            A, b = (tier_rows(c_, TIERS, NG, K_ROWS) if c_ is not None else (None, None))
            J.append((j, POOL[j][0], POOL[j][1], vr_, vi_, A, b, th_))
        out = {i: (c, u, s) for i, c, u, s in pool.imap_unordered(_dep, J)}
        G_, D_, n = [], [], 0
        for j in idx:
            c, u, s = out[j]
            if c >= 1e8 or u is None:
                continue
            n += 1
            G_.append(100*(c-POOL[j][3])/POOL[j][3])
            D_.append(100*float((u != POOL[j][2]).mean()))
        return (np.mean(G_) if G_ else np.nan, np.mean(D_) if D_ else np.nan, n)

    flat = lambda k, j: (np.ones(g.n_bus), np.zeros(g.n_bus), None, None)
    bg, bd, bn = evaluate(te, flat)
    print(f"[BASE] flat V, no cut: gap {bg:+.3f}%  discrete {bd:.2f}%  n={bn}\n",
          flush=True)

    best = None
    for ep in range(a.epochs):
        t0 = time.time()
        bidx = [int(i) for i in rng.permutation(tr)]
        if a.batch:
            bidx = bidx[:a.batch]
        out = net(Xn[bidx])
        vr, vi, cnt = out[0], out[1], out[2]
        thr = out[3] if len(out) > 3 else None
        Y = torch.cat([vr, vi, cnt] + ([thr] if thr is not None else []), 1)
        Yd = Y.detach().numpy(); nb, dY = Y.shape
        # Per-GROUP perturbation scale. The three output groups live on
        # different scales -- voltages O(0.1), tier counts in GENERATORS O(10),
        # thresholds in (0,1) -- and a single sigma leaves whichever group it
        # does not match with no usable gradient.
        sig = np.full(dY, a.sigma)
        if thr is not None:
            sig[2*g.n_bus:2*g.n_bus+a.tiers] = a.sigma_tier
            sig[-NG:] = a.sigma_thr
        else:
            sig[-a.tiers:] = a.sigma_tier
        E = rng.standard_normal((a.pairs, nb, dY))
        nb_, J, key = g.n_bus, [], []
        for k, j in enumerate(bidx):
            for tag, yv in ([("0", Yd[k])] +
                            [(f"{s}{p}", Yd[k] + sgn*sig*E[p, k])
                             for p in range(a.pairs) for s, sgn in (("+", 1), ("-", -1))]):
                c_ = np.clip(yv[2*nb_:2*nb_+a.tiers], 0, [len(T) for T in TIERS])
                th_ = np.clip(yv[2*nb_+a.tiers:], 0.02, 0.98) if thr is not None else None
                A, b = tier_rows(c_, TIERS, NG, K_ROWS)
                J.append((k, POOL[j][0], POOL[j][1], yv[:nb_], yv[nb_:2*nb_], A, b, th_))
                key.append((tag, k))
        res = list(pool.imap(_dep, J))
        C = np.array([r[1] for r in res], float)
        sc = SCALE[bidx]
        Gest = np.zeros((nb, dY)); base_c = np.full(nb, np.nan)
        for t, (tag, k) in enumerate(key):
            if tag == "0":
                base_c[k] = C[t]
            elif tag.startswith("+"):
                p = int(tag[1:]); cp_, cm_ = C[t], C[t+1]
                if cp_ < 1e8 and cm_ < 1e8:
                    Gest[k] += (cp_-cm_)/(2*sc[k])*(E[p, k]/sig)/a.pairs
        loss = (Y*torch.tensor(Gest, dtype=torch.float32)).sum()/nb
        opt.zero_grad(); loss.backward(); opt.step()
        ok = base_c < 1e8
        sel = float(np.mean(base_c[ok]/sc[ok])) if ok.any() else np.inf
        if best is None or sel < best[0]:
            best = (sel, {k: v.clone() for k, v in net.state_dict().items()})
        if ep % a.log_every == 0 or ep == a.epochs-1:
            refc = np.array([POOL[j][3] for j in bidx])
            gp_ = 100*float(np.mean((base_c[ok]-refc[ok])/refc[ok]))
            print(f"[ep {ep:3d}] train gap {gp_:+7.3f}%  fail {int((~ok).sum())}/{nb}"
                  f"  {time.time()-t0:5.1f}s  counts {cnt.mean(0).detach().numpy().round(1)}"
                  f" sd {cnt.std(0).detach().numpy().round(2)}",
                  flush=True)
    net.load_state_dict(best[1]); net.eval()
    set_down(a.down)          # full restoration for the held-out evaluation
    with torch.no_grad():
        _o = net(Xn[te])
    VR, VI, CNT = [_o[i].numpy().astype(float) for i in range(3)]
    THR = _o[3].numpy().astype(float) if len(_o) > 3 else None

    # %% THE CONTROL -- run every time, no exceptions
    print(f"\n{'='*72}\n[TEST] {CASE}, {len(te)} held out, K={a.tiers}, HARD cuts")
    print(f"{'arm':44s}{'gap':>10s}{'discrete':>11s}{'n':>5s}")
    rows = {}
    for nm, fn in [
        ("baseline  flat V, no cut", flat),
        ("learned   V only (no cut)", lambda k, j: (VR[k], VI[k], None, THR[k] if THR is not None else None)),
        ("learned   V + LEARNED tier cuts", lambda k, j: (VR[k], VI[k], CNT[k], THR[k] if THR is not None else None)),
        ("learned   V + cut, threshold 0.5", lambda k, j: (VR[k], VI[k], CNT[k], None)),
        ("CONTROL   V + dummy band [0,0]",
         lambda k, j: (VR[k], VI[k], np.zeros(a.tiers), THR[k] if THR is not None else None)),
        ("CONTROL   V + random band",
         lambda k, j: (VR[k], VI[k], rng.uniform(0, [len(T) for T in TIERS]),
                       THR[k] if THR is not None else None)),
        ("CONTROL   V + mean band (no per-instance info)",
         lambda k, j: (VR[k], VI[k], CNT.mean(0), THR[k] if THR is not None else None)),
        ("CONTROL   V + cut, MEAN threshold",
         lambda k, j: (VR[k], VI[k], CNT[k], THR.mean(0) if THR is not None else None)),
    ]:
        gp, dc, n = evaluate(te, fn)
        rows[nm] = (gp, dc)
        print(f"  {nm:42s}{gp:>+9.3f}%{dc:>10.2f}%{n:>5d}", flush=True)
    L = rows["learned   V + LEARNED tier cuts"]
    worst_ctrl = min(v[0] for k, v in rows.items() if k.startswith("CONTROL"))
    # 0.02 is far too lax a margin on 48 instances. Require the learned band to
    # beat the best control by a margin that is not noise, AND require the band
    # to actually vary per instance -- a constant band is by definition matched
    # by the mean-band control.
    spread = float(CNT.std(0).mean())
    if THR is not None:
        print(f"\n  learned thresholds: mean {THR.mean():.3f}  "
              f"per-generator sd {THR.std(0).mean():.3f}  "
              f"per-instance sd {THR.std(0).mean():.3f}")
    print(f"\n  per-instance spread of tier counts: sd {CNT.std(0).round(3)} "
          f"(mean {spread:.3f})  -- a constant band has sd ~ 0")
    verdict = ("LEARNED CUT IS REAL" if (L[0] < worst_ctrl - 0.05 and spread > 0.15) else
               "*** DECORATIVE: a control matches the learned band ***")
    print(f"\n  learned {L[0]:+.3f}%  vs best control {worst_ctrl:+.3f}%  ->  {verdict}")
    print("="*72, flush=True)
    json.dump({k: list(v) for k, v in rows.items()},
              open(f"{HERE}/results/tier_{a.tag}.json", "w"), indent=1)
    torch.save(net.state_dict(), f"{HERE}/results/net_tier_{a.tag}.pt")
    pool.close()
