"""The method: predict the linearisation point AND a cardinality cut.

Self-supervised -- trained on deployment cost, no labels, no reference in the
loss. Two heads, and the measurements that justify each:

  head_v     Predicts the linearisation point. QCAC iterates to FIND that point,
             so predicting it removes the iteration. At the exact point one
             solve recovers the optimum (-0.002%, 100% agreement) with 40 of 54
             generators still fractional.
             v_scale = 0.5, NOT the 0.15 used previously: 4.0% of V* components
             need more than 0.15 (max 0.4038), so the old clamp made the target
             unreachable and capped every run near 1%.

  head_card  Predicts a band on the commitment CARDINALITY, imposed as
             n_lo <= sum_i u_i <= n_hi. This is the cut, and it is what makes
             ONE-SHOT deployment robust to the V error that will always remain:
             at ||dV||=0.76 it takes the gap 5.78% -> 1.90% and agreement
             68.2% -> 87.0%. It survives being wrong by +-2 units, and beats a
             cut that hands over 8 correct generators.

Why THIS cut family when free-form cuts failed. The old head emitted 8 x 108 =
864 numbers per instance and asked a zeroth-order estimator to learn them; the
variance alone was fatal. This head emits TWO. Nothing here is a valid
inequality by construction either, but a cardinality band cannot cut off a
commitment it does not bound, and restoration repairs upward past a low bound.

The band is anchored to the relaxation's OWN fractional sum, so it starts near
active and has gradient from the first step. It is biased LOW on purpose:
under-estimating beats over-estimating (n*-1 -> 1.980% vs n*+1 -> 2.714%)
because restoration only ever turns generators ON.
"""
# %% imports
import os, sys, time, json, argparse, re
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))
from pipeline import (CASE, RESERVE, grid, sample, set_cuts, set_down,
                      _init, _dep)
from socp import Socp


# %% network
class VCardNet(torch.nn.Module):
    def __init__(self, n_bus, n_gen, hidden=256, v_scale=0.5,
                 init_lo=0.02, init_hi=0.10):
        super().__init__()
        self.n_bus, self.n_gen, self.v_scale = n_bus, n_gen, v_scale
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(2*n_bus, hidden), torch.nn.SiLU(),
            torch.nn.Linear(hidden, hidden), torch.nn.SiLU())
        self.head_v = torch.nn.Linear(hidden, 2*n_bus)
        self.head_card = torch.nn.Linear(hidden, 2)
        torch.nn.init.zeros_(self.head_v.weight)      # start at flat V
        torch.nn.init.zeros_(self.head_v.bias)
        torch.nn.init.normal_(self.head_card.weight, std=1e-3)
        # The band is anchored to n_min, the fewest generators that can meet
        # the reserve (largest-capacity first). n_min needs no labels and is a
        # VALID lower bound -- you cannot serve the reserve with fewer.
        #
        # The offsets are FRACTIONS of the feasible span (n_gen - n_min), not
        # absolute units, so the head adapts across cases. A fixed max_off=15
        # works on case118 (n* - n_min ~ +6.9) but cannot reach case300's
        # +28.4, which would have made the target unrepresentable -- the same
        # unreachable-target bug as v_scale=0.15.
        #
        # init 0.02/0.10 of the span binds on BOTH cases: the offset a case
        # needs is a very different fraction of its span (case118 ~0.15,
        # case300 ~0.62), so only a low init is guaranteed to start active
        # everywhere. Initialised LOW and therefore BINDING. A band that starts too high
        # never binds, and an inactive cut has zero derivative with respect to
        # its own coefficients, so it never trains. Starting too low is
        # recoverable: restoration repairs upward.
        inv = lambda t: float(np.log(t/(1-t)))
        with torch.no_grad():
            self.head_card.bias.copy_(torch.tensor([inv(init_lo), inv(init_hi)]))

    def forward(self, x, s_rlx):
        h = self.trunk(x)
        dv = self.v_scale*torch.tanh(self.head_v(h))
        Vr = 1.0 + dv[:, :self.n_bus]
        Vi = 0.0 + dv[:, self.n_bus:]
        frac = torch.sigmoid(self.head_card(h))          # both in (0, 1)
        span = (self.n_gen - s_rlx).clamp(min=1.0)        # feasible room above n_min
        off = span[:, None]*frac
        n_lo = s_rlx + torch.minimum(off[:, 0], off[:, 1])
        n_hi = s_rlx + torch.maximum(off[:, 0], off[:, 1])
        return Vr, Vi, n_lo, n_hi

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def card_rows(n_lo, n_hi, n_gen, K=8):
    """The cut, as A @ [pg; u] <= b:  sum u >= n_lo  and  sum u <= n_hi."""
    A, b = np.zeros((K, 2*n_gen)), np.ones(K)
    A[0, n_gen:] = -1.0; b[0] = -float(n_lo)
    A[1, n_gen:] = 1.0;  b[1] = float(n_hi)
    return A, b


# %% main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["v", "vcard"], default="vcard")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--sigma", type=float, default=0.03)
    # The band is measured in GENERATORS, O(10), while the V outputs are
    # O(0.01-0.4). Smoothing both at sigma=0.03 perturbs the cardinality by
    # 0.03 of a generator, which can never change a rounded commitment: those
    # two dimensions got exactly zero gradient for 50 epochs and the band never
    # moved off its initial value. Per-group sigma is not a tuning knob here,
    # it is the difference between training and not training.
    ap.add_argument("--sigma-card", type=float, default=1.0)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="vcard")
    ap.add_argument("--down", type=int, default=6)   # downward repair trials
    ap.add_argument("--n-test", type=int, default=24)
    a = ap.parse_args()
    import multiprocessing as mp
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    set_cuts(8); set_down(a.down)
    g, nl = grid(); G = g.n_gen

    # %% pooled labelled instances (references used ONLY for evaluation)
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
    nte = a.n_test
    te = [int(i) for i in perm[:nte]]; tr = [int(i) for i in perm[nte:]]
    print(f"[setup] {len(POOL)} instances -> {len(tr)} train / {len(te)} test | "
          f"ARM={a.arm}", flush=True)

    # %% anchor: n_min, the fewest generators that can meet the reserve.
    # Label-free, a VALID lower bound, and correlated 0.926 with n*. No solve
    # needed -- it is a sort and a cumulative sum.
    order = np.argsort(-g.pmax); cum = np.cumsum(g.pmax[order])
    SR = np.array([float(np.searchsorted(cum, (1.0+RESERVE)*float(p.sum())) + 1)
                   for p, _, _, _ in POOL])
    print(f"[anchor] n_min {SR.min():.0f}-{SR.max():.0f} (mean {SR.mean():.1f})",
          flush=True)
    base = float(g.pd.sum())
    X = torch.tensor(np.stack([np.r_[p, q]/base for p, q, _, _ in POOL]),
                     dtype=torch.float32)
    mu, sg = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu)/sg
    SRt = torch.tensor(SR, dtype=torch.float32)

    pool = mp.get_context("fork").Pool(a.procs, initializer=_init)

    def jobs_for(idx, Vr, Vi, n_lo, n_hi):
        J = []
        for k, j in enumerate(idx):
            A, b = (card_rows(n_lo[k], n_hi[k], G) if a.arm == "vcard"
                    else (None, None))
            J.append((j, POOL[j][0], POOL[j][1], Vr[k], Vi[k], A, b))
        return J

    def evaluate(idx, net):
        if net is None:
            J = [(j, POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus),
                  None, None) for j in idx]
        else:
            with torch.no_grad():
                vr, vi, lo, hi = net(Xn[idx], SRt[idx])
            J = jobs_for(idx, vr.numpy().astype(float), vi.numpy().astype(float),
                         lo.numpy().astype(float), hi.numpy().astype(float))
        out = {i: (c, u, s) for i, c, u, s in pool.imap_unordered(_dep, J)}
        gaps, agrs, secs, nf = [], [], [], 0
        for j in idx:
            c, u, s = out[j]; secs.append(s)
            if c >= 1e8 or u is None:
                nf += 1; continue
            gaps.append(100*(c-POOL[j][3])/POOL[j][3])
            agrs.append(100*float((u == POOL[j][2]).mean()))
        return (float(np.mean(gaps)) if gaps else np.nan,
                float(np.mean(agrs)) if agrs else np.nan,
                float(np.mean(secs)), nf)

    # Per-instance scale for the gradient. Previously this was the REFERENCE
    # cost -- a label, in a method that claims to use none. The flat-V
    # deployment cost is the same order of magnitude, needs no reference, and
    # only sets the gradient's per-instance magnitude.
    J0 = [(j, POOL[j][0], POOL[j][1], np.ones(g.n_bus), np.zeros(g.n_bus),
           None, None) for j in range(len(POOL))]
    SCALE = np.zeros(len(POOL))
    for i, c, u, s_ in pool.imap_unordered(_dep, J0):
        SCALE[i] = c if c < 1e8 else 1e5
    print(f"[scale] label-free flat-V cost: mean {SCALE.mean():,.0f}", flush=True)

    bg, ba, bs, bf = evaluate(te, None)
    print(f"[BASE] flat V, no cut: gap {bg:+.3f}%  agree {ba:.1f}%  "
          f"{bs:.2f}s/inst  fail {bf}/{len(te)}\n", flush=True)

    # %% train -- smoothed gradient on the deployment cost. No labels.
    net = VCardNet(g.n_bus, G)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    print(f"[net ] {net.n_params():,} params | outputs "
          f"{2*g.n_bus + (2 if a.arm=='vcard' else 0)} "
          f"(the old free-form cut head emitted 864)", flush=True)
    best = None
    for ep in range(a.epochs):
        t0 = time.time()
        bidx = [int(i) for i in rng.permutation(tr)]
        vr, vi, lo, hi = net(Xn[bidx], SRt[bidx])
        Y = (torch.cat([vr, vi, lo[:, None], hi[:, None]], 1) if a.arm == "vcard"
             else torch.cat([vr, vi], 1))
        Yd = Y.detach().numpy(); nb, dY = Y.shape
        E = rng.standard_normal((a.pairs, nb, dY))
        nbus = g.n_bus
        sig = np.full(dY, a.sigma)
        if a.arm == "vcard":
            sig[-2:] = a.sigma_card

        def split(yv):
            if a.arm == "vcard":
                return yv[:nbus], yv[nbus:2*nbus], yv[2*nbus], yv[2*nbus+1]
            return yv[:nbus], yv[nbus:2*nbus], None, None

        J, key = [], []
        for k, j in enumerate(bidx):
            for tag, yv in ([("0", Yd[k])] +
                            [(s, Yd[k] + sgn*sig*E[p, k])
                             for p in range(a.pairs)
                             for s, sgn in ((f"+{p}", 1), (f"-{p}", -1))]):
                Vr_, Vi_, l_, h_ = split(yv)
                A, b = (card_rows(l_, h_, G) if a.arm == "vcard" else (None, None))
                J.append((k, POOL[j][0], POOL[j][1], Vr_, Vi_, A, b))
                key.append((tag, k))
        res = list(pool.imap(_dep, J))
        C = np.array([r[1] for r in res], float)
        scale = SCALE[bidx]
        Gest = np.zeros((nb, dY))
        base_c = np.full(nb, np.nan)
        for t, (tag, k) in enumerate(key):
            if tag == "0":
                base_c[k] = C[t]
            elif tag.startswith("+"):
                p = int(tag[1:]); cp_, cm_ = C[t], C[t+1]
                if cp_ < 1e8 and cm_ < 1e8:
                    # per-dimension sigma: grad ~ (f(y+Se)-f(y-Se))/2 * S^-1 e
                    Gest[k] += (cp_-cm_)/(2*scale[k])*(E[p, k]/sig)/a.pairs
        loss = (Y*torch.tensor(Gest, dtype=torch.float32)).sum()/nb
        opt.zero_grad(); loss.backward(); opt.step()
        ok = base_c < 1e8
        # Selection uses the RAW mean deployment cost -- lower is better, no
        # reference needed. Selecting on the gap against the reference, as
        # before, is model selection on labels and quietly breaks the
        # self-supervised claim.
        sel = float(np.mean(base_c[ok]/SCALE[bidx][ok])) if ok.any() else np.inf
        gp_ = 100*float(np.mean((base_c[ok]-np.array([POOL[j][3] for j in bidx])[ok])
                                / np.array([POOL[j][3] for j in bidx])[ok])) \
            if ok.any() else np.nan     # reported for MONITORING only
        if best is None or sel < best[0]:
            best = (sel, {k: v.clone() for k, v in net.state_dict().items()})
        if ep % 5 == 0 or ep == a.epochs-1:
            msg = (f"[ep {ep:3d}] train gap {gp_:+7.3f}%  fail {int((~ok).sum())}/{nb}"
                   f"  {time.time()-t0:5.1f}s")
            if a.arm == "vcard":
                msg += f"  band [{lo.mean():.1f}, {hi.mean():.1f}]"
            print(msg, flush=True)

    net.load_state_dict(best[1])
    tg, ta, ts, tf = evaluate(te, net)
    print(f"\n{'='*70}\n[TEST] reference = QCAC iterative (== global MINLP)")
    print(f"  {'baseline  flat V, no cut':34s} gap {bg:+7.3f}%  agree {ba:5.1f}%")
    print(f"  {'learned   arm='+a.arm:34s} gap {tg:+7.3f}%  agree {ta:5.1f}%  "
          f"fail {tf}/{len(te)}")
    print(f"  {ts:.2f}s/instance   selection score {best[0]:.4f} "
          f"(label-free)\n{'='*70}", flush=True)
    json.dump(dict(arm=a.arm, base=[bg, ba], learned=[tg, ta, ts, tf],
                   args=vars(a)), open(f"{HERE}/results/vcard_{a.tag}.json", "w"),
              indent=1)
    torch.save(net.state_dict(), f"{HERE}/results/net_{a.tag}.pt")
    pool.close()
