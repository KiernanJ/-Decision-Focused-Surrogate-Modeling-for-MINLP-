# F2: PGLib api 118 -- Grid (via pandapower from_ppc round trip) vs the raw MATPOWER .m data
import sys, os, numpy as np, warnings; warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src"))
from pipeline import grid
from acopf_data import read_m
case = "pglib_opf_case118_ieee__api"
g, nl = grid(case); m = read_m(case); base = m["baseMVA"]
bus, br, gen, gc = m["bus"], m["branch"], m["gen"], m["gencost"]
ids = bus[:, 0].astype(int)            # external numbers in .m order; Grid bus order: check pd matches
print("buses:", g.n_bus, len(bus), "| pd match (sorted order):", np.allclose(np.sort(g.pd), np.sort(bus[:, 2]/base)), "| pd match in order:", np.allclose(g.pd, bus[:, 2]/base))
print("gs/bs in order:", np.allclose(g.gs, bus[:, 4]/base), np.allclose(g.bs, bus[:, 5]/base), "| vmax/vmin:", np.allclose(g.vmax, bus[:, 11]), np.allclose(g.vmin, bus[:, 12]))
# branches: compare as multiset keyed by (f, t)
pos = {b: k for k, b in enumerate(ids)}
mf, mt = np.array([pos[int(x)] for x in br[:, 0]]), np.array([pos[int(x)] for x in br[:, 1]])
print("branch count", g.n_branch, len(br), "| status all 1:", bool((br[:, 10] == 1).all()))
ok_ft = np.array_equal(g.f_bus, mf) and np.array_equal(g.t_bus, mt); print("branch f/t same order:", ok_ft)
z = 1/(g.g + 1j*g.b)
print("r, x, b, tap, rate match:", np.allclose(z.real, br[:, 2], atol=1e-9), np.allclose(z.imag, br[:, 3], atol=1e-9), np.allclose(2*g.b_fr, br[:, 4], atol=1e-9),
      np.allclose(g.tap, np.where(br[:, 8] == 0, 1, br[:, 8])), np.allclose(g.rate, br[:, 5]/base))
if not np.allclose(g.rate, br[:, 5]/base):
    d = np.abs(g.rate - br[:, 5]/base); k = np.argsort(-d)[:5]; print("   worst rate diffs", np.c_[g.rate[k], br[k, 5]/base])
# gens keyed by bus
gb = np.array([pos[int(x)] for x in gen[:, 0]])
print("gen count", g.n_gen, len(gen), "| order equal:", np.array_equal(np.sort(g.gen_bus), np.sort(gb)))
for k in range(g.n_gen):
    j = np.where(gb == g.gen_bus[k])[0]
    if len(j) != 1: print("  multiple gens at bus", g.gen_bus[k]); continue
    j = j[0]
    ref = np.array([gen[j, 9], gen[j, 8], gen[j, 4], gen[j, 3]])/base; mine = np.array([g.pmin[k], g.pmax[k], g.qmin[k], g.qmax[k]])
    c = gc[j]; c2, c1 = (c[4], c[5]) if c[3] == 3 else (0.0, c[4])
    if not (np.allclose(ref, mine) and np.isclose(g.c2[k], c2*base**2) and np.isclose(g.c1[k], c1*base)):
        print("  MISMATCH gen", k, "bus", g.gen_bus[k], mine, ref, g.c2[k], c2*base**2, g.c1[k], c1*base)
print("gen check done; ref bus", g.ref, "(.m ref row", int(np.where(bus[:, 1] == 3)[0][0]), ")")
# keyed comparison: for each .m branch find the Grid branch with the same (f, t) (or reversed) and closest x
used = set(); worst = dict(r=0, x=0, b=0, tap=0, rate=0); rev = 0
for k in range(len(br)):
    cand = [j for j in range(g.n_branch) if j not in used and ((g.f_bus[j] == mf[k] and g.t_bus[j] == mt[k]) or (g.f_bus[j] == mt[k] and g.t_bus[j] == mf[k]))]
    j = min(cand, key=lambda j: abs(z.imag[j] - br[k, 3])); used.add(j)
    if g.f_bus[j] != mf[k]: rev += 1
    worst["r"] = max(worst["r"], abs(z.real[j] - br[k, 2])); worst["x"] = max(worst["x"], abs(z.imag[j] - br[k, 3]))
    worst["b"] = max(worst["b"], abs(2*g.b_fr[j] - br[k, 4])); worst["tap"] = max(worst["tap"], abs(g.tap[j] - (br[k, 8] if br[k, 8] else 1.0)))
    worst["rate"] = max(worst["rate"], abs(g.rate[j] - br[k, 5]/base))
print("keyed branch comparison: max abs diffs", {k: f"{v:.2e}" for k, v in worst.items()}, "| reversed orientation:", rev, "| matched", len(used))
print("ratings: min/median/max pu", g.rate.min(), np.median(g.rate), g.rate.max(), "| .m RATE_A==0 count", int((br[:, 5] == 0).sum()))
