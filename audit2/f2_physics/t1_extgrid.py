# F2 task 1: slack-bus (ext_grid) unit after the audit fix, vs source data
import sys, os, numpy as np, warnings
warnings.filterwarnings("ignore")
R = sys.argv[1]; sys.path.insert(0, os.path.join(R, "src"))
import pandapower as pp, pandapower.networks as pn
from pipeline import grid
from acopf_data import read_m
print("pandapower", pp.__version__)
for case in ("case118", "case300", "pglib_opf_case118_ieee__api"):
    g, nl = grid(case)
    if case.startswith("pglib"):
        from pandapower.converter.pypower import from_ppc
        net = from_ppc(read_m(case), f_hz=60, validate_conversion=False)
    else:
        net = getattr(pn, case)()
    try: pp.runpp(net, calculate_voltage_angles=True, init="flat")
    except Exception: pp.rundcpp(net)
    ppc = net._ppc; gen = ppc["gen"]; base = ppc["baseMVA"]
    ne, ng = len(net.ext_grid), len(net.gen); nsg = len(net.sgen)
    print(f"\n=== {case}: G={g.n_gen} | ext_grid {ne}, gen {ng}, sgen {nsg} (controllable sgen {int(net.sgen.get('controllable', 0).sum()) if nsg else 0}), ppc gen rows {len(gen)}")
    # ppc gen order check: bus of each ppc gen row vs ext_grid/gen element bus
    lk = net._pd2ppc_lookups["bus"]
    eg_b = [int(lk[b]) for b in net.ext_grid.bus]; gn_b = [int(lk[b]) for b in net.gen.bus]
    ppc_b = [int(r[0]) for r in gen]
    print("  ppc gen rows 0..ne-1 at ext_grid buses:", ppc_b[:ne] == eg_b, "| rows ne.. at gen buses (same order):", ppc_b[ne:ne+ng] == gn_b)
    k = 0
    print(f"  ext_grid row in Grid: bus {g.gen_bus[k]} pmin {g.pmin[k]:.4f} pmax {g.pmax[k]:.4f} qmin {g.qmin[k]:.4f} qmax {g.qmax[k]:.4f} c2 {g.c2[k]:.4f} c1 {g.c1[k]:.4f} c0 {g.c0[k]:.2f} nl {nl[k]:.2f}")
    print("  net.ext_grid limits (MW/Mvar):", {c: float(net.ext_grid[c].iloc[0]) if c in net.ext_grid else None for c in ("max_p_mw","min_p_mw","max_q_mvar","min_q_mvar")})
    pc = net.poly_cost; r = pc[(pc.et=="ext_grid") & (pc.element==net.ext_grid.index[0])]
    print("  poly_cost ext_grid:", r[["cp2_eur_per_mw2","cp1_eur_per_mw","cp0_eur"]].values.tolist())
    print(f"  ranks: pmax rank {int((g.pmax > g.pmax[k]).sum())+1}/{g.n_gen}; c1 rank {int((g.c1 < g.c1[k]).sum())+1}; median nl {np.median(nl):.2f}")
    # compare every unit's limits vs net tables
    lim = np.r_[net.ext_grid[["min_p_mw","max_p_mw","min_q_mvar","max_q_mvar"]].values, net.gen[["min_p_mw","max_p_mw","min_q_mvar","max_q_mvar"]].values]/base
    mine = np.c_[g.pmin, g.pmax, g.qmin, g.qmax]
    bad = np.where(np.abs(np.nan_to_num(lim, nan=-99) - mine).max(1) > 1e-9)[0]
    print(f"  units whose Grid limits differ from net tables: {bad.tolist()[:20]}", (np.c_[mine[bad], lim[bad]] if len(bad) else ""))
    # negative pmin clipped?  units with pmin<0 in source
    print("  source pmin<0 units:", np.where(lim[:,0] < 0)[0].tolist(), "| qmin>0 units:", np.where(g.qmin > 0)[0].tolist(), "| qmax<0 units:", np.where(g.qmax < 0)[0].tolist())
    # MATPOWER source for pglib: compare with .m gen table directly
    if case.startswith("pglib"):
        m = read_m(case); mg = m["gen"]; mb = m["bus"]
        refbus = int(mb[mb[:,1]==3][0,0]); print("  .m ref bus", refbus, "gens at ref bus rows:", np.where(mg[:,0]==refbus)[0].tolist())
        print("  .m gen rows at ref bus (Pmax Pmin Qmax Qmin):", mg[mg[:,0]==refbus][:, [8,9,3,4]].tolist())
        print("  .m gencost for those rows:", m["gencost"][mg[:,0]==refbus].tolist())
        # match all gens: our sorted (bus,pmax) vs .m
        ours = sorted(zip([int(net.bus.name.iloc[b]) if 'name' in net.bus else b for b in []], []))
    # phase shifters / taps
    sh = ppc["branch"][:, 9]; print(f"  phase-shift branches: {int((np.abs(sh)>1e-9).sum())}; finite ratings {int(np.isfinite(g.rate).sum())}/{g.n_branch}, min rate {np.min(g.rate):.3f} pu, median {np.median(g.rate[np.isfinite(g.rate)]) if np.isfinite(g.rate).any() else np.nan:.3f}")
    print(f"  load {g.pd.sum():.3f} pu, cap {g.pmax.sum():.3f} pu, need at s=1.30: {1.1*1.3*g.pd.sum():.2f}, at 1.20: {1.1*1.2*g.pd.sum():.2f}, at 1.05 {1.1*1.05*g.pd.sum():.2f}")
    print(f"  vmin/vmax used: {g.vmin.min():.3f}-{g.vmin.max():.3f} / {g.vmax.min():.3f}-{g.vmax.max():.3f}")
