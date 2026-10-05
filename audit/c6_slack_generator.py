"""Critical 6: how the loader treats the slack-bus (ext_grid) generator.

Compares the limits src/acopf_data.py produces against pandapower's own
net.ext_grid limits, and counts how often the reference commits the unit.
Runtime: seconds.
"""
import _common
from _common import pool_raw
import numpy as np
import pandapower.networks as pn
from pipeline import grid

for case in ("case118", "case300"):
    g, nl = grid(case)
    k = int(np.flatnonzero(g.always_on)[0]) if getattr(g, "always_on", None) is not None \
        else int(np.argmax(g.pmax))
    eg = getattr(pn, case)().ext_grid
    m = pool_raw(case)
    on = sum(int(np.asarray(r["u"])[k]) for _, r in m.values())
    print(f"{case}: unit {k} (always_on flag set: {bool(g.always_on[k])})")
    print(f"   loader:     pmax {g.pmax[k]:.2f} pu, qmin/qmax {g.qmin[k]:.2f}/{g.qmax[k]:.2f} pu, "
          f"no-load cost {nl[k]:.0f} (median over units {np.median(nl):.0f})")
    print(f"   pandapower: max_p_mw {eg['max_p_mw'].values}, min/max_q_mvar "
          f"{eg['min_q_mvar'].values}/{eg['max_q_mvar'].values}")
    print(f"   committed in {on}/{len(m)} references")
print("\nalways_on is not enforced by acuc.build or socp.Socp (grep shows it is only read by qcac.solve_exact).")
