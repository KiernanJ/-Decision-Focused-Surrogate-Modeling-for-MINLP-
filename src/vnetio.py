"""Load a saved V-net (acuc/40_vnet_layer.py output) and evaluate it (2026-10-12, clean label-free redo).
A V-net file whose label-free selection picked flat V stores state=None: then V = flat (Vr = 1, Vi = 0) for every instance,
instead of crashing in load_state_dict."""
import numpy as np
import torch
import torch.nn as nn


def vnet_out(path, Xn_rows, n):
    """-> (OUT (rows, 2n) with V = (1 + OUT[:, :n], OUT[:, n:]), rho, selection label)."""
    VS = torch.load(path, weights_only=False)
    import os
    rho = float(os.environ["DEPLOY_RHO"]) if os.environ.get("DEPLOY_RHO") else VS["rho"]   # (2026-10-13) deployment penalty, selected by the label-free score of the FULL deployed method; V-net TRAINING (40) is unaffected
    if VS["state"] is None:
        return np.zeros((Xn_rows.shape[0], 2*n)), rho, "flat"
    vm = nn.Sequential(nn.Linear(Xn_rows.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 2*n))
    vm.load_state_dict(VS["state"]); vm.eval()
    with torch.no_grad():
        return vm(Xn_rows).numpy().astype(float), rho, str(VS.get("sel")) + (f" [deployment rho overridden: {rho:.0e}]" if os.environ.get("DEPLOY_RHO") else "")
