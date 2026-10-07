"""Hybrid-vehicle MILP of Dixit, Gupta & Zhang (TMLR 2025), reproduced from their code
(ddolab/DFSM-for-MILPs, MPC_data_generation.jl): T = 30, S = 3 engine levels, tau = 5,
E0 = 95, E_max = 100, P_max = 1, lossless battery, linear fuel cost; demand profile
D_t = (sin(sqrt(u) t) + 1)/5 with u ~ U[0.1, 2.0]. One "vehicle" = one cost draw."""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

T, SW, TAU, E0, EMAX, PMAX = 30, 3, 5.0, 95.0, 100.0, 1.0
NY = 3*T + 1                                   # y = [E (T+1) | P_eng (T) | z (T)]


class Vehicle:
    def __init__(self, vid, env=None):
        rng = np.random.default_rng(1000 + vid)
        self.cE = -rng.integers(10, 31, T)/1000; self.cT = -rng.integers(25, 51)/10
        self.cP = 11 + rng.integers(-50, 51, T)/500; self.cZ = 1.5 + rng.integers(-50, 51, T)/5000
        if env is None:
            env = gp.Env(empty=True); env.setParam("OutputFlag", 0); env.setParam("Threads", 1); env.start()
        self.env = env

    @staticmethod
    def demand(u):
        return (np.sin(np.sqrt(u)*np.arange(1, T+1)) + 1)/5

    def solve(self, D, integer=True, zfix=None, target=None, A=None, b=None, tl=120.):
        """integer MILP / LP relaxation / fixed-z re-solve / L1 projection (+ cut rows)."""
        m = gp.Model(env=self.env); m.Params.TimeLimit = tl; m.Params.MIPGap = 1e-4
        Pb = m.addVars(T, lb=-GRB.INFINITY); Pe = m.addVars(T, lb=0, ub=PMAX); E = m.addVars(T+1, lb=0, ub=EMAX)
        z = m.addVars(T, lb=0, ub=SW, vtype=GRB.INTEGER if (integer and zfix is None) else GRB.CONTINUOUS)
        m.addConstr(E[0] == E0)
        for i in range(T):
            m.addConstr(E[i+1] == E[i] - TAU*Pb[i]); m.addConstr(Pb[i] + Pe[i] >= D[i])
            m.addConstr(Pe[i] <= z[i]*PMAX/SW)
            if zfix is not None: m.addConstr(z[i] == float(zfix[i]), name=f"fix{i}")
        y = [E[i] for i in range(T+1)] + [Pe[i] for i in range(T)] + [z[i] for i in range(T)]
        cost = (-self.cT*(EMAX - E[T]) + gp.quicksum(float(self.cE[i])*E[i] for i in range(T))
                + gp.quicksum(float(self.cP[i])*Pe[i] + float(self.cZ[i])*z[i] for i in range(T)))
        obj = cost
        if target is not None:
            w = m.addVars(T, lb=0)
            for i in range(T):
                m.addConstr(w[i] >= z[i]-float(target[i])); m.addConstr(w[i] >= float(target[i])-z[i])
            obj = gp.quicksum(w[i] for i in range(T))
        if A is not None:
            s = m.addVars(len(b), lb=0)
            for r in range(len(b)):
                m.addConstr(gp.quicksum(float(A[r, j])*y[j] for j in range(NY) if A[r, j] != 0) <= float(b[r]) + s[r])
            obj = obj + 1e3*gp.quicksum(s[r] for r in range(len(b)))
        m.setObjective(obj, GRB.MINIMIZE); m.optimize()
        if m.SolCount == 0:
            return None
        zv = np.array([z[i].X for i in range(T)])
        dz = (np.array([m.getConstrByName(f"fix{i}").Pi for i in range(T)])
              if (zfix is not None and not m.IsMIP) else None)          # LP dual = d cost / d z
        return dict(cost=float(cost.getValue()), z=zv, y=np.array([v.X for v in y]),
                    Pe=np.array([Pe[i].X for i in range(T)]), secs=m.Runtime,
                    gap=(m.MIPGap if m.IsMIP else 0.0), dz=dz)

    def search(self, D, zq, k=5, rounds=30):
        """dual-guided local search over +-1 mode moves: rank by the LP dual of the fixed-z
        re-solve (exact sensitivity), price the top-k, accept the best improvement, repeat."""
        r = self.solve(D, zfix=zq); zq = zq.copy()
        for _ in range(rounds):
            pred = []
            for t in range(T):
                for s_ in (1, -1):
                    if 0 <= zq[t] + s_ <= SW: pred.append((r["dz"][t]*s_, t, s_))
            best = None
            for _, t, s_ in sorted(pred)[:k]:
                w = zq.copy(); w[t] += s_; rn = self.solve(D, zfix=w)
                if rn is not None and rn["cost"] < r["cost"] - 1e-9 and (best is None or rn["cost"] < best[1]["cost"]):
                    best = (w, rn)
            if best is None: break
            zq, r = best
        return zq, r
