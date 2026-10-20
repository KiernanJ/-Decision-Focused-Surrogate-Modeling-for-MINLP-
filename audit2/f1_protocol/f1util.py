"""worker initializers for the solver-seed noise check (importable module for spawn)."""
def init_seed(case, seed):
    import combopt_ac
    combopt_ac.init(case, True)
    combopt_ac._W["env"].setParam("Seed", int(seed))
