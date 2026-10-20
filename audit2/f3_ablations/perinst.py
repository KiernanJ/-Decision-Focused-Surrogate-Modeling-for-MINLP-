import pickle, sys, numpy as np
CASE, SET = sys.argv[1], sys.argv[2]; arms = sys.argv[3].split(",")
st = pickle.load(open(f"res_{CASE}_{SET}.pkl","rb"))
SR = st["meta"]["SR"]; RL = st["meta"]["RLflat"]; ids = st["meta"]["ids"] if SET == "val" else st["meta"]["scored"]
print("inst sc " + " ".join(f"{a:>8s}" for a in arms))
for i in ids:
    row = []
    for a in arms:
        r = st["arms"][a].get(i)
        if r is None: row.append("    FAIL"); continue
        if i in SR:
            J = SR[i][0]; row.append(f"{100*(r['cost']-J)/J:8.3f}")
        else:
            row.append(f"{r['cost']/RL[i]:7.4f}*")
    print(f"{i:4d} {'y' if i in SR else 'n'}  " + " ".join(row))
print("(* = no converged ref: LF ratio shown; else signed gap %)")
