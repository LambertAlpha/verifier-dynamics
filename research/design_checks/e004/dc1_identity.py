"""DC1: under natural gradient, is the geometry trajectory a function of (J_G, FPR) and their
time derivatives?  Prediction (derived): yes for 'C-type' verifiers (feature-triggered FP only,
FNR = 0, trigger active in every context, no capability coupling):
    alpha = -FPR,  A^2 = J_G' / (1 - FPR),  C^2 = (1 - J_G) FPR'.
It should break for coupling (lam != 0), attempt credit (B), deletion (X), hack acceptance (D),
false negatives (R-type FN coin) and context-specific triggers."""
import numpy as np
from utoy import Spec, UToy, geometry, ng_velocity

rng = np.random.default_rng(0)
K, m = 4, 3

def rand_theta():
    return np.concatenate([rng.normal(-1.5, 1.0, K), [rng.normal(-2, 1)], rng.normal(-1, 1, m)])

def check(spec, label, n=6):
    errs = []
    for _ in range(n):
        toy = UToy(spec)
        th = rand_theta()
        vel, aux = ng_velocity(toy, th)
        jg, jv, fpr, fnr = aux["js"]
        geo = geometry(aux["gG"], aux["gV"], aux["Fi"])
        jg_dot = aux["gG"] @ vel
        fpr_dot = aux["gF"] @ vel
        pred = {"alpha": -fpr, "A2": jg_dot / (1 - fpr), "C2": (1 - jg) * fpr_dot}
        got = {"alpha": geo["alpha"], "A2": geo["A"] ** 2, "C2": geo["C"] ** 2}
        errs.append(max(abs(pred[k] - got[k]) / (abs(got[k]) + 1e-12) for k in pred))
    print(f"{label:48s} max relative deviation from the L2 identity: {max(errs):.2e}")

w = np.full(K, 1 / K); p = np.full(K, 0.999999)
ones = np.ones(K)
check(Spec(w, p, m, event="and2", trig=ones), "C-type AND2 (identity expected)")
check(Spec(w, p, m, event="or2", trig=ones), "C-type OR2 (identity expected)")
check(Spec(w, np.array([0.9, 0.6, 0.8, 0.95]), m, event="single", trig=ones), "C-type single, p<1 (identity expected)")
check(Spec(w, p, m, event="and2", trig=ones, lam=np.array([0.8, 0.8, 0.0])), "coupled exploit lam != 0")
check(Spec(w, np.array([0.6, 0.5, 0.7, 0.4]), m, beta=np.full(K, 0.7)), "B attempt credit")
check(Spec(w, p, m, deleted=np.array([1, 0, 0, 0], bool), v0=0.5), "X deletion (v0 = 0.5)")
check(Spec(w, p, m, rho=np.full(K, 0.8)), "D hack acceptance")
check(Spec(w, p, m, event="single", trig=ones, fn=np.full(K, 0.1)), "C-type + FN coin")
check(Spec(w, p, m, event="single", trig=np.array([1, 1, 0, 0.0])), "context-specific trigger")
check(Spec(w, p, m, fp=np.full(K, 0.2), fn=np.full(K, 0.2)), "R symmetric coins")
