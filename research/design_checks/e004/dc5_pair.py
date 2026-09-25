"""DC5: hard-pair feasibility under the Adam-like flow (design check only).
Fixed benign-amplification structure B (attempt credit). Search a coupled AND2 exploit Y whose
(J_G, FPR, J_V) trajectory over the early window [0, 0.75] (5% of T = 15) matches B's. Then compare
geometry (alpha, C, C_out) and the outcome at T = 15 relative to each structure's clean run."""
import numpy as np
from scipy.optimize import minimize
from utoy import Spec
from dc4_adam import run_adam

K, m = 2, 3
w = np.array([0.5, 0.5])
T, H = 15.0, 0.75
grid_n = 7  # window points 0..H

B = Spec(w, np.array([0.7, 0.55]), m, beta=np.array([0.8, 0.8]))
thB = np.array([-1.2, -1.6, -2.0, -3.0, -3.0, 0.0])
rb = run_adam(B, thB, H, n=grid_n)
target = np.array([[x["jg"], x["fpr"], x["jv"]] for x in rb])

def y_spec(lam):
    return Spec(w, np.array([1 - 1e-9] * K), m, event="and2", trig=np.ones(K), lam=np.array([lam, lam, 0.0]))

def y_theta(par):
    u1, u2, h, phi, lam = par
    return np.array([u1, u2, h, phi, phi, 0.0]), lam

def loss(par):
    th, lam = y_theta(par)
    try:
        r = run_adam(y_spec(lam), th, H, n=grid_n)
    except Exception:
        return 1e3
    got = np.array([[x["jg"], x["fpr"], x["jv"]] for x in r])
    return float(np.sum(((got - target) / np.array([0.01, 0.01, 0.01])) ** 2))

best = None
for start in ([-2.0, -2.2, -2.0, -1.5, 1.0], [-1.8, -2.0, -1.5, -1.0, 2.0], [-2.2, -1.9, -2.5, -0.5, 0.5]):
    res = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 400, "xatol": 1e-3, "fatol": 1e-3})
    if best is None or res.fun < best.fun:
        best = res
print(f"best normalized window mismatch (sum of squares in units of 0.01): {best.fun:.3f}; params {np.round(best.x, 3)}")
thY, lam = y_theta(best.x)
ry = run_adam(y_spec(lam), thY, H, n=grid_n)
print("window  t    | J_G  B / Y      | FPR  B / Y      | alpha B / Y     | C_out B / Y")
for a, b in zip(rb, ry):
    print(f"      {a['t']:.3f} | {a['jg']:.4f} / {b['jg']:.4f} | {a['fpr']:.4f} / {b['fpr']:.4f} | {a['alpha']:+.3f} / {b['alpha']:+.3f} | {a['C_out']:.3f} / {b['C_out']:.3f}")
# outcomes at T, relative to each structure's own clean run (V = G)
for name, spec, th, clean in (("B", B, thB, Spec(w, B.p, m)), ("Y", y_spec(lam), thY, Spec(w, np.array([1 - 1e-9] * K), m, lam=np.array([lam, lam, 0.0])))):
    r = run_adam(spec, th, T, n=31); c = run_adam(clean, th, T, n=31)
    j0, jT, cT = r[0]["jg"], r[-1]["jg"], c[-1]["jg"]
    print(f"{name}: J_G(0) {j0:.3f}  J_G(T) {jT:.3f}  clean J_G(T) {cT:.3f}  normalized shortfall {(cT - jT) / (cT - j0):+.3f}  FPR(T) {r[-1]['fpr']:.3f}")
