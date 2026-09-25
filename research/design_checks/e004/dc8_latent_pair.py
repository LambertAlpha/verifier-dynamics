"""DC8: latent-decline hard pair under MF-Adam (B = 64). Fixed displacement structure D
(preference inversion on the hard context). Search a benign-amplification structure B whose
(J_G, FPR, J_V) over the early window [0, 0.75] (5% of T = 15) matches D's. Compare geometry in the
optimizer metric and outcomes at T."""
import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from utoy import Spec, UToy, geometry
from dc6_mfadam import mf_adam_velocity

K, m = 2, 3
w = np.array([0.5, 0.5]); T, H, BATCH = 15.0, 0.75, 64

def traj(spec, th0, t_end, n):
    toy = UToy(spec)
    sol = solve_ivp(lambda t, th: mf_adam_velocity(toy, th, BATCH)[0], (0, t_end), th0, rtol=1e-7,
                    atol=1e-10, dense_output=True, method="LSODA")
    rows = []
    for t in np.linspace(0, t_end, n):
        th = sol.sol(t)
        gG, gV, gF, (jg, jv, fpr, fnr) = toy.grads(th)
        _, M = mf_adam_velocity(toy, th, BATCH)
        geo = geometry(gG, gV, np.diag(M), toy.ctx_gold_grads(th))
        rows.append(dict(t=t, jg=jg, jv=jv, fpr=fpr, **geo))
    return rows

D = Spec(w, np.array([0.9, 0.3]), m, rho=np.full(K, 0.6))
thD = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])
rd = traj(D, thD, H, 7)
target = np.array([[x["jg"], x["fpr"], x["jv"]] for x in rd])

def b_spec(par):
    p1, p2, beta = par[3], par[4], par[5]
    return Spec(w, np.clip(np.array([p1, p2]), 0.05, 0.99), m, beta=np.full(K, np.clip(beta, 0.0, 1.0)))

def loss(par):
    th = np.array([par[0], par[1], par[2], 0.0, 0.0, 0.0])
    try:
        r = traj(b_spec(par), th, H, 7)
    except Exception:
        return 1e3
    got = np.array([[x["jg"], x["fpr"], x["jv"]] for x in r])
    return float(np.sum(((got - target) / 0.01) ** 2))

best = None
for start in ([0.0, 0.0, -2.0, 0.9, 0.3, 0.5], [0.5, -0.5, -1.0, 0.7, 0.5, 0.3], [-0.5, 0.5, -3.0, 0.6, 0.4, 0.8]):
    res = minimize(loss, start, method="Nelder-Mead", options={"maxiter": 500, "xatol": 1e-3, "fatol": 1e-3})
    if best is None or res.fun < best.fun:
        best = res
par = best.x
print(f"best window mismatch (sum of squares, units of 0.01): {best.fun:.3f}; B params u=({par[0]:.2f},{par[1]:.2f}) h={par[2]:.2f} p=({par[3]:.2f},{par[4]:.2f}) beta={par[5]:.2f}")
thB = np.array([par[0], par[1], par[2], 0.0, 0.0, 0.0])
rb = traj(b_spec(par), thB, H, 7)
print("   t    | J_G  D / B      | FPR  D / B      | J_V  D / B      | alpha D / B     | C D / B")
for a, b in zip(rd, rb):
    print(f"  {a['t']:.3f} | {a['jg']:.4f} / {b['jg']:.4f} | {a['fpr']:.4f} / {b['fpr']:.4f} | {a['jv']:.4f} / {b['jv']:.4f} | {a['alpha']:+.3f} / {b['alpha']:+.3f} | {a['C']:.3f} / {b['C']:.3f}")
for name, spec, th in (("D", D, thD), ("B", b_spec(par), thB)):
    r = traj(spec, th, T, 31); c = traj(Spec(w, spec.p, m), th, T, 31)
    jg = np.array([x["jg"] for x in r])
    print(f"{name}: J_G {jg[0]:.3f} -> peak {jg.max():.3f} (t={r[int(jg.argmax())]['t']:.1f}) -> {jg[-1]:.3f}; clean {c[-1]['jg']:.3f}; "
          f"norm. shortfall {(c[-1]['jg']-jg[-1])/(c[-1]['jg']-jg[0]):+.2f}")
