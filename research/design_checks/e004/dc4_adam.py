"""DC4: infinite-batch RMSprop/Adam-like flow (no momentum):
    theta' = g_V / (sqrt(v) + eps),   v' = (g_V^2 - v) / tau,   v(0) = g_V(0)^2
Metric M_t = diag(1 / (sqrt(v) + eps)) (theory note §1 working approximation).
(a) Does the NG identity (alpha = -FPR, C^2 = (1-J_G) FPR' ...) still hold?  (expected: no)
(b) Capability coupling (a reparameterization, invisible to NG) under Adam: early observables vs outcome.
(c) Hack acceptance with FNR = 0 under Adam: can gold decline?
(d) Clean-run time scale (t95) for anchoring T."""
import numpy as np
import torch
from scipy.integrate import solve_ivp
from utoy import Spec, UToy, geometry, ng_velocity

EPS, TAU = 1e-8, 1.0

def adam_rhs(toy):
    d = toy.d
    def f(t, y):
        th, v = y[:d], y[d:]
        gG, gV, gF, js = toy.grads(th)
        return np.concatenate([gV / (np.sqrt(np.maximum(v, 0)) + EPS), (gV ** 2 - v) / TAU])
    return f

def run_adam(spec, th0, T, n=121):
    toy = UToy(spec)
    g0 = toy.grads(th0)[1]
    sol = solve_ivp(adam_rhs(toy), (0, T), np.concatenate([th0, g0 ** 2 + 1e-12]), rtol=1e-7, atol=1e-10,
                    dense_output=True, method="LSODA")
    rows = []
    for t in np.linspace(0, T, n):
        y = sol.sol(t); th, v = y[:toy.d], y[toy.d:]
        gG, gV, gF, (jg, jv, fpr, fnr) = toy.grads(th)
        M = np.diag(1 / (np.sqrt(np.maximum(v, 0)) + EPS))
        vel = M @ gV
        geo = geometry(gG, gV, M, toy.ctx_gold_grads(th))
        rows.append(dict(t=t, jg=jg, jv=jv, fpr=fpr, jg_dot=gG @ vel, fpr_dot=gF @ vel, **geo))
    return rows

def run_ng(spec, th0, T, n=121):
    toy = UToy(spec)
    sol = solve_ivp(lambda t, th: ng_velocity(toy, th)[0], (0, T), th0, rtol=1e-8, atol=1e-10, dense_output=True)
    rows = []
    for t in np.linspace(0, T, n):
        th = sol.sol(t); vel, aux = ng_velocity(toy, th)
        jg, jv, fpr, fnr = aux["js"]
        geo = geometry(aux["gG"], aux["gV"], aux["Fi"], toy.ctx_gold_grads(th))
        rows.append(dict(t=t, jg=jg, jv=jv, fpr=fpr, jg_dot=aux["gG"] @ vel, fpr_dot=aux["gF"] @ vel, **geo))
    return rows

def main() -> None:
    K, m = 2, 3
    w = np.array([0.5, 0.5]); p = np.full(K, 1 - 1e-9); ones = np.ones(K)
    th0 = np.array([-2.0, -1.0, -3.0, -3.0, -3.0, 0.0])

    # (a) identity under Adam, C-type AND2
    r = run_adam(Spec(w, p, m, event="and2", trig=ones), th0, 2.0, n=5)
    for x in r[1:3]:
        print(f"(a) Adam t={x['t']:.1f}: alpha {x['alpha']:+.4f} vs -FPR {-x['fpr']:+.4f};  C^2 {x['C']**2:.3e} vs (1-J_G)FPR' {(1-x['jg'])*x['fpr_dot']:.3e}")

    # (b) capability coupling: identical t=0 distribution, lam couples exploit features to mean skill
    T = 12.0
    base = Spec(w, p, m, event="and2", trig=ones)
    coup = Spec(w, p, m, event="and2", trig=ones, lam=np.array([1.2, 1.2, 0.0]))
    th0c = th0.copy(); th0c[K + 1:] = th0[K + 1:] - coup.lam * th0[:K].mean()  # same t=0 feature logits
    for opt, runner in (("NG", run_ng), ("Adam", run_adam)):
        a, b = runner(base, th0, T), runner(coup, th0c, T)
        print(f"(b) {opt}: uncoupled end J_G {a[-1]['jg']:.3f} FPR {a[-1]['fpr']:.3f} | coupled end J_G {b[-1]['jg']:.3f} FPR {b[-1]['fpr']:.3f}")
        for i in (0, 1, 3, 6, 12):
            x, y = a[i], b[i]
            print(f"     t={x['t']:5.2f}  J_G {x['jg']:.4f}/{y['jg']:.4f}  FPR {x['fpr']:.5f}/{y['fpr']:.5f}  "
                  f"C {x['C']:.4f}/{y['C']:.4f}  C_out {x['C_out']:.4f}/{y['C_out']:.4f}  alpha {x['alpha']:+.4f}/{y['alpha']:+.4f}")

    # (c) hack acceptance, FNR = 0, under Adam vs NG
    hk = Spec(w, p, m, rho=np.full(K, 0.9))
    for opt, runner in (("NG", run_ng), ("Adam", run_adam)):
        r = runner(hk, np.array([-1.0, -1.5, -0.5, 0.0, 0.0, 0.0]), 30.0)
        jg = np.array([x["jg"] for x in r])
        print(f"(c) hack rho=0.9 FNR=0 {opt}: J_G start {jg[0]:.3f} peak {jg.max():.3f} end {jg[-1]:.3f}; min alpha {min(x['alpha'] for x in r):+.3f}")

    # (d) clean time scale
    cl = Spec(w, p, m)
    for opt, runner in (("NG", run_ng), ("Adam", run_adam)):
        r = runner(cl, th0, 20.0, n=401)
        jg = np.array([x["jg"] for x in r]); ts = np.array([x["t"] for x in r])
        t95 = ts[np.argmax(jg >= jg[0] + 0.95 * (jg[-1] - jg[0]))]
        print(f"(d) clean {opt}: J_G {jg[0]:.3f} -> {jg[-1]:.3f}, t95 = {t95:.2f} time units")


if __name__ == "__main__":
    main()
