"""DC2: benign amplification (B) vs feature exploit (Y) under NG, single context.
Derived: rho = J_V'/J_G' satisfies rho_B = 1 + FPR (1 - J_G)/J_G and rho_Y <= rho_B - FPR at matched
(J_G, FPR) (Cramer-Rao on kappa). Check numerically: for random B, construct a single-feature Y
(the most accessible exploit, kappa^2 = S(1-S)) with the same J_G and FPR and compare rho.
DC3: under NG with FNR = 0, gold should never decline (tabular-in-strategy: J_G' = Cov(G, V) >= 0);
with FNR > 0 and hack acceptance > 1 - FNR, decline should appear."""
import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit
from utoy import Spec, UToy, ng_velocity

rng = np.random.default_rng(1)
w1 = np.array([1.0])

def rho_of(spec, th):
    toy = UToy(spec)
    vel, aux = ng_velocity(toy, th)
    jg, jv, fpr, fnr = aux["js"]
    gV = toy.grads(th)[1]
    return (aux["gV"] @ vel) / (aux["gG"] @ vel), jg, fpr

worst = np.inf
for _ in range(40):
    p, beta = rng.uniform(0.3, 0.95), rng.uniform(0.2, 1.0)
    u, h = rng.normal(-1, 1), rng.normal(-2, 1)
    thB = np.array([u, h, 0.0, 0.0, 0.0])
    rB, jg, fpr = rho_of(Spec(w1, np.array([p]), 3, beta=np.array([beta])), thB)
    # Y: p ~ 1, single feature with s = FPR; solve u_Y so that J_G matches (h fixed)
    s = fpr
    uY = brentq(lambda uu: np.exp(uu) / (np.exp(uu) + np.exp(h) + 1) - jg, -30, 30)
    thY = np.array([uY, h, logit(s), 0.0, 0.0])
    rY, jgY, fprY = rho_of(Spec(w1, np.array([1 - 1e-12]), 3, event="single", trig=np.ones(1)), thY)
    assert abs(jgY - jg) < 1e-8 and abs(fprY - fpr) < 1e-8
    worst = min(worst, (rB - rY) - fpr)
    pred_B = 1 + fpr * (1 - jg) / jg
    assert abs(rB - pred_B) < 1e-6 * pred_B, (rB, pred_B)
print(f"DC2: rho_B formula exact; min over 40 matched pairs of (rho_B - rho_Y) - FPR = {worst:.2e} (>= 0 expected)")

# DC3: decline under NG
from scipy.integrate import solve_ivp
def run(spec, th0, T):
    toy = UToy(spec)
    f = lambda t, th: ng_velocity(toy, th)[0]
    sol = solve_ivp(f, (0, T), th0, rtol=1e-8, atol=1e-10, dense_output=True)
    ts = np.linspace(0, T, 200)
    jg = [toy.J(__import__("torch").tensor(sol.sol(t)))[0].item() for t in ts]
    return ts, np.array(jg)
K = 2; w2 = np.array([0.5, 0.5]); p2 = np.array([1 - 1e-9, 1 - 1e-9])
th0 = np.array([-1.0, -1.5, -0.5, 0.0, 0.0, 0.0])
for label, spec in [("hack rho=0.9, FNR=0", Spec(w2, p2, 3, rho=np.full(K, 0.9))),
                    ("hack rho=0.9, FNR=0.3 (V prefers hack)", Spec(w2, p2, 3, rho=np.full(K, 0.9), fn=np.full(K, 0.3)))]:
    ts, jg = run(spec, th0, 30.0)
    drop = jg.max() - jg[-1]
    print(f"DC3 {label:42s} J_G: start {jg[0]:.3f} peak {jg.max():.3f} (t={ts[jg.argmax()]:.1f}) end {jg[-1]:.3f}  drop from peak {drop:.3f}")
