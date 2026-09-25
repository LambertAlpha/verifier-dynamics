"""DC6: mean-field Adam (design check). Per coordinate
    theta_i' = g_i / sqrt(g_i^2 + sigma_i^2 / B + eps^2)
sigma_i^2 = per-sample variance of the REINFORCE estimate (V - b_x) d_i log pi, per-context baseline
b_x = E_x[V], computed exactly by enumeration. B = rollouts per step. B -> inf is sign-GD.
Question: do feature-exploit stalls (FNR = 0) exist under Adam, and do they depend on B?"""
import numpy as np
import torch
from scipy.integrate import solve_ivp
from utoy import Spec, UToy, geometry, ng_velocity

EPS = 1e-8

def mf_adam_velocity(toy, th, B):
    t = torch.tensor(th)
    lp = toy.logp(t)
    jac = torch.nan_to_num(torch.autograd.functional.jacobian(toy.logp, t)).numpy()  # (K, R, d)
    P = torch.exp(lp).detach().numpy()
    w = toy.sp.w
    EV = toy.EV
    bx = (P * EV).sum(1)  # per-context mean reward
    gV = np.einsum("k,kr,kri->i", w, P * (EV - bx[:, None]), jac)
    second = np.einsum("k,kr,kri->i", w, P * (EV * (1 - 2 * bx[:, None]) + bx[:, None] ** 2), jac ** 2)
    var = np.maximum(second - gV ** 2, 0.0)
    M = 1.0 / np.sqrt(gV ** 2 + var / B + EPS ** 2)
    return M * gV, M

def run(toy, th0, T, vel_fn, n=61):
    sol = solve_ivp(lambda t, th: vel_fn(th), (0, T), th0, rtol=1e-7, atol=1e-10, dense_output=True, method="LSODA")
    out = []
    for t in np.linspace(0, T, n):
        th = sol.sol(t)
        jg, jv, fpr, fnr = [x.item() for x in toy.J(torch.tensor(th))]
        out.append((t, jg, fpr))
    return out

def main() -> None:
    K, m = 2, 3
    w = np.array([0.5, 0.5]); p = np.full(K, 1 - 1e-9)
    T = 15.0
    th0 = np.array([-2.0, -1.0, -3.0, -1.0, 0.0, 0.0])
    Y = UToy(Spec(w, p, m, event="single", trig=np.ones(K)))
    CL = UToy(Spec(w, p, m))
    print("Single-feature exploit (s0 = 0.27) vs clean, J_G(T) at T = 15 (time units):")
    for label, vf_y, vf_c in [
        ("NG", lambda th: ng_velocity(Y, th)[0], lambda th: ng_velocity(CL, th)[0]),
        ("Adam B=inf (sign-GD)", lambda th: mf_adam_velocity(Y, th, 1e12)[0], lambda th: mf_adam_velocity(CL, th, 1e12)[0]),
        ("MF-Adam B=512", lambda th: mf_adam_velocity(Y, th, 512)[0], lambda th: mf_adam_velocity(CL, th, 512)[0]),
        ("MF-Adam B=64", lambda th: mf_adam_velocity(Y, th, 64)[0], lambda th: mf_adam_velocity(CL, th, 64)[0]),
        ("MF-Adam B=8", lambda th: mf_adam_velocity(Y, th, 8)[0], lambda th: mf_adam_velocity(CL, th, 8)[0]),
    ]:
        ry, rc = run(Y, th0, T, vf_y), run(CL, th0, T, vf_c)
        j0 = ry[0][1]
        print(f"  {label:22s} exploit J_G(T) {ry[-1][1]:.3f} FPR(T) {ry[-1][2]:.3f} | clean J_G(T) {rc[-1][1]:.3f} | normalized shortfall {(rc[-1][1]-ry[-1][1])/(rc[-1][1]-j0):+.3f}")


if __name__ == "__main__":
    main()
