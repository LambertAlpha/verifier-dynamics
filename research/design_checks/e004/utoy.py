"""E004 DESIGN-PHASE CHECK model (scratch; not experiment code, not part of vdyn).

Unified multi-context toy ("U-toy"). Contexts x = 1..K with weights w. Per context the response
is (s, xi, z):
  s  in {SOLVE, HACK, OTHER}: softmax logits (u_x, h, 0)
  xi ~ Bern(p_x) if s = SOLVE (task success; environment randomness, not controlled by theta)
  z  in {0,1}^m: independent features, logit_j = phi_j + lam_j * mean(u)   (lam = capability coupling)
Gold G = 1{s = SOLVE, xi = 1}. Verifier (fresh coins, expected value):
  x in deleted set:            E[V] = v0
  G = 1:                        E[V] = 1 - fn_x
  G = 0:                        E[V] = 1 - (1-fp_x)(1 - trig_x 1_E(z))(1 - rho_x 1{HACK})(1 - beta_x 1{SOLVE, fail})
Parameters theta = (u_1..u_K, h, phi_1..phi_m); the dimension never depends on the family.
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


@dataclass
class Spec:
    w: np.ndarray                 # (K,) context weights
    p: np.ndarray                 # (K,) success prob of a SOLVE attempt
    m: int = 3
    event: str = "none"           # none | single | and2 | and3 | or2
    lam: np.ndarray | None = None  # (m,) capability coupling
    fn: np.ndarray | None = None  # (K,) false-negative coin
    fp: np.ndarray | None = None  # (K,) false-positive coin
    trig: np.ndarray | None = None  # (K,) event active in context
    rho: np.ndarray | None = None   # (K,) hack acceptance
    beta: np.ndarray | None = None  # (K,) attempt credit (benign amplification)
    deleted: np.ndarray | None = None  # (K,) bool
    v0: float = 1.0
    K: int = field(init=False)

    def __post_init__(self):
        self.K = len(self.w)
        z = np.zeros(self.K)
        self.lam = np.zeros(self.m) if self.lam is None else self.lam
        for name in ("fn", "fp", "trig", "rho", "beta"):
            if getattr(self, name) is None:
                setattr(self, name, z.copy())
        self.deleted = np.zeros(self.K, bool) if self.deleted is None else self.deleted


def _event(kind: str, z: np.ndarray) -> np.ndarray:
    if kind == "none":
        return np.zeros(len(z))
    if kind == "single":
        return z[:, 0].astype(float)
    if kind == "and2":
        return (z[:, 0] & z[:, 1]).astype(float)
    if kind == "and3":
        return (z[:, 0] & z[:, 1] & z[:, 2]).astype(float)
    if kind == "or2":
        return (z[:, 0] | z[:, 1]).astype(float)
    raise ValueError(kind)


class UToy:
    def __init__(self, spec: Spec):
        self.sp = spec
        K, m = spec.K, spec.m
        self.d = K + 1 + m
        zs = np.array(list(itertools.product([0, 1], repeat=m)), dtype=int)  # (2^m, m)
        # outcome grid per context: s (3) x xi (2) x z (2^m)
        rows = []
        for s in range(3):
            for xi in range(2):
                for zi in range(len(zs)):
                    rows.append((s, xi, zi))
        self.rows = np.array(rows)
        self.zs = zs
        ev = _event(spec.event, zs)
        self.G = np.zeros((K, len(rows)))
        self.EV = np.zeros((K, len(rows)))
        for x in range(K):
            for r, (s, xi, zi) in enumerate(rows):
                g = float(s == 0 and xi == 1)
                self.G[x, r] = g
                if spec.deleted[x]:
                    self.EV[x, r] = spec.v0
                elif g:
                    self.EV[x, r] = 1 - spec.fn[x]
                else:
                    keep = (1 - spec.fp[x]) * (1 - spec.trig[x] * ev[zi])
                    keep *= 1 - spec.rho[x] * (s == 1)
                    keep *= 1 - spec.beta[x] * (s == 0 and xi == 0)
                    self.EV[x, r] = 1 - keep
        self.Gt = torch.tensor(self.G)
        self.EVt = torch.tensor(self.EV)
        self.wt = torch.tensor(spec.w)
        self.pt = torch.tensor(spec.p)
        self.lamt = torch.tensor(spec.lam)
        self.zst = torch.tensor(zs, dtype=torch.float64)
        self.s_idx = torch.tensor(self.rows[:, 0])
        self.xi = torch.tensor(self.rows[:, 1], dtype=torch.float64)
        self.z_idx = torch.tensor(self.rows[:, 2])

    def logp(self, theta: torch.Tensor) -> torch.Tensor:
        """log P(outcome | context), shape (K, R)."""
        K, m = self.sp.K, self.sp.m
        u, h, phi = theta[:K], theta[K], theta[K + 1:]
        logits = torch.stack([u, h.expand(K), torch.zeros(K)], dim=1)  # (K, 3)
        ls = torch.log_softmax(logits, dim=1)[:, self.s_idx]  # (K, R)
        solve = (self.s_idx == 0).to(torch.float64)
        lp1 = torch.log(self.pt)[:, None].expand(-1, len(self.xi))
        lp0 = torch.log1p(-self.pt)[:, None].expand(-1, len(self.xi))
        lxi = solve * torch.where(self.xi[None, :] > 0, lp1, lp0)
        # non-solve rows: xi is fixed at 0 with probability 1 (rows with xi=1 are impossible)
        imposs = (1 - solve) * self.xi
        lxi = lxi + torch.where(imposs > 0, torch.tensor(-1e300), torch.tensor(0.0))
        zl = phi + self.lamt * u.mean()  # (m,)
        lz = (self.zst * torch.nn.functional.logsigmoid(zl) + (1 - self.zst) * torch.nn.functional.logsigmoid(-zl)).sum(1)
        return ls + lxi + lz[self.z_idx][None, :]

    def probs(self, theta):
        return torch.exp(self.logp(theta))

    def J(self, theta):
        P = self.probs(theta)
        jg = (self.wt[:, None] * P * self.Gt).sum()
        jv = (self.wt[:, None] * P * self.EVt).sum()
        wrong = (self.wt[:, None] * P * (1 - self.Gt)).sum()
        fpm = (self.wt[:, None] * P * (1 - self.Gt) * self.EVt).sum()
        fnm = (self.wt[:, None] * P * self.Gt * (1 - self.EVt)).sum()
        return jg, jv, fpm / wrong, fnm / jg

    def ctx_gold(self, theta):
        P = self.probs(theta)
        return (self.wt[:, None] * P * self.Gt).sum(1)  # (K,)

    def grads(self, theta_np):
        th = torch.tensor(theta_np, requires_grad=True)
        jg, jv, fpr, fnr = self.J(th)
        gG = torch.autograd.grad(jg, th, retain_graph=True)[0].numpy()
        gV = torch.autograd.grad(jv, th, retain_graph=True)[0].numpy()
        gF = torch.autograd.grad(fpr, th, retain_graph=True)[0].numpy()
        return gG, gV, gF, (jg.item(), jv.item(), fpr.item(), fnr.item())

    def fisher(self, theta_np):
        th = torch.tensor(theta_np)
        jac = torch.autograd.functional.jacobian(self.logp, th)  # (K, R, d)
        P = self.probs(th).detach()
        wP = self.wt[:, None] * P
        jac = torch.nan_to_num(jac)
        F = torch.einsum("kr,kri,krj->ij", wP, jac, jac)
        return F.numpy()

    def ctx_gold_grads(self, theta_np):
        th = torch.tensor(theta_np)
        return torch.autograd.functional.jacobian(self.ctx_gold, th).numpy()  # (K, d)


def geometry(gG, gV, M, ctx=None):
    """(A, alpha, C, C_in, C_out) in metric M (PSD). C_in/C_out: residual inside/outside
    span{M^{1/2} g_G,x}."""
    w, V = np.linalg.eigh(M)
    Mh = V @ np.diag(np.sqrt(np.clip(w, 0, None))) @ V.T
    hG, hV = Mh @ gG, Mh @ gV
    he = hV - hG
    A2 = hG @ hG
    alpha = he @ hG / A2 if A2 > 0 else np.nan
    r = he - (alpha if A2 > 0 else 0.0) * hG
    C = np.linalg.norm(r)
    out = {"A": np.sqrt(A2), "alpha": alpha, "C": C}
    if ctx is not None:
        B = (Mh @ ctx.T)  # (d, K)
        Q, _ = np.linalg.qr(B)
        r_in = Q @ (Q.T @ r)
        out["C_in"], out["C_out"] = np.linalg.norm(r_in), np.linalg.norm(r - r_in)
    return out


def ng_velocity(toy: UToy, theta, damp=1e-10):
    gG, gV, gF, js = toy.grads(theta)
    F = toy.fisher(theta)
    Fi = np.linalg.inv(F + damp * np.trace(F) / len(F) * np.eye(len(F)))
    return Fi @ gV, dict(gG=gG, gV=gV, gF=gF, js=js, F=F, Fi=Fi)
