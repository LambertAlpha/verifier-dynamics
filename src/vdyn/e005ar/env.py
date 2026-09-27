"""E005a-R calibration environment and exact oracle (research/09_e005ar_design.md §4).

Parameter space R^d = S* (first r coordinates, behavior) ⊕ nuisance (surface form). A softmax
policy over 4 behavior classes on 8 prompt types, evaluated at its base point; surface features
w ~ N(0, Λ) independent of the class, whose score is w itself. Gold G = 1[b = 0]. Verifier

    V = s_V Bern(EV_rho(x, b)) + λ 1[v^T w > 0],   EV_rho = (1 - rho)(a' G + c0) + rho EV_con.

All targets are reward-level policy gradients in the identity metric (canonical coordinates; every
method is rotation-equivariant, §4). Behavior-block covariances come from Monte Carlo with common
random numbers (dose bisection); the nuisance block is exact.
"""

from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np

ROOT_SEED = 20261201
STREAMS = {"anchors": 0, "design": 1, "test": 2, "design_mc": 3, "test_mc": 4, "theory": 5}
PROMPTS, CLASSES = 8, 4
W_X = np.full(PROMPTS, 1.0 / PROMPTS)
R_GRID = (4, 8, 16)
BASE_TYPES = {"ordinary": (0.3, 0.7), "lowvar": (0.08, 0.2), "lowA": (0.005, 0.02)}
INCORRECT_DIRICHLET = 2.0
D_GRID = (64, 256, 1024)
SPECTRA = ("bulk", "flat", "spiked")
SPECTRUM_LEVEL = {"bulk": 0.25, "flat": 1.0, "spiked": 1.0}
SPIKED_POWER = -0.5
CONSTRUCTIONS = ("shortcut", "deletion", "partial")
ALPHA_CLEAN = (-0.6, -0.3, 0.0)
ALPHA_DOSE = (-0.3, 0.0)
ALPHA_NUIS = (-0.3, 0.0)
ALPHA_PARTIAL = (-0.3,)
DOSES = ("small", "medium", "large")
TAU_ANCHORS = {"small": 0.0030775117222044015, "medium": 0.020992935971183818,
               "large": 0.078780131088946}  # fmt: skip
M_GROUP = 8
BISECTION_ITERS = 40
FUNCTIONAL_VARIANTS = ("main", "sub", "dup", "scale", "const", "surf")


# ------------------------------------------------------------------ base policy
def make_base(r: int, btype: str, seed: np.random.SeedSequence, sid: str) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    lo, hi = BASE_TYPES[btype]
    p0 = rng.uniform(lo, hi, PROMPTS)
    rest = rng.dirichlet([INCORRECT_DIRICHLET] * (CLASSES - 1), PROMPTS) * (1 - p0)[:, None]
    u = rng.standard_normal((PROMPTS, CLASSES, r))
    a1 = float(rng.uniform(0.5, 0.95))
    c0 = float(rng.uniform(0.0, 0.3 * (1 - a1)))
    half_s = rng.permutation(PROMPTS) < PROMPTS // 2
    half_d = rng.permutation(PROMPTS) < PROMPTS // 2
    return {"sid": sid, "r": r, "btype": btype, "p": np.column_stack([p0, rest]).tolist(),
            "u": u.tolist(), "a1": a1, "c0": c0, "half_shortcut": half_s.tolist(),
            "half_deletion": half_d.tolist()}  # fmt: skip


def probs(base: dict[str, Any]) -> np.ndarray:
    return np.asarray(base["p"], dtype=float)


def scores_beh(base: dict[str, Any]) -> np.ndarray:
    """Behavior score s(x, b) = u(x, b) - E_p u(x, .), shape (8, 4, r)."""
    u = np.asarray(base["u"], dtype=float)
    p = probs(base)
    return u - np.einsum("xb,xbr->xr", p, u)[:, None, :]


def fisher_beh(base: dict[str, Any]) -> np.ndarray:
    s = scores_beh(base)
    return np.einsum("x,xb,xbi,xbj->ij", W_X, probs(base), s, s)


def gold() -> np.ndarray:
    g = np.zeros((PROMPTS, CLASSES))
    g[:, 0] = 1.0
    return g


def ev_con(base: dict[str, Any], con: str) -> np.ndarray:
    ev = gold()
    if con == "shortcut":
        ev[np.asarray(base["half_shortcut"]), 1] = 1.0
    elif con == "deletion":
        ev[np.asarray(base["half_deletion"]), :] = 0.5
    elif con == "partial":
        ev[:, 2] = 0.5
    else:
        raise ValueError(con)
    return ev


def ev_table(base: dict[str, Any], con: str | None, rho: float) -> np.ndarray:
    aff = base["a1"] * gold() + base["c0"]
    if rho == 0 or con is None:
        return aff
    return (1 - rho) * aff + rho * ev_con(base, con)


def h_beh(base: dict[str, Any], table: np.ndarray) -> np.ndarray:
    """Exact behavior-block policy gradient of the expected reward table (x, b)."""
    return np.einsum("x,xb,xb,xbr->r", W_X, probs(base), table, scores_beh(base))


def s_for_alpha(base: dict[str, Any], con: str | None, rho: float, alpha: float) -> float:
    """Verifier scale s_V giving exactly alpha: 1 + alpha = s_V[(1-rho) a' + rho <h_con,h_G>/A^2]."""
    hG = h_beh(base, gold())
    lin = (1 - rho) * base["a1"]
    if rho > 0 and con is not None:
        lin += rho * float(h_beh(base, ev_con(base, con)) @ hG) / float(hG @ hG)
    return (1 + alpha) / lin


# ------------------------------------------------------------------ nuisance block
def spectrum(name: str, d: int, r: int, fbar: float) -> np.ndarray:
    n = d - r
    if name in ("bulk", "flat"):
        return np.full(n, SPECTRUM_LEVEL[name] * fbar)
    if name == "spiked":
        s = np.arange(1, n + 1, dtype=float) ** SPIKED_POWER
        return SPECTRUM_LEVEL[name] * fbar * s / s.mean()
    raise ValueError(name)


def surf_dir(v_seed: int, n: int) -> np.ndarray:
    v = np.random.default_rng(v_seed).standard_normal(n)
    return v / np.linalg.norm(v)


def h_surf(lam: np.ndarray, v: np.ndarray) -> np.ndarray:
    """E[1[v^T w > 0] w] for w ~ N(0, diag(lam)) [proved: Λv φ(0) / sqrt(v^T Λ v)]."""
    lv = lam * v
    return lv / np.sqrt(2 * np.pi * float(v @ lv))


# ------------------------------------------------------------------ functional map (R4)
def functional_map(base: dict[str, Any], variant: str, d: int, lam: np.ndarray | None = None,
                   v: np.ndarray | None = None) -> np.ndarray:  # fmt: skip
    """W^1/2 J_f as a (d, q) matrix: zeta = x @ Phi. f = class probabilities on probe prompts,
    W = diag(weight / p) (Fisher-Rao), J rows p_xb s_xb^T (behavior block)."""
    r, p, s = base["r"], probs(base), scores_beh(base)

    def cols(prompts: list[int], wt: float) -> np.ndarray:
        return np.concatenate([np.sqrt(wt * p[x])[None, :] * s[x].T for x in prompts], axis=1)

    if variant in ("main", "scale", "const", "surf"):
        core = cols(list(range(PROMPTS)), 1.0 / PROMPTS)
    elif variant == "sub":
        core = cols([0, 1, 2, 3], 0.25)
    elif variant == "dup":
        core = cols([0, 0, 0] + list(range(1, PROMPTS)), 1.0 / (PROMPTS + 2))
    else:
        raise ValueError(variant)
    if variant == "scale":
        core = 10.0 * core
    Phi = np.zeros((d, core.shape[1]))
    Phi[:r] = core
    if variant == "const":
        Phi = np.concatenate([Phi, np.zeros((d, PROMPTS))], axis=1)
    if variant == "surf":
        assert lam is not None and v is not None
        extra = np.zeros((d, PROMPTS))
        # P(v^T w > 0 | x) = 1/2; Bernoulli Fisher weight (1/8)/(1/4); gradient E[1[.] w]
        extra[r:] = np.sqrt((1.0 / PROMPTS) / 0.25) * h_surf(lam, v)[:, None]
        Phi = np.concatenate([Phi, extra], axis=1)
    return Phi


# ------------------------------------------------------------------ point context and oracle
@dataclass
class Context:
    base: dict[str, Any]
    con: str | None
    rho: float
    sV: float
    lam_bonus: float
    d: int
    spectrum_name: str
    v_seed: int
    r: int = field(init=False)
    p: np.ndarray = field(init=False)
    S: np.ndarray = field(init=False)
    EV: np.ndarray = field(init=False)
    lam_spec: np.ndarray = field(init=False)
    v: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.r = self.base["r"]
        self.p = probs(self.base)
        self.S = scores_beh(self.base)
        self.EV = ev_table(self.base, self.con, self.rho)
        fbar = float(np.trace(fisher_beh(self.base))) / self.r
        self.lam_spec = spectrum(self.spectrum_name, self.d, self.r, fbar)
        self.v = surf_dir(self.v_seed, self.d - self.r)

    @classmethod
    def build(cls, base: dict[str, Any], con: str | None, rho: float, sV: float, lam_bonus: float,
              d: int, spectrum_name: str, v_seed: int) -> "Context":  # fmt: skip
        return cls(base, con, rho, sV, lam_bonus, d, spectrum_name, v_seed)

    @classmethod
    def of_point(cls, point: dict[str, Any], base: dict[str, Any]) -> "Context":
        return cls(base, point["construction"], point["rho"], point["sV"], point["lam_bonus"],
                   point["d"], point["spectrum"], point["v_seed"])  # fmt: skip


def geometry(hG: np.ndarray, he: np.ndarray) -> dict[str, float]:
    A2, P, Q = float(hG @ hG), float(he @ hG), float(he @ he)
    if A2 <= 0:
        return {"A2": A2, "alpha": float("nan"), "C2": Q}
    return {"A2": A2, "alpha": P / A2, "C2": max(Q - P**2 / A2, 0.0)}


def oracle(ctx: Context) -> dict[str, Any]:
    r, d = ctx.r, ctx.d
    hG = np.zeros(d)
    hG[:r] = h_beh(ctx.base, gold())
    hV = np.zeros(d)
    hV[:r] = ctx.sV * h_beh(ctx.base, ctx.EV)
    hV[r:] = ctx.lam_bonus * h_surf(ctx.lam_spec, ctx.v)
    he = hV - hG
    he_beh = he.copy()
    he_beh[r:] = 0.0
    full, beh = geometry(hG, he), geometry(hG, he_beh)
    Phi = functional_map(ctx.base, "main", d)
    fun = geometry(hG @ Phi, he @ Phi)
    return {"h_G": hG, "h_e": he, "h_e_beh": he_beh, "A2": full["A2"], "alpha": full["alpha"],
            "C2_full": full["C2"], "C2_beh": beh["C2"], "C2_f": fun["C2"],
            "A2_f": fun["A2"], "alpha_f": fun["alpha"],
            "nuis_err2": float(he[r:] @ he[r:])}  # fmt: skip


# ------------------------------------------------------------------ behavior-block Monte Carlo
def crn(base: dict[str, Any], seed: np.random.SeedSequence, n_groups: int,
        m: int = M_GROUP) -> dict[str, np.ndarray]:  # fmt: skip
    """Common random numbers for behavior-block statistics: prompt, class, verifier coin, bonus."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, PROMPTS, n_groups)
    cp = np.cumsum(probs(base), axis=1)
    uc = rng.random((n_groups, m))
    b = np.minimum((uc[..., None] > cp[x][:, None, :]).sum(-1), CLASSES - 1)
    return {"x": x, "b": b, "uz": rng.random((n_groups, m)),
            "B": (rng.random((n_groups, m)) < 0.5).astype(float)}  # fmt: skip


def rloo_adv(rew: np.ndarray) -> np.ndarray:
    m = rew.shape[-1]
    return rew - (rew.sum(-1, keepdims=True) - rew) / (m - 1)


def behavior_moments(base: dict[str, Any], EV: np.ndarray, sV: float, lam_bonus: float,
                     c: dict[str, np.ndarray], chunk: int = 50000) -> dict[str, Any]:  # fmt: skip
    """Group covariances of the behavior-block RLOO contributions (G, e = V - G) and per-rollout
    advantage moments (the nuisance block is (E a^2 / m) Λ when the bonus is off)."""
    S = scores_beh(base)
    x, b = c["x"], c["b"]
    n, m = b.shape
    G = (b == 0).astype(float)
    V = sV * (c["uz"] < EV[x[:, None], b]).astype(float) + lam_bonus * c["B"]
    aG, ae = rloo_adv(G), rloo_adv(V) - rloo_adv(G)
    r = S.shape[-1]
    s1 = np.zeros(2 * r)
    s2 = np.zeros((2 * r, 2 * r))
    for lo in range(0, n, chunk):
        sl = slice(lo, lo + chunk)
        sc = S[x[sl, None], b[sl]]
        z = np.concatenate([np.einsum("gi,gir->gr", aG[sl], sc) / m,
                            np.einsum("gi,gir->gr", ae[sl], sc) / m], axis=1)  # fmt: skip
        s1 += z.sum(0)
        s2 += z.T @ z
    mean = s1 / n
    cov = (s2 - n * np.outer(mean, mean)) / (n - 1)
    return {"G": cov[:r, :r], "e": cov[r:, r:], "eG": cov[r:, :r],
            "aGG": float((aG * aG).mean()), "aee": float((ae * ae).mean()),
            "aGe": float((aG * ae).mean())}  # fmt: skip


def _tau_stats(base: dict[str, Any], EV: np.ndarray, sV: float, lam_bonus: float,
               mom: dict[str, Any], m: int) -> dict[str, float]:  # fmt: skip
    hG = h_beh(base, gold())
    he = sV * h_beh(base, EV) - hG
    A2 = float(hG @ hG)
    u = hG / np.sqrt(A2)
    alpha = float(he @ hG) / A2
    c = he - alpha * hG
    Sd = mom["e"] - alpha * (mom["eG"] + mom["eG"].T) + alpha**2 * mom["G"]
    Pp = np.eye(len(u)) - np.outer(u, u)
    Sp = Pp @ Sd @ Pp
    tr2 = float(np.trace(Sp @ Sp))
    C2 = float(c @ c)
    return {"tau_beh": C2 / np.sqrt(2 * tr2), "trS_perp_beh": float(np.trace(Sp)),
            "trS2_perp_beh": tr2, "cSc": float(c @ Sd @ c), "u_Sd_u": float(u @ Sd @ u),
            "trS_G": float(np.trace(mom["G"])),
            "kappa_delta": (mom["aee"] - 2 * alpha * mom["aGe"] + alpha**2 * mom["aGG"]) / m,
            "bonus_on": float(lam_bonus != 0)}  # fmt: skip


def behavior_stats(ctx: Context, m: int, n_groups: int,
                   seed: np.random.SeedSequence) -> dict[str, float]:  # fmt: skip
    c = crn(ctx.base, seed, n_groups, m)
    mom = behavior_moments(ctx.base, ctx.EV, ctx.sV, ctx.lam_bonus, c)
    return _tau_stats(ctx.base, ctx.EV, ctx.sV, ctx.lam_bonus, mom, m)


def solve_dose(base: dict[str, Any], con: str, alpha: float, tau_target: float, m: int,
               n_groups: int, seed: np.random.SeedSequence,
               c: dict[str, np.ndarray] | None = None) -> dict[str, float] | None:  # fmt: skip
    """rho in (0, 1] with behavior detectability tau(rho) = target at fixed alpha (bisection with
    common random numbers). None if tau(1) < target."""
    cc = c if c is not None else crn(base, seed, n_groups, m)

    def tau(rho: float) -> float:
        sV = s_for_alpha(base, con, rho, alpha)
        EV = ev_table(base, con, rho)
        return _tau_stats(base, EV, sV, 0.0, behavior_moments(base, EV, sV, 0.0, cc), m)["tau_beh"]

    if tau(1.0) < tau_target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        if tau(mid) < tau_target:
            lo = mid
        else:
            hi = mid
    rho = 0.5 * (lo + hi)
    return {"rho": rho, "sV": s_for_alpha(base, con, rho, alpha), "tau": tau(rho)}


def c_beh_exact(base: dict[str, Any], con: str, alpha: float, rho: float) -> float:
    """Exact behavior C at dose rho with s_V fixing alpha."""
    sV = s_for_alpha(base, con, rho, alpha)
    hG = h_beh(base, gold())
    he = sV * h_beh(base, ev_table(base, con, rho)) - hG
    return float(np.sqrt(geometry(hG, he)["C2"]))


def solve_rho_for_c(base: dict[str, Any], con: str, alpha: float, c_target: float) -> float | None:
    """rho in (0, 1] with exact C_beh = c_target (C_beh is increasing in rho at fixed alpha)."""
    if c_beh_exact(base, con, alpha, 1.0) < c_target:
        return None
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if c_beh_exact(base, con, alpha, mid) < c_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------------ panel
def _v_seed(ss: np.random.SeedSequence) -> int:
    return int(ss.generate_state(1)[0])


def build_panel(split: str, limit_bases: int | None = None, n_groups: int = 200000,
                anchors: dict[str, float] | None = None) -> dict[str, Any]:  # fmt: skip
    tau = anchors or TAU_ANCHORS
    root = np.random.SeedSequence(ROOT_SEED).spawn(6)[STREAMS[split]]
    cells = list(product(R_GRID, BASE_TYPES))
    base_seeds = root.spawn(len(cells))
    nuis_cfg = list(product(D_GRID, SPECTRA))
    bases: dict[str, Any] = {}
    points: list[dict[str, Any]] = []
    dropped: list[str] = []
    for i, (r, btype) in enumerate(cells[:limit_bases] if limit_bases else cells):
        kids = base_seeds[i].spawn(3)
        sid = f"{split[0].upper()}-r{r}-{btype}"
        base = make_base(r, btype, kids[0], sid)
        bases[sid] = base
        c = crn(base, kids[1], n_groups)
        v_seeds = [_v_seed(s) for s in kids[2].spawn(len(nuis_cfg))]
        # Amendment 1: tau-anchor each construction (Monte Carlo), take the median C_beh over
        # constructions as the dose's common C*, then solve rho exactly for C_beh = C* per
        # construction, so matched sets share C_beh across constructions, d and spectra.
        sol: dict[tuple[str, float, str], dict[str, float] | None] = {}
        c_star: dict[tuple[float, str], float | None] = {}
        for alpha, dose in product(ALPHA_DOSE, DOSES):
            anch = {con: solve_dose(base, con, alpha, tau[dose], M_GROUP, n_groups, kids[1], c=c)
                    for con in CONSTRUCTIONS}  # fmt: skip
            cs = [c_beh_exact(base, con, alpha, a["rho"]) for con, a in anch.items()
                  if a is not None]  # fmt: skip
            c_star[(alpha, dose)] = float(np.median(cs)) if cs else None
            for con in CONSTRUCTIONS:
                cstar = c_star[(alpha, dose)]
                rho = None if cstar is None else solve_rho_for_c(base, con, alpha, cstar)
                if rho is None:
                    sol[(con, alpha, dose)] = None
                    dropped.append(f"{sid}|{con}|{alpha}|{dose}")
                else:
                    sol[(con, alpha, dose)] = {"rho": rho,
                                               "sV": s_for_alpha(base, con, rho, alpha)}
        a1 = base["a1"]
        c_med = {alpha: c_star[(alpha, "medium")] for alpha in ALPHA_NUIS}
        specs: list[dict[str, Any]] = []
        for alpha in ALPHA_CLEAN:
            specs.append({"case": "clean", "construction": None, "dose": "null", "alpha": alpha,
                          "rho": 0.0, "sV": (1 + alpha) / a1, "bonus": None,
                          "matched_null": f"null|{sid}|{alpha}", "matched_dose": None})
        for (con, alpha, dose), s in sol.items():
            if s is not None:
                specs.append({"case": "dose", "construction": con, "dose": dose, "alpha": alpha,
                              "rho": s["rho"], "sV": s["sV"], "bonus": None,
                              "matched_null": None, "matched_dose": f"dose|{sid}|{dose}|{alpha}"})
        for alpha in ALPHA_NUIS:
            if c_med[alpha] is not None:
                specs.append({"case": "nuis", "construction": None, "dose": "null",
                              "alpha": alpha, "rho": 0.0, "sV": (1 + alpha) / a1,
                              "bonus": c_med[alpha], "matched_null": f"null|{sid}|{alpha}",
                              "matched_dose": None})
        for con in CONSTRUCTIONS:
            for alpha in ALPHA_PARTIAL:
                s = sol[(con, alpha, "medium")]
                if s is None:
                    continue
                specs.append({"case": "partial", "construction": con, "dose": "medium",
                              "alpha": alpha, "rho": s["rho"], "sV": s["sV"],
                              "bonus": c_star[(alpha, "medium")], "matched_null": None,
                              "matched_dose": f"dose|{sid}|medium|{alpha}"})
        cache: dict[tuple[Any, ...], dict[str, float]] = {}
        for spec in specs:
            for (d, spec_name), vs in zip(nuis_cfg, v_seeds, strict=True):
                lam_bonus = 0.0
                if spec["bonus"] is not None:
                    probe = Context.build(base, spec["construction"], spec["rho"], spec["sV"],
                                          0.0, d, spec_name, vs)  # fmt: skip
                    lam_bonus = spec["bonus"] / float(np.linalg.norm(h_surf(probe.lam_spec,
                                                                            probe.v)))
                ctx = Context.build(base, spec["construction"], spec["rho"], spec["sV"],
                                    lam_bonus, d, spec_name, vs)  # fmt: skip
                o = oracle(ctx)
                key = (spec["construction"], spec["rho"], spec["sV"], lam_bonus)
                if key not in cache:
                    mom = behavior_moments(base, ctx.EV, ctx.sV, lam_bonus, c)
                    cache[key] = _tau_stats(base, ctx.EV, ctx.sV, lam_bonus, mom, M_GROUP)
                st = cache[key]
                orc = {k: v for k, v in o.items() if not k.startswith("h_")}
                orc |= st | {"trLam2": float(ctx.lam_spec @ ctx.lam_spec),
                             "trLam": float(ctx.lam_spec.sum())}  # fmt: skip
                pid = (f"{sid}-{spec['case']}-{spec['construction'] or 'none'}-{spec['dose']}"
                       f"-a{spec['alpha']}-d{d}-{spec_name}")
                points.append({"pid": pid, "sid": sid, "r": r, "btype": btype,
                               "case": spec["case"], "construction": spec["construction"],
                               "dose": spec["dose"], "alpha_target": spec["alpha"], "d": d,
                               "spectrum": spec_name, "rho": spec["rho"], "sV": spec["sV"],
                               "lam_bonus": lam_bonus, "v_seed": vs,
                               "matched_null": spec["matched_null"],
                               "matched_dose": spec["matched_dose"], "oracle": orc})
    return {"split": split, "root_seed": ROOT_SEED, "anchors": tau, "bases": bases,
            "points": points, "dropped": dropped}  # fmt: skip
