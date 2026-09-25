"""DC7: preference inversion (displacement). Hack acceptance rho exceeds the success rate p_x of a
genuine attempt on hard contexts, so the verifier prefers hacking there. Shared hack logit h couples
contexts. FNR = 0 (failed attempts are true negatives). Outcome under NG and MF-Adam."""
import numpy as np
import torch
from utoy import Spec, UToy, ng_velocity
from dc6_mfadam import mf_adam_velocity, run

K, m = 2, 3
w = np.array([0.5, 0.5])
T = 15.0
th0 = np.array([0.0, 0.0, -2.0, 0.0, 0.0, 0.0])
cases = {
    "no inversion (rho 0.6 < p = 0.9, 0.8)": Spec(w, np.array([0.9, 0.8]), m, rho=np.full(K, 0.6)),
    "inversion on hard ctx (p = 0.9, 0.3; rho 0.6)": Spec(w, np.array([0.9, 0.3]), m, rho=np.full(K, 0.6)),
    "inversion on both (p = 0.5, 0.3; rho 0.7)": Spec(w, np.array([0.5, 0.3]), m, rho=np.full(K, 0.7)),
}
for label, spec in cases.items():
    toy = UToy(spec); clean = UToy(Spec(w, spec.p, m))
    for opt, vf, vc in (("NG", lambda th: ng_velocity(toy, th)[0], lambda th: ng_velocity(clean, th)[0]),
                        ("MF-Adam B=64", lambda th: mf_adam_velocity(toy, th, 64)[0], lambda th: mf_adam_velocity(clean, th, 64)[0])):
        r, c = run(toy, th0, T, vf), run(clean, th0, T, vc)
        jg = np.array([x[1] for x in r])
        print(f"{label:48s} {opt:13s} J_G {jg[0]:.3f} -> peak {jg.max():.3f} (t={r[int(jg.argmax())][0]:.1f}) -> {jg[-1]:.3f}; clean {c[-1][1]:.3f}; "
              f"norm. shortfall {(c[-1][1]-jg[-1])/(c[-1][1]-jg[0]):+.2f}; drop from peak {jg.max()-jg[-1]:.3f}")
