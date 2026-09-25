# E001 summary

| prediction | run | quantity | observed | expected | tol | pass |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | IC1/natural | max |generic - closed-form ODE| over (q, s) | 1.59153e-10 | 0 | 1e-07 | yes |
| P1 | IC1/vanilla | max |generic - closed-form ODE| over (q, s) | 6.45207e-11 | 0 | 1e-07 | yes |
| P1 | IC2/natural | max |generic - closed-form ODE| over (q, s) | 1.76914e-10 | 0 | 1e-07 | yes |
| P1 | IC2/vanilla | max |generic - closed-form ODE| over (q, s) | 1.14261e-10 | 0 | 1e-07 | yes |
| P1 | IC3/natural | max |generic - closed-form ODE| over (q, s) | 1.59153e-10 | 0 | 1e-07 | yes |
| P1 | IC3/vanilla | max |generic - closed-form ODE| over (q, s) | 6.45207e-11 | 0 | 1e-07 | yes |
| P2 | IC1/natural | max |s/q - s0/q0| | 1.62213e-11 | 0 | 1e-08 | yes |
| P2 | IC1/vanilla | max |I(t) - I(0)| | 4.9819e-10 | 0 | 1e-06 | yes |
| P2 | IC1/vanilla | I(0) vs registered | -100.414 | -100.414 | 1e-06 | yes |
| P2 | IC2/natural | max |s/q - s0/q0| | 3.55271e-15 | 0 | 1e-08 | yes |
| P2 | IC2/vanilla | max |I(t) - I(0)| | 0 | 0 | 1e-06 | yes |
| P2 | IC2/vanilla | I(0) vs registered | 0 | 0 | 1e-06 | yes |
| P2 | IC3/natural | max |s/q - s0/q0| | 1.45992e-08 | 0 | 1e-08 | **NO** |
| P2 | IC3/vanilla | max |I(t) - I(0)| | 4.9819e-10 | 0 | 1e-06 | yes |
| P2 | IC3/vanilla | I(0) vs registered | 100.414 | 100.414 | 1e-06 | yes |
| P3 | IC1/natural | (q, s) at T | [1, 0.0333333] | [1, 0.0333333] | 1e-08 | yes |
| P3 | IC1/natural | Delta at T | 2.38578e-12 | 0 | 1e-08 | yes |
| P3 | IC2/natural | max |s - q| (diagonal) | 2.66454e-15 | 0 | 1e-08 | yes |
| P3 | IC2/natural | q, s non-decreasing | True | True |  | yes |
| P3 | IC2/natural | (q, s, Delta) at T (limit not reached) | [0.952204, 0.952204, 0.0455119] |  |  | info |
| P3 | IC3/natural | (q, s) at T | [0.0333333, 1] | [0.0333333, 1] | 1e-08 | yes |
| P3 | IC3/natural | Delta at T | 0.966667 | 0.966667 | 1e-08 | yes |
| P4 | IC1/vanilla | s when q = 0.9999 (t = 1.014e+04) | 0.0113971 | 0.011397 | 1e-05 | yes |
| P4 | IC1/vanilla | natural-flow s at the same q | 0.03333 |  |  | info |
| P4 | IC2/vanilla | max |s - q| (diagonal) | 0 | 0 | 1e-05 | yes |
| P4 | IC3/vanilla | q when s = 0.9999 (t = 1.014e+04) | 0.0113971 | 0.011397 | 1e-05 | yes |
| P4 | IC3/vanilla | natural-flow q at the same s | 0.03333 |  |  | info |
| P5 | IC1/natural | sign pattern of dDelta/dt | +- | +- |  | yes |
| P5 | IC1/natural | no shrinking -> growing switch | True | True |  | yes |
| P5 | IC1/natural | (q, s) at first sign switch (t = 0.8587) | [0.5, 0.0166667] | [0.5, 0.0166667] | 1e-06 | yes |
| P5 | IC1/vanilla | sign pattern of dDelta/dt | - | - |  | yes |
| P5 | IC1/vanilla | no shrinking -> growing switch | True | True |  | yes |
| P5 | IC2/natural | sign pattern of dDelta/dt | +- | +- |  | yes |
| P5 | IC2/natural | no shrinking -> growing switch | True | True |  | yes |
| P5 | IC2/natural | (q, s) at first sign switch (t = 3.086) | [0.5, 0.5] | [0.5, 0.5] | 1e-06 | yes |
| P5 | IC2/vanilla | sign pattern of dDelta/dt | +- | +- |  | yes |
| P5 | IC2/vanilla | no shrinking -> growing switch | True | True |  | yes |
| P5 | IC2/vanilla | (q, s) at first sign switch (t = 17.75) | [0.5, 0.5] | [0.5, 0.5] | 1e-06 | yes |
| P5 | IC3/natural | sign pattern of dDelta/dt | + | + |  | yes |
| P5 | IC3/natural | no shrinking -> growing switch | True | True |  | yes |
| P5 | IC3/vanilla | sign pattern of dDelta/dt | +- | +- |  | yes |
| P5 | IC3/vanilla | no shrinking -> growing switch | True | True |  | yes |
| P5 | IC3/vanilla | (q, s) at first sign switch (t = 7851) | [0.0113643, 0.999871] | [0.011364, 0.999871] | 1e-06 | yes |
| P6 | IC1/natural | A_F: max |err| / (atol + rtol|x|) | 0.0449274 | <= 1 | 1 | yes |
| P6 | IC1/natural | alpha_F: max |err| / (atol + rtol|x|) | 6.69205e-09 | <= 1 | 1 | yes |
| P6 | IC1/natural | C_F: max |err| / (atol + rtol|x|) | 4.5985e-07 | <= 1 | 1 | yes |
| P6 | IC1/natural | A_E: max |err| / (atol + rtol|x|) | 1.26753e-06 | <= 1 | 1 | yes |
| P6 | IC1/natural | alpha_E: max |err| / (atol + rtol|x|) | 0.186872 | <= 1 | 1 | yes |
| P6 | IC1/natural | C_E: max |err| / (atol + rtol|x|) | 9.39004e-08 | <= 1 | 1 | yes |
| P6 | IC1/natural | (alpha_F, C_F, C_E) at T | [-0.0333333, 1.28478e-11, 2.30626e-12] |  |  | info |
| P6 | IC1/vanilla | A_F: max |err| / (atol + rtol|x|) | 6.48905e-05 | <= 1 | 1 | yes |
| P6 | IC1/vanilla | alpha_F: max |err| / (atol + rtol|x|) | 9.02256e-09 | <= 1 | 1 | yes |
| P6 | IC1/vanilla | C_F: max |err| / (atol + rtol|x|) | 2.77671e-07 | <= 1 | 1 | yes |
| P6 | IC1/vanilla | A_E: max |err| / (atol + rtol|x|) | 1.53346e-06 | <= 1 | 1 | yes |
| P6 | IC1/vanilla | alpha_E: max |err| / (atol + rtol|x|) | 7.56638e-06 | <= 1 | 1 | yes |
| P6 | IC1/vanilla | C_E: max |err| / (atol + rtol|x|) | 3.06238e-08 | <= 1 | 1 | yes |
| P6 | IC1/vanilla | (alpha_F, C_F, C_E) at T | [-0.012019, 1.10284e-07, 1.20176e-08] |  |  | info |
| P6 | IC2/natural | A_F: max |err| / (atol + rtol|x|) | 9.88607e-10 | <= 1 | 1 | yes |
| P6 | IC2/natural | alpha_F: max |err| / (atol + rtol|x|) | 1.47785e-09 | <= 1 | 1 | yes |
| P6 | IC2/natural | C_F: max |err| / (atol + rtol|x|) | 6.7834e-09 | <= 1 | 1 | yes |
| P6 | IC2/natural | A_E: max |err| / (atol + rtol|x|) | 1.77625e-09 | <= 1 | 1 | yes |
| P6 | IC2/natural | alpha_E: max |err| / (atol + rtol|x|) | 1.52318e-09 | <= 1 | 1 | yes |
| P6 | IC2/natural | C_E: max |err| / (atol + rtol|x|) | 6.50146e-09 | <= 1 | 1 | yes |
| P6 | IC2/natural | (alpha_F, C_F, C_E) at T | [-0.952204, 0.0101967, 0.00217531] |  |  | info |
| P6 | IC2/vanilla | A_F: max |err| / (atol + rtol|x|) | 7.44498e-08 | <= 1 | 1 | yes |
| P6 | IC2/vanilla | alpha_F: max |err| / (atol + rtol|x|) | 1.38622e-09 | <= 1 | 1 | yes |
| P6 | IC2/vanilla | C_F: max |err| / (atol + rtol|x|) | 2.42617e-07 | <= 1 | 1 | yes |
| P6 | IC2/vanilla | A_E: max |err| / (atol + rtol|x|) | 1.31458e-07 | <= 1 | 1 | yes |
| P6 | IC2/vanilla | alpha_E: max |err| / (atol + rtol|x|) | 1.24732e-09 | <= 1 | 1 | yes |
| P6 | IC2/vanilla | C_E: max |err| / (atol + rtol|x|) | 5.39587e-08 | <= 1 | 1 | yes |
| P6 | IC2/vanilla | (alpha_F, C_F, C_E) at T | [-0.999292, 1.88368e-05, 5.01078e-07] |  |  | info |
| P6 | IC3/natural | A_F: max |err| / (atol + rtol|x|) | 4.63612e-10 | <= 1 | 1 | yes |
| P6 | IC3/natural | alpha_F: max |err| / (atol + rtol|x|) | 1.00081e-09 | <= 1 | 1 | yes |
| P6 | IC3/natural | C_F: max |err| / (atol + rtol|x|) | 0.043543 | <= 1 | 1 | yes |
| P6 | IC3/natural | A_E: max |err| / (atol + rtol|x|) | 8.6067e-10 | <= 1 | 1 | yes |
| P6 | IC3/natural | alpha_E: max |err| / (atol + rtol|x|) | 1.56862e-08 | <= 1 | 1 | yes |
| P6 | IC3/natural | C_E: max |err| / (atol + rtol|x|) | 1.22528e-06 | <= 1 | 1 | yes |
| P6 | IC3/natural | (alpha_F, C_F, C_E) at T | [-1, 8.17811e-06, 6.91877e-11] |  |  | info |
| P6 | IC3/vanilla | A_F: max |err| / (atol + rtol|x|) | 6.54146e-10 | <= 1 | 1 | yes |
| P6 | IC3/vanilla | alpha_F: max |err| / (atol + rtol|x|) | 1.09658e-09 | <= 1 | 1 | yes |
| P6 | IC3/vanilla | C_F: max |err| / (atol + rtol|x|) | 6.48194e-05 | <= 1 | 1 | yes |
| P6 | IC3/vanilla | A_E: max |err| / (atol + rtol|x|) | 1.07168e-09 | <= 1 | 1 | yes |
| P6 | IC3/vanilla | alpha_E: max |err| / (atol + rtol|x|) | 1.48128e-08 | <= 1 | 1 | yes |
| P6 | IC3/vanilla | C_E: max |err| / (atol + rtol|x|) | 1.5153e-06 | <= 1 | 1 | yes |
| P6 | IC3/vanilla | (alpha_F, C_F, C_E) at T | [-0.999999, 0.000993917, 9.99888e-07] |  |  | info |
| P7 | IC1/natural | FD dDelta/dt vs b(a+b)+c^2 (Fisher): max |err| / allowed | 0.000539195 | <= 1 | 1 | yes |
| P7 | IC1/vanilla | FD dDelta/dt vs b(a+b)+c^2 (Euclidean): max |err| / allowed | 0.000219966 | <= 1 | 1 | yes |
| P7 | IC1/vanilla | fraction of points where Fisher-metric sign is wrong | 0.396509 |  |  | info |
| P7 | IC1/vanilla | Fisher-metric prediction at t=0 has opposite sign | [0.002772, -0.000388565] | opposite signs |  | yes |
| P7 | IC2/natural | FD dDelta/dt vs b(a+b)+c^2 (Fisher): max |err| / allowed | 0.00210621 | <= 1 | 1 | yes |
| P7 | IC2/vanilla | FD dDelta/dt vs b(a+b)+c^2 (Euclidean): max |err| / allowed | 0.00203205 | <= 1 | 1 | yes |
| P7 | IC2/vanilla | fraction of points where Fisher-metric sign is wrong | 0 |  |  | info |
| P7 | IC3/natural | FD dDelta/dt vs b(a+b)+c^2 (Fisher): max |err| / allowed | 0.000392615 | <= 1 | 1 | yes |
| P7 | IC3/vanilla | FD dDelta/dt vs b(a+b)+c^2 (Euclidean): max |err| / allowed | 0.000220365 | <= 1 | 1 | yes |
| P7 | IC3/vanilla | fraction of points where Fisher-metric sign is wrong | 0.234414 |  |  | info |

| run | field evals | wall s | q(T) | s(T) | Δ(T) |
| --- | --- | --- | --- | --- | --- |
| IC1/natural | 311 | 1.3 | 0.9999999999 | 0.03333333333 | 2.38578321e-12 |
| IC1/vanilla | 1265 | 0.4 | 0.9999989879 | 0.01201897301 | 1.216384077e-08 |
| IC2/natural | 383 | 0.2 | 0.9522035569 | 0.9522035569 | 0.04551194317 |
| IC2/vanilla | 1355 | 0.4 | 0.9992918803 | 0.9992918803 | 0.00070761829 |
| IC3/natural | 311 | 0.2 | 0.03333333333 | 0.9999999999 | 0.9666666666 |
| IC3/vanilla | 1265 | 0.3 | 0.01201897301 | 0.9999989879 | 0.9879800271 |
