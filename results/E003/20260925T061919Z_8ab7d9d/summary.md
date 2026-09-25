# E003 summary (REGISTERED)

| prediction | run | quantity | observed | expected | tol | pass |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | all | gold(0): max - min across structures | 8.67362e-19 | 0 | 1e-12 | yes |
| P1 | all | gold(0) vs registered | [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] | [0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001] | 1e-12 | yes |
| P1 | all | fpr(0): max - min across structures | 3.1225e-17 | 0 | 1e-12 | yes |
| P1 | all | fpr(0) vs registered | [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01] | [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01] | 1e-12 | yes |
| P1 | all | fnr(0): max - min across structures | 0 | 0 | 1e-12 | yes |
| P1 | all | fnr(0) vs registered | [0, 0, 0, 0, 0, 0, 0, 0] | [0, 0, 0, 0, 0, 0, 0, 0] | 1e-12 | yes |
| P1 | all | verifier_accuracy(0): max - min across structures | 1.11022e-16 | 0 | 1e-12 | yes |
| P1 | all | verifier_accuracy(0) vs registered | [0.99001, 0.99001, 0.99001, 0.99001, 0.99001, 0.99001, 0.99001, 0.99001] | [0.99001, 0.99001, 0.99001, 0.99001, 0.99001, 0.99001, 0.99001, 0.99001] | 1e-12 | yes |
| P1 | all | fp_mass(0): max - min across structures | 3.1225e-17 | 0 | 1e-12 | yes |
| P1 | all | fp_mass(0) vs registered | [0.00999, 0.00999, 0.00999, 0.00999, 0.00999, 0.00999, 0.00999, 0.00999] | [0.00999, 0.00999, 0.00999, 0.00999, 0.00999, 0.00999, 0.00999, 0.00999] | 1e-12 | yes |
| P1 | all | A_F(0): max - min across structures | 2.08167e-17 | 0 | 1e-12 | yes |
| P1 | all | A_F(0) vs registered | [0.031607, 0.031607, 0.031607, 0.031607, 0.031607, 0.031607, 0.031607, 0.031607] | [0.031607, 0.031607, 0.031607, 0.031607, 0.031607, 0.031607, 0.031607, 0.031607] | 1e-12 | yes |
| P1 | all | alpha_F(0): max - min across structures | 2.34188e-16 | 0 | 1e-12 | yes |
| P1 | all | alpha_F(0) vs registered | [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01] | [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01] | 1e-12 | yes |
| P2 | RFP | C(0) | 0 | 0 | 1e-06 | yes |
| P2 | AND3 | C(0) | 0.0330196 | 0.03302 | 1e-06 | yes |
| P2 | AND-SYM | C(0) | 0.042384 | 0.042384 | 1e-06 | yes |
| P2 | AND-MID | C(0) | 0.0519096 | 0.05191 | 1e-06 | yes |
| P2 | AND-ASYM-A | C(0) | 0.0889334 | 0.088933 | 1e-06 | yes |
| P2 | AND-ASYM-B | C(0) | 0.0984004 | 0.0984 | 1e-06 | yes |
| P2 | OR | C(0) | 0.0992743 | 0.099274 | 1e-06 | yes |
| P2 | SINGLE | C(0) | 0.0993992 | 0.099399 | 1e-06 | yes |
| P2 | all | C(0) strictly increasing in registered order | True | True |  | yes |
| P3a | RFP | max |log J_G - log q0 - Lambda(state)| | 5.77316e-15 | 0 | 1e-07 | yes |
| P3b | RFP | outcome class at T_end | success | success |  | yes |
| P3b | RFP | FPR(T_end) vs S_inf | 0.01 | 0.01 | 1e-05 | yes |
| P4 | RFP | max |eta - eta(0)| (constant) | 5.36514e-29 | 0 | 1e-08 | yes |
| P4 | RFP | eta(0) (registered to 4 decimals) | [0, 0] |  |  | info |
| P5 | RFP | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | RFP | min logit step (non-decreasing) | 3.31006e-05 | >= -1e-10 | 1e-10 | yes |
| P3a | AND3 | max |log J_G - log q0 - Lambda(state)| | 4.68925e-09 | 0 | 1e-07 | yes |
| P3b | AND3 | outcome class at T_end | success | success |  | yes |
| P3b | AND3 | FPR(T_end) vs S_inf | 0.0272738 | 0.027274 | 1e-05 | yes |
| P4 | AND3 | min step of eta (>= -1e-8) | 1.16316e-07 | nondecreasing | 1e-08 | yes |
| P4 | AND3 | eta(0) (registered to 4 decimals) | [0.110351, 0.1104] |  |  | info |
| P5 | AND3 | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | AND3 | min logit step (non-decreasing) | -6.60583e-14 | >= -1e-10 | 1e-10 | yes |
| P6 | AND3 | relative drift of (1-s1)/(1-s2) | 1.26565e-13 | 0 | 1e-07 | yes |
| P6 | AND3 | relative drift of (1-s1)/(1-s3) | 1.66533e-13 | 0 | 1e-07 | yes |
| P6 | AND3 | max |s_i - s_1| (symmetry) | 1.16518e-13 | 0 | 1e-07 | yes |
| P3a | AND-SYM | max |log J_G - log q0 - Lambda(state)| | 8.81211e-10 | 0 | 1e-07 | yes |
| P3b | AND-SYM | outcome class at T_end | success | success |  | yes |
| P3b | AND-SYM | FPR(T_end) vs S_inf | 0.0622871 | 0.062287 | 1e-05 | yes |
| P4 | AND-SYM | min step of eta (>= -1e-8) | 3.06195e-07 | nondecreasing | 1e-08 | yes |
| P4 | AND-SYM | eta(0) (registered to 4 decimals) | [0.181818, 0.1818] |  |  | info |
| P5 | AND-SYM | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | AND-SYM | min logit step (non-decreasing) | -1.78302e-13 | >= -1e-10 | 1e-10 | yes |
| P6 | AND-SYM | relative drift of (1-s1)/(1-s2) | 3.18523e-13 | 0 | 1e-07 | yes |
| P6 | AND-SYM | max |s_i - s_1| (symmetry) | 2.39003e-13 | 0 | 1e-07 | yes |
| P3a | AND-MID | max |log J_G - log q0 - Lambda(state)| | 8.66528e-10 | 0 | 1e-07 | yes |
| P3b | AND-MID | outcome class at T_end | success | success |  | yes |
| P3b | AND-MID | FPR(T_end) vs S_inf | 0.123215 | 0.123215 | 1e-05 | yes |
| P4 | AND-MID | min step of eta (>= -1e-8) | 3.149e-07 | nondecreasing | 1e-08 | yes |
| P4 | AND-MID | eta(0) (registered to 4 decimals) | [0.272727, 0.2727] |  |  | info |
| P5 | AND-MID | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | AND-MID | min logit step (non-decreasing) | -1.60427e-14 | >= -1e-10 | 1e-10 | yes |
| P6 | AND-MID | relative drift of (1-s1)/(1-s2) | 3.1649e-11 | 0 | 1e-07 | yes |
| P3a | AND-ASYM-A | max |log J_G - log q0 - Lambda(state)| | 4.77527e-10 | 0 | 1e-07 | yes |
| P3b | AND-ASYM-A | outcome class at T_end | stall | stall |  | yes |
| P3b | AND-ASYM-A | J_G(T_end) vs q_inf | 0.230039 | 0.230039 | 0.001 | yes |
| P4 | AND-ASYM-A | min step of eta (>= -1e-8) | -2.08428e-07 | nondecreasing | 1e-08 | **NO** |
| P4 | AND-ASYM-A | eta(0) (registered to 4 decimals) | [0.800505, 0.8005] |  |  | info |
| P5 | AND-ASYM-A | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | AND-ASYM-A | min logit step (non-decreasing) | -2.44249e-14 | >= -1e-10 | 1e-10 | yes |
| P6 | AND-ASYM-A | relative drift of (1-s1)/(1-s2) | 6.41008e-08 | 0 | 1e-07 | yes |
| P3a | AND-ASYM-B | max |log J_G - log q0 - Lambda(state)| | 4.50229e-10 | 0 | 1e-07 | yes |
| P3b | AND-ASYM-B | outcome class at T_end | stall | stall |  | yes |
| P3b | AND-ASYM-B | J_G(T_end) vs q_inf | 0.107674 | 0.107674 | 0.001 | yes |
| P4 | AND-ASYM-B | min step of eta (>= -1e-8) | -4.38403e-06 | nondecreasing | 1e-08 | **NO** |
| P4 | AND-ASYM-B | eta(0) (registered to 4 decimals) | [0.980004, 0.98] |  |  | info |
| P5 | AND-ASYM-B | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | AND-ASYM-B | min logit step (non-decreasing) | -7.54952e-15 | >= -1e-10 | 1e-10 | yes |
| P6 | AND-ASYM-B | relative drift of (1-s1)/(1-s2) | 5.62684e-08 | 0 | 1e-07 | yes |
| P3a | OR | max |log J_G - log q0 - Lambda(state)| | 4.2843e-10 | 0 | 1e-07 | yes |
| P3b | OR | outcome class at T_end | stall | stall |  | yes |
| P3b | OR | J_G(T_end) vs q_inf | 0.198991 | 0.199499 | 0.001 | yes |
| P4 | OR | max step of eta (<= 1e-8) | -8.33767e-08 | nonincreasing | 1e-08 | yes |
| P4 | OR | eta(0) (registered to 4 decimals) | [0.997487, 0.9975] |  |  | info |
| P5 | OR | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | OR | min logit step (non-decreasing) | 3.31002e-05 | >= -1e-10 | 1e-10 | yes |
| P6 | OR | max |s_i - s_1| (symmetry) | 8.65974e-15 | 0 | 1e-07 | yes |
| P3a | SINGLE | max |log J_G - log q0 - Lambda(state)| | 7.4292e-10 | 0 | 1e-07 | yes |
| P3b | SINGLE | outcome class at T_end | stall | stall |  | yes |
| P3b | SINGLE | J_G(T_end) vs q_inf | 0.1 | 0.1 | 0.001 | yes |
| P4 | SINGLE | max |eta - eta(0)| (constant) | 7.56774e-06 | 0 | 1e-08 | **NO** |
| P4 | SINGLE | eta(0) (registered to 4 decimals) | [1, 1] |  |  | info |
| P5 | SINGLE | max (J_G - clean J_G) <= 1e-12 | 0 | <= 0 | 1e-12 | yes |
| P6 | SINGLE | min logit step (non-decreasing) | -3.24185e-14 | >= -1e-10 | 1e-10 | yes |
| P7 | all | Spearman(C(0), shortfall) | 0.913223 | 0.913 | 0.0005 | yes |
| P7 | all | discordant (C(0), shortfall) pairs | [[AND-ASYM-B, OR]] | [[AND-ASYM-B, OR]] |  | yes |
| O2/O4 | RFP | (J_G, FPR, clean gap) at t=10 | [0.952268, 0.01, 0.00434522] |  |  | info |
| O2/O4 | RFP | (J_G, FPR, clean gap) at t=25 | [1, 0.01, 3.94058e-09] |  |  | info |
| O2/O4 | RFP | (J_G, FPR, clean gap) at t=100 | [1, 0.01, 0] |  |  | info |
| O2/O4 | RFP | (J_G, FPR, clean gap) at t=500 | [1, 0.01, 0] |  |  | info |
| O2/O4 | AND3 | (J_G, FPR, clean gap) at t=10 | [0.948031, 0.0269918, 0.00858232] |  |  | info |
| O2/O4 | AND3 | (J_G, FPR, clean gap) at t=25 | [1, 0.0272738, 1.13637e-08] |  |  | info |
| O2/O4 | AND3 | (J_G, FPR, clean gap) at t=100 | [1, 0.0272738, -2.22045e-16] |  |  | info |
| O2/O4 | AND3 | (J_G, FPR, clean gap) at t=500 | [1, 0.0272738, 0] |  |  | info |
| O2/O4 | AND-SYM | (J_G, FPR, clean gap) at t=10 | [0.940487, 0.0607863, 0.0161259] |  |  | info |
| O2/O4 | AND-SYM | (J_G, FPR, clean gap) at t=25 | [1, 0.0622871, 3.53177e-08] |  |  | info |
| O2/O4 | AND-SYM | (J_G, FPR, clean gap) at t=100 | [1, 0.0622871, 0] |  |  | info |
| O2/O4 | AND-SYM | (J_G, FPR, clean gap) at t=500 | [1, 0.0622871, 0] |  |  | info |
| O2/O4 | AND-MID | (J_G, FPR, clean gap) at t=10 | [0.926589, 0.118282, 0.0300241] |  |  | info |
| O2/O4 | AND-MID | (J_G, FPR, clean gap) at t=25 | [1, 0.123215, 1.39098e-07] |  |  | info |
| O2/O4 | AND-MID | (J_G, FPR, clean gap) at t=100 | [1, 0.123215, 0] |  |  | info |
| O2/O4 | AND-MID | (J_G, FPR, clean gap) at t=500 | [1, 0.123215, 0] |  |  | info |
| O2/O4 | AND-ASYM-A | (J_G, FPR, clean gap) at t=10 | [0.222768, 0.968459, 0.733846] |  |  | info |
| O2/O4 | AND-ASYM-A | (J_G, FPR, clean gap) at t=25 | [0.230039, 1, 0.769961] |  |  | info |
| O2/O4 | AND-ASYM-A | (J_G, FPR, clean gap) at t=100 | [0.230039, 1, 0.769961] |  |  | info |
| O2/O4 | AND-ASYM-A | (J_G, FPR, clean gap) at t=500 | [0.230039, 1, 0.769961] |  |  | info |
| O2/O4 | AND-ASYM-B | (J_G, FPR, clean gap) at t=10 | [0.106751, 0.99143, 0.849862] |  |  | info |
| O2/O4 | AND-ASYM-B | (J_G, FPR, clean gap) at t=25 | [0.107674, 1, 0.892326] |  |  | info |
| O2/O4 | AND-ASYM-B | (J_G, FPR, clean gap) at t=100 | [0.107674, 1, 0.892326] |  |  | info |
| O2/O4 | AND-ASYM-B | (J_G, FPR, clean gap) at t=500 | [0.107674, 1, 0.892326] |  |  | info |
| O2/O4 | OR | (J_G, FPR, clean gap) at t=10 | [0.149184, 0.936391, 0.80743] |  |  | info |
| O2/O4 | OR | (J_G, FPR, clean gap) at t=25 | [0.186023, 0.995437, 0.813977] |  |  | info |
| O2/O4 | OR | (J_G, FPR, clean gap) at t=100 | [0.196788, 0.999815, 0.803212] |  |  | info |
| O2/O4 | OR | (J_G, FPR, clean gap) at t=500 | [0.198991, 0.999994, 0.801009] |  |  | info |
| O2/O4 | SINGLE | (J_G, FPR, clean gap) at t=10 | [0.0992423, 0.992423, 0.857371] |  |  | info |
| O2/O4 | SINGLE | (J_G, FPR, clean gap) at t=25 | [0.1, 1, 0.9] |  |  | info |
| O2/O4 | SINGLE | (J_G, FPR, clean gap) at t=100 | [0.1, 1, 0.9] |  |  | info |
| O2/O4 | SINGLE | (J_G, FPR, clean gap) at t=500 | [0.1, 1, 0.9] |  |  | info |

| run | optimizer | field evals | wall s | J_G(T) | FPR(T) |
| --- | --- | --- | --- | --- | --- |
| RFP | natural | 107 | 1.1 | 1 | 0.01 |
| AND3 | natural | 461 | 0.5 | 1 | 0.027273795 |
| AND-SYM | natural | 539 | 0.5 | 1 | 0.062287123 |
| AND-MID | natural | 572 | 0.6 | 1 | 0.12321473 |
| AND-ASYM-A | natural | 554 | 0.5 | 0.23003947 | 1 |
| AND-ASYM-B | natural | 524 | 0.5 | 0.10767378 | 1 |
| OR | natural | 764 | 0.7 | 0.1989905 | 0.99999351 |
| SINGLE | natural | 551 | 0.5 | 0.1 | 1 |
