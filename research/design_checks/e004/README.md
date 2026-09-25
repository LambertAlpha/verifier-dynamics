# E004 design-phase checks (E000d) — NOT experiment code

These scratch scripts produced the design-phase facts F1–F8 in `research/06_e004_design.md`
(registry entry **E000d**, retroactive, not pre-registered). They are kept as run, so they are
excluded from the repository's lint and type checks and are not part of `vdyn`. The only edit
after running: the module-level code of `dc4_adam.py` and `dc6_mfadam.py` is wrapped in `main()`
so that later checks can import their helpers without re-running them.

Run from the repository root, e.g. `uv run python research/design_checks/e004/dc1_identity.py`.

| script | check | facts |
| --- | --- | --- |
| `utoy.py` | unified multi-prompt toy (exact enumeration, autodiff, geometry incl. `C_in/C_out`) | — |
| `dc1_identity.py` | observable identity under NG; where it breaks | F1, F2 (NG) |
| `dc2_dc3.py` | B vs exploit slope ratio; no decline under NG with FNR = 0 | F3, F4 |
| `dc4_adam.py` | infinite-batch Adam-like flow: identity, coupling, clean `t95` | F1 (Adam), F2 |
| `dc5_pair.py` | mechanism hard pair (B vs coupled AND2 exploit) | F7 |
| `dc6_mfadam.py` | mean-field Adam; exploit stall vs batch size | F5 |
| `dc7_inversion.py` | preference inversion under NG and mean-field Adam | F6 |
| `dc8_latent_pair.py` | latent-decline outcome hard pair (D vs B) under mean-field Adam | F8 |

Nothing here is evidence for E004; the numbers only shaped the design.
