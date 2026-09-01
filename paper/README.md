# Paper

`kappa.tex` — *Does a Discovered Reinforcement Learning Rule Minimise Anything?
A gradient-field audit of DiscoRL, and what it does in the MDP whose
soft-optimal policy is exactly a GFlowNet.*

Build:

```
latexmk -pdf kappa.tex
```

Every number in the paper comes from a script in `../research/`, and the scripts
recompute from scratch:

| Paper | Script |
|---|---|
| Table 1, estimator validation | `research/kappa.py` |
| Table 2, prediction probes | `research/calibrate.py` |
| Variance decomposition, entropy test | `research/thermo.py` |
| Sensitivity measurement | `research/disco_probe.py` |
| Claims about the released implementation | `research/verify_disco_source.py` |
| Reduction checks R1–R4, Remark 3 pricing | `research/tiapkin.py` |
| Reduction MDP, three arms, soft-family fit | `research/temperature.py` |
| Pipeline check | `research/temperature_pipeline_check.py` |
| Measured residual scale | `research/temperature_residual_scale.py` |
| Actor-critic fixed point, c-to-lambda map | `research/temperature_rule_fixedpoint.py` |
| Two-family fit | `research/temperature_family_compare.py` |
| Gap confound, withdrawn family verdict | `research/temperature_gap_confound.py` |
| Critic ablation | `research/temperature_critic_ablation.py` |
| Meta-objective comparison, detailed balance | `research/metagfn.py` |

`verify_disco_source.py` is pinned to commit `9059a29f` of
`google-deepmind/disco_rl` and is re-run weekly in CI, so a claim about someone
else's code cannot rot unnoticed.
