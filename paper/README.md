# Paper

`kappa.tex` — *Does a Discovered Reinforcement Learning Rule Minimise Anything?
A gradient-field audit of DiscoRL.*

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

The last one is pinned to commit `9059a29f` of `google-deepmind/disco_rl` and is
re-run weekly in CI, so a claim about someone else's code cannot rot unnoticed.
