# Paper draft

[paper.md](paper.md) is a full-article write-up of E01-E03, meant to be read on GitHub. It is not a submission PDF and not a hardware paper.

**What it is.** A negative result: a Layden quench on a 2-local surrogate of a spike-and-slab posterior does not beat add-delete-swap. Scheduled depolarizing noise does not rescue it. Amplitude damping at γ = 0.01 already doubles TV versus the enumerated posterior.

**What it is not.** A claim that qe-MCMC is new to Bayesian statistics (it is not). A claim about IBM (we did not run). A claim that we invented quantum-inspired MCMC or coarse-graining.

Every number is copied from `results/**/summary.md`. Figures were remade from those tables:

```bash
uv run python docs/paper/make_figures.py
```

Voice notes for edits live in `.cursor/rules/40-paper-voice.mdc`.
