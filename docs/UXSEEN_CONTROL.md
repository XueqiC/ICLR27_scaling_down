# Unique-data × seen-tokens control (V31, distillation)

**Question.** Is the distillation capability response (δ_c = L_c(S_KD) − L_c(S0), endpoint = capability
loss) driven by **unique-data diversity** or by **cumulative seen-token exposure**? The earlier
fixed-epoch D-ladder confounded the two (larger D means both more unique data and more tokens).

**Design.** gemma3-1b student, teacher gpt-5.6-luna, recipe `full`, LoRA. Unique-example pool
U ∈ {75, 600} per domain, 3 seeds each = 6 continuous trajectories. Each trajectory saves an
evaluated checkpoint at matched **processed-token** milestones ≈ 500k and ≈ 1000k. U=75 reaches a
milestone through repetition (many epochs); U=600 reaches it with more unique data and little
repetition. Domain ratio and benchmarks fixed. The two checkpoints of one run are **paired**
observations, never independent seeds. Contrast = U75 − U600 at matched seen-tokens (positive =
the small, heavily repeated pool damages capability more). Paired t-interval, n=3.

**Result.**

| seen-tokens | cap | ΔL U75 | ΔL U600 | U75 − U600 (95% CI) |
|---|---|---:|---:|---|
| ~500k | math | +0.388 | +0.091 | **+0.296 [+0.236, +0.357]** |
| ~500k | code | +0.384 | +0.097 | **+0.287 [+0.249, +0.325]** |
| ~500k | qa | +2.361 | −1.238 | **+3.599 [+3.102, +4.096]** |
| ~1000k | math | +1.066 | +0.111 | **+0.955 [+0.887, +1.022]** |
| ~1000k | code | +0.797 | +0.137 | **+0.660 [+0.522, +0.799]** |
| ~1000k | qa | +5.180 | −0.619 | **+5.799 [+4.690, +6.909]** |

Every paired CI excludes zero.

**Reading (two axes, both real).**
1. **Seen-tokens (training volume) drives loss UP.** For a fixed pool, damage grows with training:
   U=75 code δ 0.38→0.80 (500k→1000k); U=600 code δ 0.10→0.14. This is the transfer response —
   training on teacher traces of the training benchmarks shifts the student on the held-out
   measurement benchmarks.
2. **At matched seen-tokens, unique-data diversity strongly MITIGATES damage.** U=600 (diverse,
   ~1 epoch) has far lower loss increase than U=75 (75 examples repeated ~15×) at the same token
   budget, and the gap grows with training.

So the response is explained by **neither token exposure alone nor unique-example count alone** —
both matter. The diversity effect is **not code-specific**: math and QA show it too, QA most
dramatically (repetition sends QA loss to +5.2, diversity IMPROVES it to −0.62). This reframes the
earlier "code has data-dependence" observation, which came from a fixed-epoch ladder in which
unique-data and seen-tokens moved together.

**Caveat.** At matched seen-tokens the U=75 arm is heavily overtrained (≈15 epochs on 75 examples),
so "low diversity" and "high repetition/overfitting" are the same axis in this design; the control
establishes that diversity/repetition matters beyond cumulative tokens, not the mechanism. Single
student size (gemma3-1b), single teacher/recipe. Regenerate: `python3 analysis/v31_uxseen_control.py`.
