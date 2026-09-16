# Adversarial review log

Before finalising, the submission was put through two independent, deliberately
skeptical reviews — one focused on experimental **rigor**, one on **completeness and
framing** — and then revised to address every substantive point. This log records the
concerns and how each was resolved, in the interest of honesty.

## Concerns raised (first pass)

1. **The headline experiment was close to tautological.** The original
   `operator_layer.py` hard-codes exact arithmetic and only learns a trivial
   word→operation gate, so "extrapolates to unseen magnitudes" mostly restated "the
   exact calculator still works."
2. **The baseline was a strawman.** The original MLP regressed a single scalar
   normalised by `N²` and demanded exact-integer match, so it could not even fit
   training — "calculator beats broken regressor" proves little. No **NALU/NAC**
   (the standard arithmetic-extrapolation baseline) was included.
3. **The factorization ablation conflated variables.** Fourier and digit-place both
   scored 100% because both were *given hand-coded operators*; only the learned-ID arm
   had to learn — so it tested "hand-coded vs learned," not "which factorization."
   Its 0% was also partly a label-coverage artifact.
4. **Overclaims.** "The generalization lives in the operator, not the representation"
   and "isolates the cause … it is architectural" were asserted without a controlled
   test; "MLP scores near zero" omitted that it scored near zero on *training* too.
5. **Novelty was not honestly scoped.** The mechanism (phasor codes + elementwise
   complex-product binding) is holographic/VSA binding (Plate 1995; Kanerva 2009); the
   log-domain trick is the classical log-number system; Fourier number features are
   known from "grokking." The abstract read as if the mechanism was new.

## Revisions made

- **Added a controlled experiment** (`experiments/end2end.py`): same representation,
  same inputs, same output space, **answer-only supervision** (no operation labels),
  varying only the operator. The **clean ablation** keeps the shared differentiable
  matched-filter decode over the same candidate codebook and replaces *only* the fixed
  complex-product bind with a learned map — so unseen-magnitude answers stay reachable
  and a failure is a genuine failure of the learned operator, not a label artifact.
  Result (3 seeds): fixed operator **1.00** train / **1.00** extrapolation; learned
  bind **1.00** / **0.00**; NAC/NALU learns the operation but only approximately
  (exact-match 0.05; mean absolute error grows 2.1 → 4.3 with magnitude).
- **Added the NAC/NALU baseline** (Trask et al. 2018) and reported it fairly
  (exact-match, within-±1, and error growth), rather than hiding behind exact-match.
- **Reframed the factorization ablation** as operator *admissibility* (which factors
  admit an exact operator, and of what form), reported n, noted the learned-ID
  label-coverage caveat, and deferred the learned comparison to `end2end.py`.
- **Softened the overclaims** to controlled-setting statements, and added two honest
  caveats: HOL's only learned parameters are the tiny op-word gate (trivial to fit),
  and the weaker answer-classifier ablation has a label ceiling and is kept only for
  completeness.
- **Scoped the novelty** in the abstract and a new *Related work* section: the
  mechanism is prior art; the contribution is the **framing** (a mathematical IR with
  explicit operators), the **controlled result**, and the **"right factors"** analysis.
- **Stated limitations and the next step** explicitly: the result is scoped to `+`/`−`
  with deterministic operand encoding; the path to a full paper is an end-to-end
  transformer with *learned* parsing, operator block ablated vs present.

## Final verdict (second pass)

Both reviewers confirmed the substantive concerns were resolved. Summary of their
sign-off: the controlled experiment converts the earlier near-tautology into a
defensible causal claim; the NAC handling is fair; novelty is honestly scoped; the
compiler/IR framing is correctly positioned as motivation rather than the
contribution; and the factorization analysis — the most original part — is properly
foregrounded. Remaining work is exactly the documented next step (learned-parsing
transformer). Assessment: a defensible course submission and a credible paper seed.
