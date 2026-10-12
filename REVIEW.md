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

## Follow-up: learned parsing from raw text (mentor's next step)

The mentor's sign-off singled out the shared-decoder ablation and the honest scoping,
and named one extension: "learn operand parsing from raw text, keeping finite decode
range and cost explicit." This was then built (`experiments/parse_end2end.py`,
REPORT.md §8) and, as throughout this project, reported honestly rather than optimised
into a clean win:

- A character-level transformer reads the raw string and must produce the operand codes
  itself (parsing is no longer given). The operand code is assembled by the exact
  per-digit, per-place homomorphism — the Kronecker *value ⊗ position* bias — so reading
  generalises across magnitudes via a weight-shared digit reader; eval uses exact
  (argmax) reads so a correctly-read operand gives an exact answer.
- **Honest negative:** learned parsing does **not** bootstrap from the answer signal
  alone in this small setting (fixed operator, answer-only: 0.00 train / 0.00 extrap).
  This is the known frontier (length/magnitude generalization of learned arithmetic
  parsing) and is reported as unsolved, not papered over. Several stabilisation attempts
  (coarse-to-fine frequency warmup; straight-through reads) were tried and their failure
  on the answer-only arm is documented rather than hidden.
- **Causal claim carries over:** once parsing is supervised (a disclosed per-digit
  cross-entropy, with the answer loss then training only the combiner, so reader and
  combiner are learned the same way for both arms), the fixed operator extrapolates
  perfectly (1.00/1.00) while the identical model with a learned bind over the *same*
  parsed operands collapses out of distribution (1.00/0.00); a digit-sequence transformer
  fits in-range and collapses out (1.00/0.02). The learned bind shares the same decode
  over the same codebook, so its collapse is a real failure of the learned operator, not
  a label-coverage artifact.
- **Finite decode range/cost stated explicitly** (the mentor's specific ask): a matched
  filter over `[−200, 400]`, `NC = 601` candidates, `≈ 76,928` mul-adds/example,
  `O(magnitude)` — a bounded-range evaluator, not an unbounded arithmetic unit.

Net: the operator-versus-representation result now holds with *learned* parsing, and the
one remaining open piece (bootstrapping the parse from answer-only supervision) is named
precisely rather than obscured.

## Follow-up 2: spelled-number words and the multiply operator

Prompted by a direct question — does it handle "forty eight times ninety six"? — the
parse experiment was extended to English **number words** and the **multiplicative**
operator (`experiments/parse_mult.py`, REPORT.md §9). A word-level transformer reads
`["forty","eight","times","ninety","six"]`, parses each operand, and the fixed log-code
complex product multiplies; the input resolves to **4608**.

- The extrapolation axis is the **product** (train `a·b < 3000`, test `a·b ≥ 3000`, up to
  `9801`), with parsing in-distribution (all number words seen) — so this isolates the
  *operator*, complementing §8 (which isolated reading-generalization). Fixed multiply:
  **1.00 / 1.00**; learned bind (same codebook): 0.57 / 0.00; digit-sequence baseline:
  1.00 / 0.00. The operator is again the only thing that generalizes in magnitude.
- The finite decode range and its cost are reported and are deliberately larger than §8:
  window `[1, 9801]`, `NC = 9801`, `≈ 9.4×10⁵` mul-adds/example (~15× the additive band),
  `O(product)`. A double-precision matched filter is used because adjacent large products
  are log-close (with perfect reads the decode is exact over all 2,869 achievable
  products) — an honest, documented precision/cost knob rather than a hidden failure.
- Diagnostics were reported rather than hidden: the fixed arm first extrapolated at 0.80
  purely from parser read-coverage of rarely-seen large operands (decode with perfect
  reads was already 1.00), fixed by balanced parse supervision; and the float32-vs-float64
  decode distinction was traced and stated.
