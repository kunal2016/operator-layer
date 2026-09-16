# Operator Layer: a mathematical IR inside a neural representation

**Author:** Kunal Sinha · **Live demo:** https://arithmetic-operator-layer.netlify.app/

**Problem.** The usual way to ask this is "can we put mathematical structure into an
embedding so the vector itself performs mathematics?" — `embedding → answer`. A
compiler analogy suggests a stronger framing. A compiler never treats `9 + 9` as
raw bits from the start; it keeps progressively lower-level representations that
*preserve semantics*. At the intermediate-representation (IR) level, `ADD(9, 9)` is
not two bit-strings — it carries the meaning "integer addition of these two
values," and only later is lowered to machine ops and bits. Crucially, ADD there is
an *explicit operator*, not something inferred from the operands. That reframes the
goal: instead of hoping an embedding magically infers arithmetic, build a
**mathematical IR inside a neural representation** — a latent number representation
plus explicit operators — so computation becomes compositional in that latent
space: `M(9) ⊙ M(9) → M(18)`, then `M(18) ⊙ M(9) → M(162)`, and so on. This matters
because the failure mode of `embedding → answer` is precisely that a learned network
fits its training range of magnitudes and then fails to extrapolate; the structure
is present in the vector, but the model does not inherit it.

**Solution.** We store each number as a bank of complex phasors (a spectral/Fourier
code). Two exact homomorphisms fall out of the phase: shifting a number rotates
every phasor, so addition becomes an elementwise complex product of additive codes;
and since a logarithm turns products into sums, multiplication becomes the same
product in a logarithmic code. A fixed matched filter decodes the wave back to the
exact integer. This is the mathematical IR: deterministic, invertible,
zero-parameter, compositional. The **Operator Layer** is the pipeline the analogy
predicts — natural language → neural encoder → mathematical IR → operator → result.
A tiny gate learns only the easy, language-level step (mapping the phrasing
"plus" / "add" / "sum" → +); the exact bind applies it. Arithmetic is never learned;
it is an explicit operator, so it holds at any magnitude.

**Result.** The claim is tested *causally*, holding representation, inputs, output
space, and supervision fixed and varying only the operator (`experiments/end2end.py`,
`+`/`−`, **answer-only supervision** so op-selection must be learned, 3 seeds,
train `<120` / test `≥120`). The clean ablation keeps the same
inputs and the *same* differentiable decode over the *same* codebook and replaces only
the bind with a learned map (so unseen answers stay reachable — no label artifact):
the fixed operator extrapolates (**1.00**), the learned bind fits training and collapses
(**0.00**), and a NAC/NALU baseline learns the operation only approximately (exact-match
`0.05`, error growing with magnitude). HOL's only learned part is a tiny op-word gate,
so the substance is the fixed operator's exactness, not a hard learning problem. Separately, the codes decode
`+`, `−`, `×`, `÷` exactly across the tested ranges, and chained additions stay exact
to 32 terms.

**Why it matters.** In this controlled setting the cause of generalization is
*architectural* — the operator, not the features. Making the operation an explicit
primitive — a mathematical IR with transformations — is what converts "memorize a
range" into "compute for any magnitude," as a compiler preserves semantics while
changing representation.

**Honest scoping.** The encoding itself is not new — phasor codes bound by an
elementwise complex product are Holographic Reduced Representations / VSA (Plate 1995;
Kanerva 2009), log-domain multiplication is the classical log-number system, and
Fourier number features are known to emerge in trained networks (grokking; Nanda et
al. 2023). The contribution is the *framing* (a mathematical IR with explicit
operators), the *controlled result* (operator-as-architecture, not features, is what
extrapolates), and the *representation-theoretic "right factors" analysis* — not a new
primitive.

**Compiler correspondence.** The tightest analogy is to a compiler's
*front-end → IR → constant-folding* path. The number codes plus explicit operators
are the IR (`M(9) ⊙ M(9) → M(18)`, like `%c = add i32 9, 9`), and evaluating the bind
is precisely constant folding (`%c = 18`). The Operator Layer is therefore an
IR-level *evaluator*, not a code generator — the natural home for exact arithmetic in
representation space. In contrast, the naive `embedding → answer` model has no IR at
all: a black-box regressor from a vector to a number, which is why it cannot
extrapolate. Current scope is a single binary operation (a fixed, shallow AST);
nested/parenthesised expressions and rendering the result back to natural language
are straightforward next steps.
