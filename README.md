# Operator Layer — arithmetic a model can actually use (Kronecker V2)

**Author:** Kunal Sinha

**Live demo:** https://arithmetic-operator-layer.netlify.app/

*"What if embeddings can store mathematical structure — so that 9 combined with 9
is 18, and 81?"*

**Motivation — a mathematical IR.** A compiler never treats `9 + 9` as raw bits from
the start; it keeps representations that *preserve semantics*. At the
intermediate-representation (IR) level, `ADD(9, 9)` carries the *meaning* "integer
addition," and ADD is an *explicit operator*, not something inferred from the
operands. That reframes the goal from "make an embedding itself do math"
(`embedding → answer`) to building a **mathematical IR inside a neural
representation** — a latent number code plus explicit operators — so computation is
compositional: `M(9) ⊙ M(9) → M(18)`, then `M(18) ⊙ M(9) → M(162)`. See
[`WRITEUP.md`](WRITEUP.md) for the full framing.

Storing arithmetic in an embedding turns out to be the easy half: combine two
number-waves with a fixed product and the answer's wave falls out, exactly, with
zero parameters. The hard half — and the contribution of this repo — is getting a
**learned model to use that structure and generalize** to numbers outside its
training range. A **Homomorphic Operator Layer (HOL)** does it by making the exact
operation an *architectural primitive* rather than something the network must
learn.

Live demo (calculator + result): open `webapp/index.html`, or deploy the
`webapp/` folder to any static host (e.g. drag it onto Netlify Drop, or connect
this repo — `netlify.toml` sets the publish directory).

---

## 1. The number code (generic spectral / Fourier)

Each number is stored as a bank of complex phasors — a spectral code, not a
place-value or residue scheme:

```
additive code   a(n) = [ e^{i ω_k n}   ]_k        (for +, -)
log code        m(n) = [ e^{i ν_k ln n} ]_k        (for x, /)
```

Both are deterministic and invertible (decode by a matched filter over the range).
Two homomorphisms come straight from the phase, so a fixed elementwise complex
product (the **bind**) computes the operation exactly, for any inputs:

```
a(x + y) = a(x) ⊙ a(y)      (adding numbers  = phases add)
m(x · y) = m(x) ⊙ m(y)      (multiplying     = log-phases add)
```

Subtraction and division are the conjugate binds. See `homomorphic/spectral.py`.

**Exactness** (`experiments/exactness.py`, zero parameters):

| operation | result |
|---|---|
| `+`, `−` | exact, tested to `a,b < 4000` |
| `×`, `÷` | exact over the tested product range |
| chained `+` | exact to **32 terms**, no drift |

---

## 2. The contribution: the Homomorphic Operator Layer

Feeding these (correct) structured features to a normal network is **not enough**:
it memorizes the training range and fails on larger numbers. The fix is to put the
operation *in the architecture*. The layer splits the task along its natural seam:

1. **Encode** — numbers → spectral waves; query words → ordinary learned token
   embeddings.
2. **Gate** — a tiny learned gate maps the *phrasing* ("plus" / "add" / "sum" → `+`)
   to an operation. This is the only thing trained.
3. **Bind & decode** — the selected exact bind combines the number-waves; a fixed
   decoder reads the integer. No parameters, valid for any magnitude.

The network learns *what the query means*; it never has to learn arithmetic.

**Controlled result** (`experiments/end2end.py`): the claim is tested *causally* —
same representation, same inputs, same output space, **answer-only supervision** (no
op labels, so op-selection must be learned), varying only the operator. `+`/`−`,
numbers `0..199`, train `<120` / test `≥120`, 3 seeds:

| model (answer-only supervision) | train | **unseen magnitude** |
|---|---|---|
| **HOL** — fixed operator (complex-product bind) | 1.00 | **1.00** |
| **learned bind, shared decode** — clean ablation | 1.00 | **0.00** |
| learned answer-classifier — weaker ablation | 1.00 | 0.00 |
| **NAC / NALU** (Trask et al. 2018) on scalar values | 0.13 | 0.05 (±1: 0.10) |

The **clean ablation** keeps the same operand codes *and the same differentiable decode
over the same candidate codebook*, replacing only the fixed complex-product bind with a
learned map — so unseen-magnitude answers stay reachable and a failure is a real
failure of the *learned operator*, not a label artifact. It fits training and collapses
out of distribution; the fixed operator holds at 1.00. NAC genuinely learns the
operation (weights → `[1,±1]`) but only *approximately*, so its error grows with
magnitude. **In this controlled setting, the generalization comes from the operator,
not the features.** (Caveat: HOL's only learned part is the tiny op-word gate — trivial
to fit; the substance is the fixed operator's exactness. `operator_layer.py` covers
`+ − × ÷` as breadth; the causal claim rests on the clean ablation.)

**Honest scoping.** The *mechanism* is not new: phasor codes bound by an elementwise
complex product are Holographic Reduced Representations / VSA (Plate 1995; Kanerva
2009), log-domain multiplication is the classical log-number system, and Fourier
number features are known to emerge in trained networks (grokking; Nanda et al. 2023).
The contribution is the **framing** (a mathematical IR with explicit operators), the
**controlled result** (operator-as-architecture, not features, is what extrapolates),
and the **"right factors" analysis** (`FACTORS.md`) — not a new primitive.

---

## 3. Run

```bash
pip install numpy torch
export PYTHONPATH=.
python experiments/exactness.py        # exact +,-,x,/ in code space; 32-term chains
python experiments/end2end.py          # CONTROLLED test: operator vs no-operator vs NAC
python experiments/operator_layer.py   # breadth: + - x / via an op-word gate
python experiments/factorization.py    # operator admissibility by factorization (FACTORS.md)
```

```
9 + 9 -> 18     9 * 9 -> 81     123 + 456 -> 579     48 * 79 -> 3792   (0 params)
operator layer   train 0.99   UNSEEN magnitude 0.997
MLP on features  train ~0      UNSEEN magnitude ~0
```

## Layout

```
homomorphic/spectral.py       spectral number codes, the bind, the decoder
experiments/exactness.py      zero-parameter exactness of +,-,x,/ and chains
experiments/end2end.py        controlled test: operator vs no-operator vs NAC/NALU
experiments/operator_layer.py breadth over + - x / (op-word gate)
experiments/factorization.py  operator admissibility by factorization (see FACTORS.md)
tests/test_spectral.py        fast exactness tests (pytest)
results/                      JSON produced by the experiments
webapp/index.html             interactive companion (calculator + result)
webapp/results/               result JSON the webapp reports
report/                       written report (REPORT.md + PDF)
WRITEUP.md                    ~500-word summary (mathematical-IR framing)
FACTORS.md                    "what are the right mathematical factors?" (Experiment 2)
```

Everything is deterministic, self-contained, and runs on CPU in about a minute.
The number code appends alongside the existing word/byte dimensions — the "keep the
existing dims for language, add math in new ones" split — made exact, invertible,
and usable by a model that generalizes.
