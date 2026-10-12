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

**Learned parsing from raw text** (`experiments/parse_end2end.py`, see REPORT.md §8).
A character-level transformer reads the raw string (`' 48 plus        79'`) and must
itself produce the operand codes — parsing is no longer given. Two honest findings:
learned parsing does **not** bootstrap from the answer signal alone here (the open, hard
part), but **once parsing is supervised the architectural claim carries over in full** —
the fixed operator extrapolates to unseen magnitudes while a learned bind over the *same*
parsed operands collapses:

| char transformer reads raw text | train | unseen magnitude |
|---|---|---|
| parse + fixed operator (answer-only) | 0.00 | 0.00 |
| parse + fixed operator (+ parse supervision) | 1.00 | **1.00** |
| parse + learned bind (+ parse supervision, clean ablation) | 1.00 | **0.00** |
| digit-sequence decoder (no operator, strong baseline) | 1.00 | 0.02 |

Decode is a finite matched filter over `[−200, 400]` (`NC = 601` candidates ≈ `76,928`
mul-adds/example): a bounded-range evaluator, `O(magnitude)` by construction.

**Spelled numbers and multiplication** (`experiments/parse_mult.py`, REPORT.md §9).
A word-level transformer reads English number words and multiplies —
*forty eight times ninety six* → **4608**. With parsing in-distribution, the
extrapolation axis is the product (train `a·b < 3000`, test `a·b ≥ 3000`, up to `9801`):

| reads "forty eight times ninety six" | train | unseen large product |
|---|---|---|
| parse + fixed multiply (operator) | 1.00 | **1.00** |
| parse + learned bind (clean ablation) | 0.57 | **0.00** |
| digit-sequence decoder (no operator, baseline) | 1.00 | 0.00 |

Product decode window `[1, 9801]` (`NC = 9801` ≈ `9.4×10⁵` mul-adds/example, ~15× the
additive band) — the explicit `O(magnitude)` cost of the wider window.

**Honest scoping.** The *mechanism* is not new: phasor codes bound by an elementwise
complex product are Holographic Reduced Representations / VSA (Plate 1995; Kanerva
2009), log-domain multiplication is the classical log-number system, and Fourier
number features are known to emerge in trained networks (grokking; Nanda et al. 2023).
The contribution is the **framing** (a mathematical IR with explicit operators), the
**controlled result** (operator-as-architecture, not features, is what extrapolates),
and the **"right factors" analysis** (`FACTORS.md`) — not a new primitive.

---

## Cost and limitations

This is a **bounded-range evaluator** (a constant-folding IR), not a drop-in for a 32/64-bit
ALU. The key split: the **bind** (the operation) is cheap and magnitude-independent, while
the **decode** (reading the integer back) is a search whose cost grows with the range. See
REPORT.md §10 for the full treatment and measurements.

- **Readout is `O(range)`, not `O(1)`.** The bind — add/multiply as an elementwise complex
  product — is `O(K)` (K = 48–64 frequencies), exact, and never overflows (`a(x)⊙a(y) =
  a(x+y)` for any `x, y`; measured ~10 µs whether the operands are `5` or `10^15`). But
  decoding is a matched filter over the candidate window: `O(NC·K)`. Measured decode time is
  linear in the window — 10² → 0.9 ms, 10⁴ → 15 ms, 10⁶ → 1.06 s — so brute-forcing a
  5-digit product (≈10¹⁰ window) is impractical. Structured (CRT / hierarchical) decodes can
  reach `O(K·log range)` but reintroduce carries.
- **Finite unambiguous range per frequency set.** A fixed `K` resolves a bounded set of
  values; covering 32/64-bit ranges needs either `O(range)` frequencies or a CRT / digit⊗place
  structuring (see `FACTORS.md`).
- **Two precision ceilings, and they differ sharply.** Phases are stored in float. The
  additive step is constant, so `+`/`−` stay exact nearly to `2⁵³ ≈ 9×10¹⁵` (float64). The
  multiplicative step shrinks as `1/P`, so `×`/`÷` lose integer resolution far earlier —
  measured, the log band resolves `P` from `P±1` only to ~**10⁸** in float64 (~10⁴ in
  float32). `10¹⁵ × 10¹⁵ = 10³⁰` is doubly lost: `10³⁰` is not even a distinct float64 integer
  (`1e30 + 1 == 1e30`).
- **No GPU required.** The bind never needs one; the decode is embarrassingly parallel so a
  GPU raises throughput, but the real lever is the decode *algorithm*, not the hardware.

**Scaling to large magnitudes** (brute-force decode; bind is ~10 µs for every row). Time
extrapolates the measured CPU anchor (10⁶ window = 1.06 s, linear); GPU is an optimized-matmul
estimate (~10⁵× the single-core path; optimistic and memory-bound past ~10¹²). *Exactness* —
not time — is usually the binding constraint; an ✗ is wrong regardless of compute budget.

| result magnitude | CPU (1 core) | good GPU (est.) | exact `+`/`−` | exact `×`/`÷` |
|---|---|---|---|---|
| 10³ | ~1 ms | <1 µs | ✓ | ✓ |
| 10⁶ | ~1 s | ~10 µs | ✓ | ✓ |
| 10⁹ | ~18 min | ~10 ms | ✓ | ✗ |
| 10¹² | ~12 days | ~11 s | ✓ | ✗ |
| 10¹⁵ | ~34 years | ~3 hours | ✓ (near 2⁵³) | ✗ |
| 10³⁰ | ~3×10¹⁶ years | ~3×10¹¹ years | ✗ (not representable) | ✗ |

Brute-force decode is only sensible to ~10⁶–10⁹; past that both cost and precision rule it
out. A structured `O(K·log range)` decoder would be µs–ms at any row but does not lift the
exactness ceilings — those need higher-precision phases or a residue (CRT) representation.

**Fast decoder, measured** (`experiments/fast_decode.py`, REPORT.md §10). We built that
structured decoder for the additive band — a geometric frequency ladder decoded by
coarse-to-fine phase unwrapping, `O(K)` with `K ≈ log_radix(range)` frequencies, no candidate
scan. It is exact to 10¹⁵ and decodes in a flat ~10–20 µs regardless of magnitude:

| range | freqs `K` | hierarchical | brute force | speedup |
|---|---|---|---|---|
| 10⁶ | 6 | ~13 µs | ~1.1 s | ~8×10⁴× |
| 10⁹ | 7 | ~17 µs | ~18 min | ~6×10⁷× |
| 10¹² | 9 | ~18 µs | ~12 days | ~6×10¹⁰× |

`123456789 + 987654321` binds and decodes to `1111111110` in ~18 µs. It removes the cost
wall but not the precision wall (still exact only to ~2⁵³; breaks at 10¹⁶).
- **Scope of the learned results.** §8/§9 use `+ − ×`, operands to ~200 (additive) / 99
  (multiplicative), a single binary operation; learned operand parsing does not yet bootstrap
  from answer-only supervision (REPORT.md §8).

---

## 3. Run

```bash
pip install numpy torch
export PYTHONPATH=.
python experiments/exactness.py        # exact +,-,x,/ in code space; 32-term chains
python experiments/end2end.py          # CONTROLLED test: operator vs no-operator vs NAC
python experiments/parse_end2end.py    # LEARNED parsing from raw text (char transformer); REPORT.md 8
python experiments/parse_mult.py        # spelled-number words + MULTIPLY ("forty eight times ninety six"); REPORT.md 9
python experiments/fast_decode.py        # O(K log range) hierarchical decoder vs brute force; REPORT.md 10
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
experiments/parse_end2end.py  learned parsing from raw text (char transformer); REPORT.md 8
experiments/parse_mult.py     spelled-number words + multiply ("forty eight times ninety six"); REPORT.md 9
experiments/fast_decode.py    O(K log range) hierarchical decoder vs brute force; REPORT.md 10
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
