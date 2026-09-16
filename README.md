# Operator Layer — arithmetic a model can actually use (Kronecker V2)

**Author:** Kunal Sinha

**Live demo:** https://arithmetic-operator-layer.netlify.app/

*"What if embeddings can store mathematical structure — so that 9 combined with 9
is 18, and 81?"*

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

**Result** (`experiments/operator_layer.py`): templated natural-language queries
(`"<a> plus <b>"`, several phrasings per op), trained on numbers `< 80`, tested on
pairs whose larger operand is `≥ 80` — magnitudes never seen. Mean over 3 seeds:

| model | train (in-range) | **unseen magnitude** |
|---|---|---|
| **Homomorphic Operator Layer** | 0.99 | **0.997** |
| MLP on the same spectral features | ~0.01 | ~0.001 |

The operator layer extrapolates almost perfectly; a learned readout on identical
features does not. **The generalization lives in the operator, not the features.**

---

## 3. Run

```bash
pip install numpy torch
export PYTHONPATH=.
python experiments/exactness.py        # exact +,-,x,/ in code space; 32-term chains
python experiments/operator_layer.py   # operator layer vs a learned MLP baseline
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
experiments/operator_layer.py the HOL vs a learned baseline (magnitude extrapolation)
tests/test_spectral.py        fast exactness tests (pytest)
results/                      JSON produced by the experiments
webapp/index.html             interactive companion (calculator + result)
webapp/results/               result JSON the webapp reports
report/                       written report (REPORT.md + PDF)
```

Everything is deterministic, self-contained, and runs on CPU in about a minute.
The number code appends alongside the existing word/byte dimensions — the "keep the
existing dims for language, add math in new ones" split — made exact, invertible,
and usable by a model that generalizes.
