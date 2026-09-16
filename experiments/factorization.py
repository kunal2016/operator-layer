"""
Experiment 2 -- "What are the right mathematical factors?"

The Kronecker insight is that a *structured* factorization (byte x position) beats an
arbitrary learned token ID. We ask a precise question: which factorizations of a
NUMBER *admit an exact operator*, and of what form? This is about operator
ADMISSIBILITY, not a learned-model generalization comparison -- the controlled
learned test (representation held fixed, only the operator varied, answer-only
supervision) is `end2end.py`. Here we simply exhibit each factorization's natural
operator and confirm it is exact across magnitudes:

  * fourier      : additive character code  e^{i w_k n}. Addition DIAGONALIZES:
                   the operator is a fixed elementwise complex product (pointwise,
                   0 params). Exact -- but the code has a bounded unambiguous range.
  * digit_place  : Kronecker-style  digit (x) place  one-hot planes (base 10).
                   Addition does NOT diagonalize (carries couple places); the
                   natural operator is digit-wise add + carry -- exact and
                   UNBOUNDED, but not pointwise (sequential carry).
  * learned_id   : an arbitrary learned embedding table (nn.Embedding). No algebraic
                   operator exists; the model must memorize, so numbers outside the
                   training set have no usable representation.

The point: a factorization is useful for an operation iff it ADMITS a simple exact
operator (Fourier -> pointwise; digit-place -> carry; arbitrary ID -> none). Among
structured factors there is a range<->operation tradeoff (pointwise-but-bounded vs
carry-but-unbounded). The learned_id arm below only illustrates "no operator ->
memorization"; its 0% is a memorization failure (and note: with a fixed answer-class
head, test-range answers are also unseen labels). The rigorous learned comparison is
in `end2end.py`.
"""
import json, os, random
import numpy as np
import torch, torch.nn as nn
from homomorphic import SpectralNumbers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
S = SpectralNumbers()

# ----------------------------------------------------------------- fourier
def fourier_op_acc(mag, n=120):
    """pointwise complex bind + matched-filter decode, for a,b near `mag`."""
    ok = 0
    for _ in range(n):
        a = rng.randint(0, mag); b = rng.randint(0, mag)
        ok += S.compute(a, b, '+', mag) == a + b
    return ok / n

# ----------------------------------------------------------------- digit (x) place
P, BASE = 8, 10                      # up to 10^8 - 1
def digits(n):  return [(n // BASE**i) % BASE for i in range(P)]
def digit_rep(n):
    r = np.zeros((P, BASE));
    for i, d in enumerate(digits(n)): r[i, d] = 1.0
    return r
def digit_add(ra, rb):               # exact carry operator, in representation space
    da, db = ra.argmax(1), rb.argmax(1); carry = 0; out = []
    for i in range(P):
        s = int(da[i]) + int(db[i]) + carry; out.append(s % BASE); carry = s // BASE
    rr = np.zeros((P, BASE))
    for i, d in enumerate(out): rr[i, d] = 1.0
    return rr
def digit_decode(r): return sum(int(d) * BASE**i for i, d in enumerate(r.argmax(1)))
def digit_op_acc(mag, n=300):
    ok = 0
    for _ in range(n):
        a = rng.randint(0, mag); b = rng.randint(0, mag)
        if a + b >= BASE**P: continue
        ok += digit_decode(digit_add(digit_rep(a), digit_rep(b))) == a + b
    return ok / n

MAG_BANDS = [50, 500, 3000, 30000, 200000]
bands = []
for m in MAG_BANDS:
    nsamp = 120 if m < 50000 else 60
    fo = fourier_op_acc(m, n=nsamp)
    di = digit_op_acc(m, n=nsamp)
    bands.append({"magnitude": m, "n_samples": nsamp,
                  "fourier": round(fo, 3), "digit_place": round(di, 3)})
    print(f"[band] a,b~{m:>8d} (n={nsamp})  fourier(pointwise)={fo:.3f}   digit_place(carry)={di:.3f}")

# ----------------------------------------------------------------- learned ID
# arbitrary token embeddings: train to add within a vocab, test on unseen numbers.
N, SPLIT = 200, 130
def learned_id_extrap(seed=0, epochs=200):
    torch.manual_seed(seed); r = np.random.default_rng(seed)
    def pairs(lo, hi, k):
        A = r.integers(0, N, k); B = r.integers(0, N, k)
        m = (np.maximum(A, B) >= lo) & (np.maximum(A, B) < hi)
        return A[m], B[m]
    Atr, Btr = pairs(0, SPLIT, 12000); Ate, Bte = pairs(SPLIT, N, 12000)
    emb = nn.Embedding(N, 32)
    head = nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 2 * N))
    opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), 3e-3)
    lf = nn.CrossEntropyLoss()
    At, Bt = torch.tensor(Atr), torch.tensor(Btr); yt = torch.tensor(Atr + Btr)
    Ae, Be = torch.tensor(Ate), torch.tensor(Bte); ye = torch.tensor(Ate + Bte)
    def fwd(A, B): return head(torch.cat([emb(A), emb(B)], -1))
    for _ in range(epochs):
        opt.zero_grad(); lf(fwd(At, Bt), yt).backward(); opt.step()
    with torch.no_grad():
        tr = (fwd(At, Bt).argmax(1) == yt).float().mean().item()
        te = (fwd(Ae, Be).argmax(1) == ye).float().mean().item()
    return tr, te
lid = [learned_id_extrap(s) for s in [0, 1, 2]]
lid_tr = float(np.mean([x[0] for x in lid])); lid_te = float(np.mean([x[1] for x in lid]))
print(f"[learned_id] train={lid_tr:.3f}  UNSEEN(in-vocab)={lid_te:.3f}  "
      f"(no representation for numbers outside the vocabulary at all)")

result = {
    "config": {"P_places": P, "base": BASE, "vocab_N": N, "split": SPLIT},
    "operator_by_magnitude": bands,
    "learned_id": {"train": lid_tr, "unseen_in_vocab": lid_te},
    "summary": {
        "fourier": "pointwise complex-product operator; exact within a bounded range",
        "digit_place": "digit-wise carry operator; exact and unbounded, but not pointwise",
        "learned_id": "no algebraic operator; memorizes, cannot generalize to unseen numbers",
    },
}
json.dump(result, open(os.path.join(OUT, "factorization.json"), "w"), indent=2)
print("[factorization] wrote results/factorization.json")
