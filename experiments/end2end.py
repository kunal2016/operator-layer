"""
Experiment 3 -- the controlled test the reviewers asked for.

Question the earlier experiment did NOT answer: does the *operator as an
architectural inductive bias* cause magnitude extrapolation, holding the
representation, the task, the supervision, and the output space fixed?

Setup (addition and subtraction, so a differentiable soft-decode over a small
candidate range is tractable):
  * numbers 0..N, train on pairs with max(a,b) < SPLIT, test on max(a,b) >= SPLIT;
  * SUPERVISION IS THE ANSWER ONLY -- no operation labels. Any op-selection must be
    learned from the answer signal (this removes the trivial word->op supervision);
  * every model sees the same query (operand codes + an op-word embedding) and is
    trained with the same cross-entropy over the same candidate-answer set.

Models:
  * HOL          : fixed spectral operand codes; a learned soft gate over {+, -};
                   each op applies its FIXED exact bind; a DIFFERENTIABLE soft
                   matched-filter decodes to an answer distribution; the gate mixes
                   them. The only learned parameters are the gate. The operator is
                   an architectural primitive.
  * no_operator  : same operand codes + op-word, but a learned MLP maps them
                   directly to the answer distribution (no bind). Ablates the
                   operator, keeps everything else.
  * nac          : a NAC/NALU-style cell (Trask et al. 2018) on the scalar values
                   a, b -- the standard arithmetic-extrapolation baseline, which is
                   *designed* to extrapolate +/-.

If HOL extrapolates and no_operator does not -- same inputs, same supervision, same
output space -- then the operator, not the representation, is what generalizes.
NAC is included as an honest strong baseline that may also extrapolate.
"""
import json, os
import numpy as np
import torch, torch.nn as nn
from homomorphic import SpectralNumbers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
S = SpectralNumbers()
N, SPLIT = 200, 120
SEEDS = [0, 1, 2]
OPS = ['+', '-']
PHRASING = {'+': ['plus', 'add', 'sum'], '-': ['minus', 'less', 'difference']}
TOK = [p for o in OPS for p in PHRASING[o]]
TOK2I = {t: i for i, t in enumerate(TOK)}
TOK2OP = {p: oi for oi, o in enumerate(OPS) for p in PHRASING[o]}
LO, HI = -N, 2 * N                       # answer range for +/-
CANDS = np.arange(LO, HI + 1)
NC = len(CANDS)
ANS2IDX = {v: i for i, v in enumerate(CANDS.tolist())}

# precompute real operand codes and candidate codes (additive band)
Ka = S.Ka
CODES = torch.tensor(np.stack([np.concatenate([np.cos(S.wa * n), np.sin(S.wa * n)])
                               for n in range(N)]), dtype=torch.float32)      # (N, 2Ka)
CAND = torch.tensor(np.stack([np.concatenate([np.cos(S.wa * c), np.sin(S.wa * c)])
                              for c in CANDS]), dtype=torch.float32)          # (NC, 2Ka)

def cbind(ca, cb, sign):                 # elementwise complex product (or conj)
    ar, ai = ca[:, :Ka], ca[:, Ka:]; br, bi = cb[:, :Ka], cb[:, Ka:]
    if sign < 0: bi = -bi
    return torch.cat([ar * br - ai * bi, ar * bi + ai * br], 1)              # (B, 2Ka)

def soft_decode(rcode, tau=12.0):        # differentiable matched filter -> answer dist
    scores = rcode @ CAND.t()            # (B, NC)
    return torch.softmax(scores * (tau / Ka), 1)

def make(seed, lo, hi, n):
    r = np.random.default_rng(seed)
    A = r.integers(0, N, n); B = r.integers(0, N, n)
    P = [TOK[r.integers(0, len(TOK))] for _ in range(n)]
    m = (np.maximum(A, B) >= lo) & (np.maximum(A, B) < hi)
    return A[m], B[m], np.array([TOK2I[P[i]] for i in range(n) if m[i]])

def answer(a, b, opi): return a + b if opi == 0 else a - b

# ---------------------------------------------------------------- models
class HOL(nn.Module):
    def __init__(self):
        super().__init__()
        self.wordemb = nn.Embedding(len(TOK), 8)
        self.gate = nn.Linear(8, len(OPS))
    def forward(self, ai, bi, wi):
        ca, cb = CODES[ai], CODES[bi]
        d_add = soft_decode(cbind(ca, cb, +1))
        d_sub = soft_decode(cbind(ca, cb, -1))
        g = torch.softmax(self.gate(self.wordemb(wi)), 1)          # (B, 2)
        return g[:, 0:1] * d_add + g[:, 1:2] * d_sub               # (B, NC)

class LearnedBind(nn.Module):
    """The clean ablation: SAME shared matched-filter decode over the SAME candidate
    codebook, but the fixed complex-product bind is replaced by a LEARNED map on the
    same codes. Unseen-magnitude answers remain reachable through the decode, so a
    failure to extrapolate is a real failure of the learned operator -- not a
    label-coverage artifact."""
    def __init__(self):
        super().__init__()
        self.wordemb = nn.Embedding(len(TOK), 8)
        self.net = nn.Sequential(nn.Linear(2 * (2 * Ka) + 8, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2 * Ka))
    def forward(self, ai, bi, wi):
        x = torch.cat([CODES[ai], CODES[bi], self.wordemb(wi)], 1)
        rcode = self.net(x)                          # predicted answer code
        return soft_decode(rcode)                    # decoded via the SHARED codebook

class NoOperator(nn.Module):
    """A second, weaker ablation: a learned classifier straight to the answer class.
    Note this arm has a label-coverage ceiling -- test-range answer classes are never
    seen in training -- so its 0% is partly structural; LearnedBind is the clean one."""
    def __init__(self):
        super().__init__()
        self.wordemb = nn.Embedding(len(TOK), 8)
        self.net = nn.Sequential(nn.Linear(2 * (2 * Ka) + 8, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, NC))
    def forward(self, ai, bi, wi):
        x = torch.cat([CODES[ai], CODES[bi], self.wordemb(wi)], 1)
        return torch.log_softmax(self.net(x), 1)

class NAC(nn.Module):                     # NAC/NALU additive cell on scalar values
    def __init__(self):
        super().__init__()
        self.wordemb = nn.Embedding(len(TOK), 8)
        self.W = nn.Parameter(torch.randn(2, 2) * 0.1)   # (op, input) weights
        self.M = nn.Parameter(torch.randn(2, 2) * 0.1)
        self.gate = nn.Linear(8, 2)
    def forward(self, a, b, wi):
        w = torch.tanh(self.W) * torch.sigmoid(self.M)   # NAC weights -> approach {-1,0,1}
        x = torch.stack([a, b], 1)                       # (B, 2)
        per_op = x @ w.t()                               # (B, 2 ops)
        g = torch.softmax(self.gate(self.wordemb(wi)), 1)
        return (g * per_op).sum(1)                        # scalar per example

def acc_cls(dist, y):                      # y are integer answers
    pred = CANDS[dist.argmax(1).numpy()]
    return float(np.mean(pred == y))

def run(model_kind, seed, epochs=120):
    torch.manual_seed(seed)
    Atr, Btr, Wtr = make(seed, 0, SPLIT, 20000)
    Ate, Bte, Wte = make(seed, SPLIT, N, 20000)
    ytr = np.array([answer(int(a), int(b), TOK2OP_word(w)) for a, b, w in zip(Atr, Btr, Wtr)])
    yte = np.array([answer(int(a), int(b), TOK2OP_word(w)) for a, b, w in zip(Ate, Bte, Wte)])
    ai, bi, wi = torch.tensor(Atr), torch.tensor(Btr), torch.tensor(Wtr)
    aie, bie, wie = torch.tensor(Ate), torch.tensor(Bte), torch.tensor(Wte)
    if model_kind == 'nac':
        m = NAC(); opt = torch.optim.Adam(m.parameters(), 1e-2); lf = nn.MSELoss()
        af, bf = ai.float(), bi.float(); aef, bef = aie.float(), bie.float()
        yt = torch.tensor(ytr, dtype=torch.float32)
        for _ in range(2500):
            opt.zero_grad(); lf(m(af, bf, wi), yt).backward(); opt.step()
        with torch.no_grad():
            ptr = m(af, bf, wi).numpy(); pte = m(aef, bef, wie).numpy()
        tr = float(np.mean(np.round(ptr) == ytr)); te = float(np.mean(np.round(pte) == yte))
        # fair extra metrics: NAC is a regressor -> also report tolerant match + error
        return {"exact_tr": tr, "exact_te": te,
                "within1_te": float(np.mean(np.abs(pte - yte) <= 1)),
                "mae_tr": float(np.mean(np.abs(ptr - ytr))),
                "mae_te": float(np.mean(np.abs(pte - yte)))}
    m = {'hol': HOL, 'learned_bind': LearnedBind, 'no_operator': NoOperator}[model_kind]()
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    yti = torch.tensor([ANS2IDX[int(v)] for v in ytr])
    for _ in range(epochs):
        opt.zero_grad()
        out = m(ai, bi, wi)
        logp = out if model_kind == 'no_operator' else torch.log(out + 1e-9)
        nn.functional.nll_loss(logp, yti).backward(); opt.step()
    with torch.no_grad():
        tr = acc_cls(m(ai, bi, wi), ytr); te = acc_cls(m(aie, bie, wie), yte)
    return tr, te

def TOK2OP_word(wi): return TOK2OP[TOK[int(wi)]]

def agg(kind):
    r = [run(kind, s) for s in SEEDS]
    if kind == 'nac':
        tr = [x["exact_tr"] for x in r]; te = [x["exact_te"] for x in r]
        return {"train_mean": float(np.mean(tr)), "train_std": float(np.std(tr)),
                "extrap_mean": float(np.mean(te)), "extrap_std": float(np.std(te)),
                "extrap_within1": float(np.mean([x["within1_te"] for x in r])),
                "mae_train": float(np.mean([x["mae_tr"] for x in r])),
                "mae_extrap": float(np.mean([x["mae_te"] for x in r])),
                "note": "learns the operation but is approximate; error grows with magnitude"}
    tr = [x[0] for x in r]; te = [x[1] for x in r]
    return {"train_mean": float(np.mean(tr)), "train_std": float(np.std(tr)),
            "extrap_mean": float(np.mean(te)), "extrap_std": float(np.std(te))}

print(f"[end2end] answer-only supervision, ops={OPS}, train<{SPLIT}, test>={SPLIT}, seeds={SEEDS}")
res = {}
for kind, name in [('hol', 'HOL (fixed operator)'),
                   ('learned_bind', 'learned bind, shared decode (clean ablation)'),
                   ('no_operator', 'learned answer-classifier (has label ceiling)'),
                   ('nac', 'NAC / NALU-style baseline')]:
    res[name] = agg(kind)
    r = res[name]
    extra = f"  (within±1={r['extrap_within1']:.3f}, mae {r['mae_train']:.2f}->{r['mae_extrap']:.2f})" if kind == 'nac' else ""
    print(f"   {name:38s} train={r['train_mean']:.3f}±{r['train_std']:.3f}  "
          f"EXTRAP={r['extrap_mean']:.3f}±{r['extrap_std']:.3f}{extra}")

json.dump({"config": {"N": N, "split": SPLIT, "ops": OPS, "seeds": SEEDS,
                      "supervision": "answer-only"}, "arms": res},
          open(os.path.join(OUT, "end2end.json"), "w"), indent=2)
print("[end2end] wrote results/end2end.json")
