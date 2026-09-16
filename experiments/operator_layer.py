"""
The contribution: a Homomorphic Operator Layer (HOL) that makes arithmetic
GENERALISE to unseen magnitudes.

Natural-language arithmetic queries "<a> <word> <b>", <word> one of several
phrasings per operation. Numbers are spectral codes. The answer must be produced
for magnitudes far outside the training range (train on numbers < SPLIT, test on
pairs whose larger operand is >= SPLIT).

  * HOL         : a tiny gate learns phrasing -> operation; the exact homomorphic
                  bind + decode produces the number. The operation is an
                  ARCHITECTURAL PRIMITIVE, not something the network must learn.
  * MLP (feat)  : the SAME spectral features fed to an MLP that must regress the
                  answer directly -- the ablation showing features alone do not
                  transfer out of distribution.
"""
import json, os
import numpy as np
import torch, torch.nn as nn
from homomorphic import SpectralNumbers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
S = SpectralNumbers()
N, SPLIT = 120, 80
SEEDS = [0, 1, 2]
OPS = ['+', '-', '*']
PHRASING = {'+': ['plus', 'add', 'sum'], '-': ['minus', 'less'], '*': ['times', 'multiply', 'product']}
TOK = [p for o in OPS for p in PHRASING[o]]
TOK2I = {t: i for i, t in enumerate(TOK)}
TOK2OP = {p: oi for oi, o in enumerate(OPS) for p in PHRASING[o]}
def answer(a, b, opi): return [a + b, a - b, a * b][opi]

# precompute code tables (0..N-1)
ADD = np.stack([S.add_code(n) for n in range(N)]).astype(np.float32)
LOG = np.stack([S.log_code(max(n, 1)) for n in range(N)]).astype(np.float32)

def make(seed, lo, hi, n):
    rng = np.random.default_rng(seed)
    A = rng.integers(0, N, n); B = rng.integers(0, N, n)
    P = [TOK[rng.integers(0, len(TOK))] for _ in range(n)]
    m = (np.maximum(A, B) >= lo) & (np.maximum(A, B) < hi)
    return A[m], B[m], [P[i] for i in range(n) if m[i]]

# ---- HOL: gate (phrasing -> op) + exact bind/decode ------------------------
def run_hol(seed):
    torch.manual_seed(seed)
    trA, trB, trP = make(seed, 0, SPLIT, 24000)
    teA, teB, teP = make(seed, SPLIT, N, 24000)
    gate = nn.Embedding(len(TOK), len(OPS))
    opt = torch.optim.Adam(gate.parameters(), 1e-2); lf = nn.CrossEntropyLoss()
    Xtr = torch.tensor([TOK2I[p] for p in trP]); ytr = torch.tensor([TOK2OP[p] for p in trP])
    for _ in range(200):
        opt.zero_grad(); lf(gate(Xtr), ytr).backward(); opt.step()
    pred_op = gate(torch.arange(len(TOK))).argmax(1).tolist()   # phrasing -> op map
    def acc(A, B, P, k=700):
        ok = 0
        for a, b, p in list(zip(A, B, P))[:k]:
            opi = pred_op[TOK2I[p]]
            ok += S.compute(int(a), int(b), OPS[opi], N) == answer(int(a), int(b), TOK2OP[p])
        return ok / min(k, len(A))
    return acc(trA, trB, trP), acc(teA, teB, teP)

# ---- MLP on the same features (regression) ---------------------------------
class Reg(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.pe = nn.Embedding(len(TOK), 16)
        self.net = nn.Sequential(nn.Linear(din + 16, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
    def forward(self, x, pi): return self.net(torch.cat([x, self.pe(pi)], 1)).squeeze(1)

def feats(A, B):
    return np.concatenate([ADD[A], ADD[B], LOG[A], LOG[B]], axis=1)

SCALE = float(N * N)
def run_mlp(seed, epochs=140):
    torch.manual_seed(seed)
    trA, trB, trP = make(seed, 0, SPLIT, 9000)
    teA, teB, teP = make(seed, SPLIT, N, 3000)
    Xtr = torch.tensor(feats(trA, trB)); Xte = torch.tensor(feats(teA, teB))
    Ptr = torch.tensor([TOK2I[p] for p in trP]); Pte = torch.tensor([TOK2I[p] for p in teP])
    ytr = torch.tensor([answer(int(a), int(b), TOK2OP[p]) for a, b, p in zip(trA, trB, trP)], dtype=torch.float32) / SCALE
    yteA = np.array([answer(int(a), int(b), TOK2OP[p]) for a, b, p in zip(teA, teB, teP)])
    m = Reg(Xtr.shape[1]); opt = torch.optim.Adam(m.parameters(), 2e-3); lf = nn.MSELoss()
    for _ in range(epochs):
        m.train(); opt.zero_grad(); lf(m(Xtr, Ptr), ytr).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        tr = (torch.round(m(Xtr, Ptr) * SCALE) == torch.round(ytr * SCALE)).float().mean().item()
        pe = torch.round(m(Xte, Pte) * SCALE).numpy()
    te = float(np.mean(pe == yteA))
    return tr, te

def agg(fn):
    trs, tes = zip(*[fn(s) for s in SEEDS])
    return {"train_mean": float(np.mean(trs)), "train_std": float(np.std(trs)),
            "extrap_mean": float(np.mean(tes)), "extrap_std": float(np.std(tes))}

print(f"[HOL] numbers 0..{N-1}, train<{SPLIT}, extrapolate max(a,b)>={SPLIT}, seeds={SEEDS}")
res = {}
res["HOL (operator layer)"] = agg(run_hol)
print("   HOL (operator layer)      train=%.3f  EXTRAP=%.3f±%.3f" % (
    res["HOL (operator layer)"]["train_mean"], res["HOL (operator layer)"]["extrap_mean"],
    res["HOL (operator layer)"]["extrap_std"]))
res["MLP on spectral features"] = agg(run_mlp)
print("   MLP on spectral features  train=%.3f  EXTRAP=%.3f±%.3f" % (
    res["MLP on spectral features"]["train_mean"], res["MLP on spectral features"]["extrap_mean"],
    res["MLP on spectral features"]["extrap_std"]))

json.dump({"config": {"N": N, "split": SPLIT, "seeds": SEEDS, "ops": OPS}, "arms": res},
          open(os.path.join(OUT, "operator_layer.json"), "w"), indent=2)
print("[HOL] wrote results/operator_layer.json")
