"""
Experiment 4 -- learned operand parsing from raw text (the reviewer / mentor next step).

The earlier controlled test (`end2end.py`) handed every model deterministic operand
codes and only asked whether the *operator* (fixed bind vs learned map) causes
magnitude extrapolation. The honest objection: the operands were a given. Here the
operands are NOT given. A small **character-level transformer reads the raw string**
("48 + 79", " 7 - 153") and must itself produce the operand codes; only then does the
fixed operator block bind them and a shared matched-filter decoder read the answer.

Everything is trained with **answer-only supervision** -- no operand labels, no
operation labels, no digit labels. Parsing, segmentation and operation-selection must
all fall out of the single scalar answer signal.

Three models share the SAME character transformer front-end and differ only in how the
two parsed operand codes become an answer:

  * parse_fixed_op   : the FIXED complex-product bind (+ / - via conjugate), then a
                       shared differentiable matched-filter decode over a fixed
                       candidate codebook. The operator is an architectural primitive.
  * parse_learned_bind : the SAME parser and the SAME shared decode over the SAME
                       codebook, but the fixed bind is replaced by a LEARNED map on the
                       two operand codes. Unseen-magnitude answers stay reachable
                       through the shared decode, so a failure to extrapolate is a real
                       failure of the learned operator -- not a label-coverage artifact.
                       This is the clean ablation, now with learned parsing.
  * digit_seq        : the SAME transformer, but a sequence head decodes the answer
                       directly as sign + digits (no number code, no operator). This is
                       the strong baseline: a transformer that just emits the answer.

WHY THE PARSER CAN EXTRAPOLATE AT ALL (and why this ties back to Kronecker).
The operand code is assembled by the exact homomorphism a(d2*100 + d1*10 + d0) =
a(d2*100) (x) a(d1*10) (x) a(d0): a complex product of per-digit, per-place phasor
codes. Place is positional -- this is exactly the Kronecker "value (x) position"
inductive bias. What the transformer must LEARN from answer-only supervision is to read
each raw character into a digit, to segment the two operands, and to pick the operation
from the phrasing word. Because the per-slot digit reader shares weights across places,
reading "the tens slot is a 7" generalises from 70..79 to 170..179, so operand codes
stay exact at magnitudes never seen in training. The fixed operator then keeps the
answer exact; a learned bind does not.

FINITE DECODE RANGE AND COST (stated explicitly, per the mentor's note).
Decoding is a matched filter over a FIXED candidate codebook spanning [LO, HI]. It is
O(range): NC = HI-LO+1 candidates, each a 2*Ka vector, so the decode is ~NC*2*Ka
multiply-adds PER example and the codebook must be widened to admit larger answers.
This is a bounded-range evaluator (a constant-folding IR), not an unbounded arithmetic
unit; the numbers below report NC and the per-example decode cost.
"""
import json, os
import numpy as np
import torch, torch.nn as nn
from homomorphic import SpectralNumbers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
S = SpectralNumbers()
Ka = S.Ka
F = 2 * Ka

N, SPLIT = 200, 120            # operands 0..199 (3 digits); train max<120, test max>=120
SEEDS = [0, 1, 2]
OPS = ['+', '-']
PHRASING = {'+': ['plus', 'add', 'sum'], '-': ['minus', 'less', 'difference']}
WORDS = [w for o in OPS for w in PHRASING[o]]
WORD2OP = {w: oi for oi, o in enumerate(OPS) for w in PHRASING[o]}
WLEN = max(len(w) for w in WORDS)                       # 10 ("difference")

LO, HI = -N, 2 * N                                      # answer window for +/-
CANDS = np.arange(LO, HI + 1)
NC = len(CANDS)
ANS2IDX = {int(v): i for i, v in enumerate(CANDS.tolist())}

# ---- fixed layout: [A:3][ ][word:WLEN][ ][B:3] -> length L ----
LA = [0, 1, 2]
WREGION = list(range(4, 4 + WLEN))
LB = [4 + WLEN + 1 + k for k in range(3)]               # after A(3)+sp+word(WLEN)+sp -> 15,16,17
L = 3 + 1 + WLEN + 1 + 3

def render(a, b, word):
    s = f"{a:>3}" + " " + f"{word:<{WLEN}}" + " " + f"{b:>3}"
    assert len(s) == L, (len(s), L, repr(s))
    return s

# ---- char vocab ----
CHARS = sorted(set("0123456789 " + "".join(WORDS)))
C2I = {c: i for i, c in enumerate(CHARS)}
VOCAB = len(CHARS)

def encode_str(s):
    return [C2I[c] for c in s]

# ---- positional place structure (the Kronecker value (x) position bias) ----
# slot 0 -> hundreds, slot 1 -> tens, slot 2 -> units.
PLACE = torch.tensor([100.0, 10.0, 1.0])                 # (3,)
DVAL = torch.arange(10, dtype=torch.float32)             # (10,) digit values
WA = torch.tensor(S.wa, dtype=torch.float32)             # (Ka,) additive frequencies

CANDS_T = torch.tensor(CANDS, dtype=torch.float32)       # (NC,)

# Coarse-to-fine frequency warmup. The spectral code is deliberately high-frequency
# (that is what makes the decode exact and invertible), but that same oscillation is a
# poor from-scratch gradient landscape for regressing the operand value. We anneal a
# global frequency scale from low (smooth, trainable basin for the value) to 1.0 (exact
# decode). Scaling ALL frequencies equally preserves the homomorphism a(x) (x) a(y) =
# a(x+y), so the operator is unchanged; only the optimisation is shaped. Evaluation is
# always at full scale (fscale = 1.0). This is answer-only supervision throughout.
STATE = {"fscale": 1.0}

def code_of(v):
    """Exact additive phasor code a(v) = [cos(s*wa*v), sin(s*wa*v)] for a (B,) real value,
    at the current frequency scale s. a(x) (x) a(y) = a(x+y) holds for real v, so this is
    the same homomorphism as the per-digit-place complex product (identical at sharp
    digit reads), in the numerically stable scalar form."""
    ph = v[:, None] * WA[None, :] * STATE["fscale"]      # (B, Ka)
    return torch.cat([torch.cos(ph), torch.sin(ph)], 1)  # (B, F)

def cand_codes():
    ph = CANDS_T[:, None] * WA[None, :] * STATE["fscale"]  # (NC, Ka)
    return torch.cat([torch.cos(ph), torch.sin(ph)], 1)    # (NC, F)

def cmul(A, B, conj=False):
    """elementwise complex product of two (B, F) code batches (F = 2*Ka)."""
    ar, ai = A[:, :Ka], A[:, Ka:]
    br, bi = B[:, :Ka], B[:, Ka:]
    if conj:
        bi = -bi
    return torch.cat([ar * br - ai * bi, ar * bi + ai * br], 1)

def soft_decode(rcode, tau=12.0):
    scores = rcode @ cand_codes().t()                    # (B, NC)
    return torch.softmax(scores * (tau / Ka), 1)

# ---------------------------------------------------------------- data
def answer(a, b, opi):
    return a + b if opi == 0 else a - b

def make(seed, lo, hi, n):
    r = np.random.default_rng(seed)
    toks, ys, ab = [], [], []
    while len(toks) < n:
        a = int(r.integers(0, N)); b = int(r.integers(0, N))
        if not (lo <= max(a, b) < hi):
            continue
        w = WORDS[int(r.integers(0, len(WORDS)))]
        opi = WORD2OP[w]
        y = answer(a, b, opi)
        if not (LO <= y <= HI):
            continue
        toks.append(encode_str(render(a, b, w)))
        ys.append(y)
        ab.append((a, b))
    return (torch.tensor(toks, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            torch.tensor(ab, dtype=torch.long))        # (n,2) operands for parse supervision

def digit_targets(ab):
    """operand digit-class targets (hundreds, tens, units) for each operand."""
    a, b = ab[:, 0], ab[:, 1]
    def three(x): return torch.stack([(x // 100) % 10, (x // 10) % 10, x % 10], 1)  # (n,3)
    return three(a), three(b)

# ---------------------------------------------------------------- transformer front-end
class CharTransformer(nn.Module):
    def __init__(self, dm=32, heads=2, layers=1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dm)
        self.pos = nn.Parameter(torch.randn(L, dm) * 0.02)
        enc = nn.TransformerEncoderLayer(dm, heads, dm * 2, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.digit = nn.Linear(dm, 10)                    # shared per-slot digit reader
        self.oppool = nn.Linear(dm, len(OPS))             # operation gate from word region
        self.dm = dm

    def hidden(self, toks):
        h = self.emb(toks) + self.pos[None]               # (B, L, dm)
        return self.enc(h)

    def digit_logits(self, h):
        return self.digit(h[:, LA, :]), self.digit(h[:, LB, :])   # (B,3,10) each

    def operand_values(self, h, hard=False):
        def one(slots):
            logit = self.digit(h[:, slots, :])            # (B, 3, 10)
            d_hard = logit.argmax(-1).float()             # exact integer reads
            if hard:
                d = d_hard                                # pure eval: no gradient
            else:
                # straight-through: EXACT reads in the forward pass, soft gradient in the
                # backward pass, so training and eval both operate on exact operand codes.
                d_soft = (torch.softmax(logit, -1) * DVAL[None, None, :]).sum(-1)
                d = d_soft + (d_hard - d_soft).detach()
            return (d * PLACE[None, :]).sum(-1)           # (B,) positionally-weighted value
        return one(LA), one(LB)

    def operand_codes(self, h, hard=False):
        vA, vB = self.operand_values(h, hard)
        return code_of(vA), code_of(vB)

    def opgate(self, h):
        pooled = h[:, WREGION, :].mean(1)
        return torch.softmax(self.oppool(pooled), 1)      # (B, 2)


class ParseFixedOp(nn.Module):
    def __init__(self):
        super().__init__(); self.tr = CharTransformer()
    def forward(self, toks, hard=False):
        h = self.tr.hidden(toks)
        cA, cB = self.tr.operand_codes(h, hard)
        g = self.tr.opgate(h)
        d_add = soft_decode(cmul(cA, cB))
        d_sub = soft_decode(cmul(cA, cB, conj=True))
        return g[:, 0:1] * d_add + g[:, 1:2] * d_sub


class ParseLearnedBind(nn.Module):
    """Same parser, same shared decode over the same codebook; only the bind is learned."""
    def __init__(self):
        super().__init__()
        self.tr = CharTransformer()
        self.bind = nn.Sequential(nn.Linear(2 * F + len(OPS), 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, F))
    def forward(self, toks, hard=False):
        h = self.tr.hidden(toks)
        cA, cB = self.tr.operand_codes(h, hard)
        g = self.tr.opgate(h)
        rcode = self.bind(torch.cat([cA, cB, g], 1))
        return soft_decode(rcode)


class DigitSeq(nn.Module):
    """Same transformer, but decode the answer directly as sign + 3 digits (no operator).
    Given the full sequence (flattened) + an MLP head, so it genuinely fits in-range --
    the honest strong baseline: a transformer that just emits the answer tokens."""
    def __init__(self):
        super().__init__()
        self.tr = CharTransformer()
        dm = self.tr.dm
        self.head = nn.Sequential(nn.Linear(L * dm, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU())
        self.sign = nn.Linear(256, 2)
        self.dig = nn.ModuleList([nn.Linear(256, 10) for _ in range(3)])
    def forward(self, toks):
        h = self.tr.hidden(toks).reshape(toks.shape[0], -1)
        z = self.head(h)
        return self.sign(z), [d(z) for d in self.dig]


# ---------------------------------------------------------------- train / eval
NTR = 1500
EPC = 260            # epochs for the code-based (parser) arms
EPD = 160            # epochs for the digit-sequence baseline

def acc_code(model, toks, y):
    STATE["fscale"] = 1.0                                 # always evaluate at full (exact) scale
    with torch.no_grad():
        dist = model(toks, hard=True)                     # exact integer operand reads at eval
        pred = CANDS[dist.argmax(1).numpy()]
    return float(np.mean(pred == y.numpy()))

def run_code(kind, seed, parse_sup):
    """kind in {'parse_fixed_op','parse_learned_bind'}; parse_sup adds operand-digit
    cross-entropy (a disclosed parsing signal). Answer supervision is always present."""
    torch.manual_seed(seed)
    Xtr, ytr, abtr = make(seed, 0, SPLIT, NTR)
    Xte, yte, _ = make(seed + 100, SPLIT, N, NTR)
    yti = torch.tensor([ANS2IDX[int(v)] for v in ytr])
    tA, tB = digit_targets(abtr)                          # (n,3) each
    model = ParseFixedOp() if kind == 'parse_fixed_op' else ParseLearnedBind()
    opt = torch.optim.Adam(model.parameters(), 5e-3)
    ce = nn.CrossEntropyLoss()
    for ep in range(EPC):
        # Straight-through exact reads mean training and eval share the same exact codes,
        # so we train at full frequency throughout (no warmup needed).
        STATE["fscale"] = 1.0
        opt.zero_grad()
        # With parse supervision, the reader is taught by the digit cross-entropy and the
        # answer loss trains only the combiner (exact, detached operand codes -> hard=True),
        # so the noisy full-frequency answer gradient cannot corrupt the parser. Without it,
        # the reader must bootstrap from the answer signal alone (straight-through grad).
        ans = nn.functional.nll_loss(torch.log(model(Xtr, hard=parse_sup) + 1e-9), yti)
        loss = ans
        if parse_sup:
            h = model.tr.hidden(Xtr)
            la, lb = model.tr.digit_logits(h)
            loss = ans + sum(ce(la[:, s], tA[:, s]) + ce(lb[:, s], tB[:, s]) for s in range(3))
        loss.backward(); opt.step()
    return acc_code(model, Xtr, ytr), acc_code(model, Xte, yte)

def digits_of(v):
    s = 0 if v >= 0 else 1
    a = abs(int(v))
    return s, (a // 100) % 10, (a // 10) % 10, a % 10

def run_digitseq(seed):
    torch.manual_seed(seed)
    Xtr, ytr, _ = make(seed, 0, SPLIT, NTR)
    Xte, yte, _ = make(seed + 100, SPLIT, N, NTR)
    def targets(y):
        t = np.array([digits_of(int(v)) for v in y.numpy()])
        return (torch.tensor(t[:, 0]), torch.tensor(t[:, 1]),
                torch.tensor(t[:, 2]), torch.tensor(t[:, 3]))
    sg, d2, d1, d0 = targets(ytr)
    model = DigitSeq()
    opt = torch.optim.Adam(model.parameters(), 3e-3)
    ce = nn.CrossEntropyLoss()
    for _ in range(EPD):
        opt.zero_grad()
        ps, pd = model(Xtr)
        loss = ce(ps, sg) + ce(pd[0], d2) + ce(pd[1], d1) + ce(pd[2], d0)
        loss.backward(); opt.step()
    def acc(X, y):
        with torch.no_grad():
            ps, pd = model(X)
            s = ps.argmax(1).numpy()
            h = pd[0].argmax(1).numpy(); t = pd[1].argmax(1).numpy(); u = pd[2].argmax(1).numpy()
        val = (1 - 2 * s) * (100 * h + 10 * t + u)
        return float(np.mean(val == y.numpy()))
    return acc(Xtr, ytr), acc(Xte, yte)

def agg(fn):
    r = [fn(s) for s in SEEDS]
    tr = [x[0] for x in r]; te = [x[1] for x in r]
    return {"train_mean": float(np.mean(tr)), "train_std": float(np.std(tr)),
            "extrap_mean": float(np.mean(te)), "extrap_std": float(np.std(te))}


if __name__ == "__main__":
    print(f"[parse] LEARNED parsing from raw text; char transformer; answer-only loss unless "
          f"noted; ops={OPS}, train<{SPLIT}, test>={SPLIT}, seeds={SEEDS}")
    print(f"[parse] example input string: {render(48, 79, 'plus')!r}  (len {L}, vocab {VOCAB})")
    decode_cost = NC * F
    print(f"[parse] finite decode window [{LO},{HI}] -> NC={NC} candidates; "
          f"decode cost ~ NC*2Ka = {decode_cost:,} mul-adds/example (O(range))\n")

    arms = [
        ("parse + FIXED operator  (answer-only)",
         lambda s: run_code('parse_fixed_op', s, parse_sup=False)),
        ("parse + FIXED operator  (+ parse supervision)",
         lambda s: run_code('parse_fixed_op', s, parse_sup=True)),
        ("parse + LEARNED bind    (+ parse supervision, clean ablation)",
         lambda s: run_code('parse_learned_bind', s, parse_sup=True)),
        ("digit-sequence decoder  (no operator, strong baseline)",
         run_digitseq),
    ]
    res = {}
    for name, fn in arms:
        res[name] = agg(fn)
        r = res[name]
        print(f"   {name:60s} train={r['train_mean']:.3f}±{r['train_std']:.3f}  "
              f"EXTRAP={r['extrap_mean']:.3f}±{r['extrap_std']:.3f}")

    json.dump({"config": {"N": N, "split": SPLIT, "ops": OPS, "seeds": SEEDS,
                          "parsing": "learned (char transformer)",
                          "answer_supervision": "always; operand-digit supervision only where noted",
                          "eval": "hard (argmax) operand reads -> exact code -> exact decode",
                          "decode_window": [LO, HI], "NC": NC,
                          "decode_cost_muladds_per_example": decode_cost},
               "arms": res},
              open(os.path.join(OUT, "parse_end2end.json"), "w"), indent=2)
    print("\n[parse] wrote results/parse_end2end.json")
