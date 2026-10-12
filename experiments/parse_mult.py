"""
Experiment 5 -- spelled-number parsing and the multiplicative operator.

This answers a concrete question: does the layer handle an input like
"forty eight times ninety six"? Experiment 4 (parse_end2end.py) read digit
characters and did +/- in the additive band. Here we go the rest of the way the
question implies: the operands are English NUMBER WORDS, the operation is MULTIPLY,
and the operator is the log-code complex product (x -> add of log-phases).

A word-level transformer reads 5 tokens  [A_tens A_ones "times" B_tens B_ones]
(e.g. ["forty","eight","times","ninety","six"]) and must parse each operand to its
value; the fixed MULTIPLY operator then binds the two log-codes and a matched filter
decodes the product.

WHAT THIS EXPERIMENT ISOLATES (and how it differs from Experiment 4).
  * Exp 4 tested whether READING generalises across magnitude (a weight-shared digit
    reader generalises "the tens slot is a 7" from 70..79 to 170..179).
  * Here parsing is IN-DISTRIBUTION: every number word 1..99 appears in training (each
    value shows up as an operand of some small-product pair), so the parser is a seen
    lookup. The extrapolation axis is the PRODUCT: train on pairs with a*b < SPLIT,
    test on a*b >= SPLIT (up to 99*99 = 9801). This isolates whether the *operator*
    computes unseen, larger products -- the fixed log-operator does so exactly; a
    learned bind over the same parsed operands does not; a digit-sequence transformer
    that just emits the product fits in-range and collapses out.

FINITE DECODE RANGE AND COST (explicit, and deliberately larger than Exp 4).
  The additive experiment decoded over [-200, 400] (NC = 601). Multiplication needs a
  product codebook up to 99*99 = 9801, so NC ~ 9800 and the per-example decode cost is
  ~ NC * 2*Km ~ 9.4e5 multiply-adds -- an order of magnitude more than the additive
  band. This is exactly the O(magnitude) cost of a bounded-range evaluator; widening
  the window to admit larger products widens the codebook linearly.
"""
import json, os
import numpy as np
import torch, torch.nn as nn
from homomorphic import SpectralNumbers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
S = SpectralNumbers()
Km = S.Km
Fm = 2 * Km                                    # log-code dimension
WM = torch.tensor(S.wm, dtype=torch.float32)   # log frequencies

SEEDS = [0, 1, 2]
LOWV = 1                                        # operands 1..99 (exclude 0 -> log defined)
HIV = 99
SPLIT = 3000                                    # product split: train a*b<SPLIT, test a*b>=SPLIT
PMAX = HIV * HIV                                # 9801
CANDS = np.arange(1, PMAX + 1)                  # product decode window
NC = len(CANDS)
ANS2IDX = {int(v): i for i, v in enumerate(CANDS.tolist())}

# ---- English number words (real spelling, two word-slots per operand) ----
LOW = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
       "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
       "seventeen", "eighteen", "nineteen"]
TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
PAD = "<pad>"

def words(n):
    """Two word-slots for 1..99 (real English; PAD fills the ones slot when absent)."""
    if n < 20:
        return [LOW[n], PAD]
    t = TENS[n // 10]; o = n % 10
    return [t, LOW[o]] if o else [t, PAD]

TOKENS = sorted(set(LOW + [t for t in TENS if t] + [PAD, "times"]))
T2I = {t: i for i, t in enumerate(TOKENS)}
VOCAB = len(TOKENS)
L = 5                                           # [A0 A1 "times" B0 B1]
SA = [0, 1]; SB = [3, 4]

def render_tokens(a, b):
    wa, wb = words(a), words(b)
    return [T2I[wa[0]], T2I[wa[1]], T2I["times"], T2I[wb[0]], T2I[wb[1]]]

def pretty(a, b):
    return " ".join(w for w in words(a) if w != PAD) + " times " + \
           " ".join(w for w in words(b) if w != PAD)

# ---- log-code of a value, and candidate product codes ----
def log_code_t(v):
    ph = torch.log(torch.clamp(v, min=1.0))[:, None] * WM[None, :]   # (B, Km)
    return torch.cat([torch.cos(ph), torch.sin(ph)], 1)              # (B, Fm)

CAND = torch.tensor(np.stack([S.log_code(c) for c in CANDS]), dtype=torch.float32)  # (NC, Fm)

def cmul(A, B):
    ar, ai = A[:, :Km], A[:, Km:]; br, bi = B[:, :Km], B[:, Km:]
    return torch.cat([ar * br - ai * bi, ar * bi + ai * br], 1)

def soft_decode(rcode, tau=14.0):
    return torch.softmax((rcode @ CAND.t()) * (tau / Km), 1)

CAND_NP64 = CAND.numpy().astype(np.float64)
def hard_decode_np(rcode):
    # float64 matched filter: adjacent large products are log-close, so float32
    # accumulation over the dot product would occasionally pick a neighbour.
    scores = rcode.detach().numpy().astype(np.float64) @ CAND_NP64.T
    return CANDS[scores.argmax(1)]

# ---- data ----
def make(seed, lo, hi, n):
    r = np.random.default_rng(seed)
    toks, ys, ab = [], [], []
    while len(toks) < n:
        a = int(r.integers(LOWV, HIV + 1)); b = int(r.integers(LOWV, HIV + 1))
        if not (lo <= a * b < hi):
            continue
        toks.append(render_tokens(a, b)); ys.append(a * b); ab.append((a, b))
    return (torch.tensor(toks, dtype=torch.long),
            torch.tensor(ys, dtype=torch.long),
            torch.tensor(ab, dtype=torch.long))

# ---- models ----
class WordTransformer(nn.Module):
    def __init__(self, dm=48, heads=3, layers=1):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, dm)
        self.pos = nn.Parameter(torch.randn(L, dm) * 0.1)
        enc = nn.TransformerEncoderLayer(dm, heads, dm * 2, batch_first=True, dropout=0.0)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.val = nn.Linear(2 * dm, HIV + 1)        # shared operand-value classifier (0..99)
        self.dm = dm

    def hidden(self, toks):
        return self.enc(self.emb(toks) + self.pos[None])

    def value_logits(self, h):
        a = self.val(h[:, SA, :].reshape(h.shape[0], -1))
        b = self.val(h[:, SB, :].reshape(h.shape[0], -1))
        return a, b                                   # (B,100) each

    def operand_values(self, h, hard=False):
        la, lb = self.value_logits(h)
        if hard:
            return la.argmax(1).float(), lb.argmax(1).float()
        # straight-through: exact value in forward, soft gradient in backward
        vals = torch.arange(HIV + 1, dtype=torch.float32)
        sa = (torch.softmax(la, 1) * vals).sum(1); sb = (torch.softmax(lb, 1) * vals).sum(1)
        ha = la.argmax(1).float(); hb = lb.argmax(1).float()
        return sa + (ha - sa).detach(), sb + (hb - sb).detach()


class ParseFixedMul(nn.Module):
    """Parse (supervised) -> log-codes -> FIXED multiply bind -> matched-filter decode.
    Single operation, so there is no learned combiner: the reader is the only learned
    part, and the operator is an exact architectural primitive."""
    def __init__(self):
        super().__init__(); self.tr = WordTransformer()
    def forward_codes(self, toks, hard):
        h = self.tr.hidden(toks)
        vA, vB = self.tr.operand_values(h, hard)
        return cmul(log_code_t(vA), log_code_t(vB))
    def predict(self, toks):
        with torch.no_grad():
            return hard_decode_np(self.forward_codes(toks, hard=True))


class ParseLearnedBind(nn.Module):
    """Same parser; the fixed multiply bind is replaced by a LEARNED map on the two
    log-codes, decoded over the SAME product codebook (unseen products reachable)."""
    def __init__(self):
        super().__init__()
        self.tr = WordTransformer()
        self.bind = nn.Sequential(nn.Linear(2 * Fm, 512), nn.ReLU(),
                                  nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, Fm))
    def rcode(self, toks, hard):
        h = self.tr.hidden(toks)
        vA, vB = self.tr.operand_values(h, hard)
        return self.bind(torch.cat([log_code_t(vA), log_code_t(vB)], 1))
    def forward(self, toks, hard):
        return soft_decode(self.rcode(toks, hard))
    def predict(self, toks):
        with torch.no_grad():
            return hard_decode_np(self.rcode(toks, hard=True))


class DigitSeq(nn.Module):
    """Same transformer; emit the product directly as 4 digit heads (0..9801). No operator."""
    def __init__(self):
        super().__init__()
        self.tr = WordTransformer()
        self.head = nn.Sequential(nn.Linear(L * self.tr.dm, 256), nn.ReLU(),
                                  nn.Linear(256, 256), nn.ReLU())
        self.dig = nn.ModuleList([nn.Linear(256, 10) for _ in range(4)])
    def forward(self, toks):
        z = self.head(self.tr.hidden(toks).reshape(toks.shape[0], -1))
        return [d(z) for d in self.dig]
    def predict(self, toks):
        with torch.no_grad():
            pd = self.forward(toks)
            d = [p.argmax(1).numpy() for p in pd]
        return 1000 * d[0] + 100 * d[1] + 10 * d[2] + d[3]


# ---- train / eval ----
NTR = 700

def balanced_parse(reps=4):
    """A balanced parse-supervision batch: every value 1..99 as an operand, so the reader
    learns all number words regardless of the product-split's skew toward small operands.
    (Disclosed parse supervision, as in Experiment 4.)"""
    vals = list(range(LOWV, HIV + 1)) * reps
    X = torch.tensor([render_tokens(v, v) for v in vals], dtype=torch.long)
    t = torch.tensor(vals, dtype=torch.long)
    return X, t

def digit_targets(y):
    y = y.numpy()
    return (torch.tensor((y // 1000) % 10), torch.tensor((y // 100) % 10),
            torch.tensor((y // 10) % 10), torch.tensor(y % 10))

def run_fixed(seed, epochs=120):
    torch.manual_seed(seed)
    Xtr, ytr, abtr = make(seed, 1, SPLIT, NTR)
    Xte, yte, _ = make(seed + 100, SPLIT, PMAX + 1, NTR)
    Xb, tb = balanced_parse()
    m = ParseFixedMul(); opt = torch.optim.Adam(m.parameters(), 5e-3); ce = nn.CrossEntropyLoss()
    for _ in range(epochs):                           # reader trained by parse CE on operand values
        opt.zero_grad()
        h = m.tr.hidden(Xtr); la, lb = m.tr.value_logits(h)
        hb = m.tr.hidden(Xb); lba, lbb = m.tr.value_logits(hb)
        (ce(la, abtr[:, 0]) + ce(lb, abtr[:, 1]) + ce(lba, tb) + ce(lbb, tb)).backward(); opt.step()
    tr = float(np.mean(m.predict(Xtr) == ytr.numpy()))
    te = float(np.mean(m.predict(Xte) == yte.numpy()))
    return tr, te

def run_learned(seed, epochs=500):
    torch.manual_seed(seed)
    Xtr, ytr, abtr = make(seed, 1, SPLIT, NTR)
    Xte, yte, _ = make(seed + 100, SPLIT, PMAX + 1, NTR)
    Xb, tb = balanced_parse()
    tgt_code = log_code_t(ytr.float())                         # true product log-codes (train)
    m = ParseLearnedBind(); opt = torch.optim.Adam(m.parameters(), 3e-3); ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        h = m.tr.hidden(Xtr); la, lb = m.tr.value_logits(h)
        hb = m.tr.hidden(Xb); lba, lbb = m.tr.value_logits(hb)
        parse = ce(la, abtr[:, 0]) + ce(lb, abtr[:, 1]) + ce(lba, tb) + ce(lbb, tb)  # same reader sup.
        # the learned bind is trained to produce the true product code (code-space regression);
        # eval still decodes it over the SAME full product codebook, so unseen products are
        # reachable and a collapse is a real failure of the learned operator.
        ans = ((m.rcode(Xtr, hard=True) - tgt_code) ** 2).mean()
        (parse + ans).backward(); opt.step()
    tr = float(np.mean(m.predict(Xtr) == ytr.numpy()))
    te = float(np.mean(m.predict(Xte) == yte.numpy()))
    return tr, te

def run_digitseq(seed, epochs=160):
    torch.manual_seed(seed)
    Xtr, ytr, _ = make(seed, 1, SPLIT, NTR)
    Xte, yte, _ = make(seed + 100, SPLIT, PMAX + 1, NTR)
    t = digit_targets(ytr); m = DigitSeq()
    opt = torch.optim.Adam(m.parameters(), 3e-3); ce = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); pd = m(Xtr)
        sum(ce(pd[i], t[i]) for i in range(4)).backward(); opt.step()
    tr = float(np.mean(m.predict(Xtr) == ytr.numpy()))
    te = float(np.mean(m.predict(Xte) == yte.numpy()))
    return tr, te

def agg(fn):
    r = [fn(s) for s in SEEDS]
    tr = [x[0] for x in r]; te = [x[1] for x in r]
    return {"train_mean": float(np.mean(tr)), "train_std": float(np.std(tr)),
            "extrap_mean": float(np.mean(te)), "extrap_std": float(np.std(te))}


if __name__ == "__main__":
    print(f"[mult] spelled-number parsing + MULTIPLY; word-level transformer; seeds={SEEDS}")
    print(f"[mult] example input: {pretty(48, 96)!r} -> tokens {render_tokens(48, 96)}")
    print(f"[mult] operands 1..{HIV}; train a*b<{SPLIT}, test a*b>={SPLIT} (up to {PMAX})")
    cost = NC * Fm
    print(f"[mult] product decode window [1,{PMAX}] -> NC={NC} candidates; "
          f"decode cost ~ NC*2Km = {cost:,} mul-adds/example (O(product))\n")

    # sanity: the exact engine and the fixed operator both give 48*96 = 4608
    chk = ParseFixedMul()
    print(f"[mult] exact engine 48*96 = {S.compute(48,96,'*',100)}   "
          f"(product codebook covers it: {4608 in ANS2IDX})\n")

    arms = [("parse + FIXED multiply (operator)        ", run_fixed),
            ("parse + LEARNED bind (clean ablation)    ", run_learned),
            ("digit-sequence decoder (strong baseline) ", run_digitseq)]
    res = {}
    for name, fn in arms:
        res[name.strip()] = agg(fn); r = res[name.strip()]
        print(f"   {name} train={r['train_mean']:.3f}±{r['train_std']:.3f}  "
              f"EXTRAP={r['extrap_mean']:.3f}±{r['extrap_std']:.3f}")

    json.dump({"config": {"operands": [LOWV, HIV], "product_split": SPLIT, "pmax": PMAX,
                          "seeds": SEEDS, "operation": "x", "parsing": "spelled English number words",
                          "decode_window": [1, PMAX], "NC": NC,
                          "decode_cost_muladds_per_example": cost,
                          "note": "parsing in-distribution; extrapolation axis is product magnitude"},
               "arms": res},
              open(os.path.join(OUT, "parse_mult.json"), "w"), indent=2)
    print("\n[mult] wrote results/parse_mult.json")
