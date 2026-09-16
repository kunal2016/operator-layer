"""
The spectral codes store arithmetic exactly: combining code(a) and code(b) with a
fixed bind and decoding gives a (op) b, with zero trainable parameters.
"""
import json, os, random
import numpy as np
from homomorphic import SpectralNumbers

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(OUT, exist_ok=True)
S = SpectralNumbers()
rng = random.Random(0)

# ---- exact +, -, x, / over ranges ------------------------------------------
def add_acc(hi, n=800):
    ok = 0
    for _ in range(n):
        a, b = rng.randrange(hi), rng.randrange(hi)
        ok += S.compute(a, b, '+', hi) == a + b
    return ok / n

def mul_acc(fmax, n=400):
    ok = 0
    for _ in range(n):
        a, b = rng.randint(1, fmax), rng.randint(1, fmax)
        ok += S.compute(a, b, '*', max(a, b)) == a * b
    return ok / n

add_pts = [{"range": h, "acc": add_acc(h, 400)} for h in [200, 500, 1000, 2000, 4000]]
mul_pts = [{"factor_max": f, "acc": mul_acc(f, 120)} for f in [50, 100, 200, 300]]
print("[exact] addition:")
for r in add_pts: print(f"    a,b<{r['range']:5d}  acc={r['acc']:.3f}")
print("[exact] multiplication:")
for r in mul_pts: print(f"    a,b<={r['factor_max']:5d} (prod<= {r['factor_max']**2:7d})  acc={r['acc']:.3f}")

# quotient + subtraction spot check
sub_ok = np.mean([S.compute(a, b, '-', 2000) == a - b
                  for a, b in [(rng.randrange(2000), rng.randrange(2000)) for _ in range(300)]])
div_ok = np.mean([S.compute(a * b, b, '/', a * b) == a
                  for a, b in [(rng.randint(1, 200), rng.randint(1, 60)) for _ in range(300)]])
print(f"[exact] subtraction acc={sub_ok:.3f}   division acc={div_ok:.3f}")

# ---- chained addition (no drift) -------------------------------------------
crng = random.Random(12345)
chain = []
for k in [2, 4, 8, 16, 32]:
    ok = 0
    for _ in range(300):
        xs = [crng.randrange(300) for _ in range(k)]
        code = S.add_code(xs[0])
        for x in xs[1:]:
            code = S._cmul(code, S.add_code(x), S.Ka)
        ok += S.dec_add(code, range(0, k * 300 + 1)) == sum(xs)
    chain.append({"terms": k, "acc": ok / 300})
    print(f"[exact] chain of {k:2d} adds: acc={ok/300:.3f}")

json.dump({"add": add_pts, "mul": mul_pts, "sub": float(sub_ok), "div": float(div_ok),
           "chain": chain, "add_dim": S.add_dim, "log_dim": S.log_dim},
          open(os.path.join(OUT, "exactness.json"), "w"), indent=2)
print("[exact] wrote results/exactness.json")
