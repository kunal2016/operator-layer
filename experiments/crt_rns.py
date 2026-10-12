"""
Experiment 7 -- the CRT / residue factorization, measured (head-to-head vs Fourier).

The appendix weighs the Residue Number System (CRT) as the *tightest* Kronecker
factorization: Z/M ~= (x)_i Z/p_i is a ring isomorphism, so BOTH addition and
multiplication are componentwise (mod p_i), exact, with no carry. This prototype
measures the two things it claims over the Fourier/phasor code -- and the price it pays.

What it lifts:
  * COST -- decode is CRT reconstruction over m moduli, O(m) big-int steps
    (m ~ log(range)), not the O(range) matched-filter scan.
  * The MULTIPLICATIVE PRECISION CEILING -- residues are exact small integers, so there
    is no float erosion. The Fourier log-code loses integer resolution above ~10^8 in
    float64 and cannot represent 10^30 at all; RNS computes 10^15 * 10^15 = 10^30
    EXACTLY, as long as M = prod(p_i) > the result.

What it costs (reported honestly, not hidden):
  * DISCRETE / non-smooth -- the residue code jumps with n; it is a poor continuous,
    differentiable neural representation (the whole point of the Fourier code).
  * NO MAGNITUDE ORDERING -- you cannot tell a < b from residues without reconstructing.
  * BOUNDED BY A HARD WRAP -- exact only on [0, M); past M it wraps modulo M (like
    fixed-width integer overflow), rather than degrading gracefully.
  * DIVISION / fractions are exact only when divisible and coprime to the moduli; making
    multiplication a *rotation* (to match the operator-layer bind) needs a second,
    per-prime discrete-log code (noted, not built here).

The additive operator primitive still holds per modulus: with a residue phasor
e^{2*pi*i*r/p}, addition is e^{2*pi*i*(r_a+r_b)/p} = (phasor_a)(x)(phasor_b) -- the same
complex-product bind as the Fourier code, now exact and unbounded in precision.
"""
import time, random, math


def make_moduli(target):
    """Distinct primes whose product M exceeds `target` (so [0,M) holds every result)."""
    pool = [999983, 999979, 999961, 999959, 999953, 999931, 999917, 999907,
            999883, 999863, 999853, 999809, 999773, 999769, 999763, 999749]
    ps, M = [], 1
    for p in pool:
        ps.append(p); M *= p
        if M > target:
            break
    return ps, M


class RNS:
    """Residue number system: n <-> (n mod p_i). Exact integer arithmetic, no float."""
    def __init__(self, target=10**40):
        self.p, self.M = make_moduli(target)
        self.m = len(self.p)

    def encode(self, n):
        return [n % p for p in self.p]                       # the residue code

    def add(self, A, B):
        return [(a + b) % p for a, b, p in zip(A, B, self.p)]   # componentwise, exact, no carry

    def mul(self, A, B):
        return [(a * b) % p for a, b, p in zip(A, B, self.p)]   # componentwise, exact, no carry

    def decode(self, A):
        """Garner's CRT reconstruction: O(m) big-int steps -> exact n in [0, M)."""
        x, P = A[0], self.p[0]
        for i in range(1, self.m):
            t = ((A[i] - x) * pow(P, -1, self.p[i])) % self.p[i]
            x += P * t
            P *= self.p[i]
        return x


def exactness():
    print("--- RNS exact +/x over a huge range (no float) ---")
    R = RNS(target=10**40)
    print(f"  {R.m} moduli near 10^6  ->  exact on [0, M),  M = prod(p_i) ~ 10^{len(str(R.M))-1}")
    rng = random.Random(0)
    ok_add = ok_mul = 0
    for _ in range(5000):
        a = rng.randrange(10**9); b = rng.randrange(10**9)       # products up to ~10^18
        ok_add += R.decode(R.add(R.encode(a), R.encode(b))) == a + b
        ok_mul += R.decode(R.mul(R.encode(a), R.encode(b))) == a * b
    print(f"  5000 random pairs (operands <10^9, products <10^18): add {ok_add}/5000, mul {ok_mul}/5000")

    print("\n--- the case the Fourier log-code gets WRONG ---")
    a = b = 10**15
    dec = R.decode(R.mul(R.encode(a), R.encode(b)))
    print(f"  10^15 * 10^15 = {dec}")
    print(f"  exact 10^30?  {dec == 10**30}   (Fourier log-code: not even representable in float64)")
    for a, b in [(48, 96), (123456789, 987654321), (999999999999, 999999999999)]:
        print(f"  {a} * {b} = {R.decode(R.mul(R.encode(a), R.encode(b)))}   "
              f"({'exact' if R.decode(R.mul(R.encode(a),R.encode(b)))==a*b else 'WRONG'})")


def head_to_head_multiplicative_ceiling():
    print("\n--- multiplicative exactness: Fourier log-code vs RNS ---")
    from homomorphic import SpectralNumbers
    import numpy as np
    S = SpectralNumbers()
    R = RNS(target=10**40)

    def fourier_status(P):                                    # how the log-code fares at P
        Pf = float(P)
        if Pf + 1.0 == Pf:                                    # neighbours collapse -> not representable
            return "no (unrepresentable)"
        c = S.log_code(Pf)
        cand = np.array([Pf - 1.0, Pf, Pf + 1.0])
        sc = np.array([c @ S.log_code(x) for x in cand])
        return "yes" if cand[int(sc.argmax())] == Pf else "no (resolution lost)"

    print(f"  {'result P':>10} | {'Fourier log-code':>22} | RNS")
    for P in [10**4, 10**6, 10**8, 10**10, 10**12, 10**20, 10**30]:
        rns_ok = R.decode(R.encode(P)) == P and P < R.M
        print(f"  {('10^'+str(len(str(P))-1)):>10} | {fourier_status(P):>22} | {'exact' if rns_ok else 'NO'}")
    print("  (Fourier: float64 log-phase step ~1/P -> resolution gone above ~10^8, then")
    print("   not even representable above ~2^53; RNS: exact small-integer residues, exact to M)")


def cost_and_tradeoffs():
    print("\n--- decode cost + the honest tradeoffs ---")
    R = RNS(target=10**40)
    rng = random.Random(1)
    ns = [rng.randrange(R.M) for _ in range(5000)]
    codes = [R.encode(n) for n in ns]
    t0 = time.time()
    for c in codes:
        R.decode(c)
    dt = (time.time() - t0) / len(ns)
    print(f"  CRT reconstruction: {R.m} moduli, {dt*1e6:.1f} us/decode  (O(m), not O(range); "
          f"brute-force scan of [0,M~10^{len(str(R.M))-1}) is physically impossible)")

    # additive operator primitive still holds, per modulus, via residue phasors
    import numpy as np
    p = R.p[0]; a, b = 7, 9
    ph = lambda r: np.exp(2j * np.pi * r / p)
    bound = ph(a % p) * ph(b % p)
    print(f"  additive bind per modulus is still a complex product: "
          f"phasor({a})*phasor({b}) == phasor({a}+{b})? "
          f"{np.isclose(bound, ph((a+b) % p))}")

    print("  tradeoffs (why this is NOT a drop-in replacement for the Fourier code):")
    print("   - discrete & non-smooth: residues jump with n -> poor differentiable neural code")
    print("   - no magnitude ordering: cannot compare a<b without reconstructing")
    print("   - hard modular wrap at M (overflow), not graceful precision decay")
    print("   - division/fractions exact only when divisible & coprime to moduli;")
    print("     making x a rotation (operator-layer bind) needs a per-prime discrete-log code")


if __name__ == "__main__":
    exactness()
    head_to_head_multiplicative_ceiling()
    cost_and_tradeoffs()
