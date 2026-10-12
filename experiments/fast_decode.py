"""
Experiment 6 -- a fast decoder: hierarchical phase unwrapping, O(K log range).

The repo decodes by a brute-force matched filter (O(range): scan every candidate),
which is the O(magnitude) wall in the cost analysis (REPORT 10). This is the better
decoder that analysis names: a coarse-to-fine phase-unwrapping readout that recovers
the integer ANALYTICALLY from the phases -- no candidate scan -- in O(K) with
K ~ log_radix(range) frequencies. It is exact up to the float64 precision ceiling and
runs in microseconds where brute force would take minutes to years.

Design. A geometric frequency ladder omega_j = 2*pi / lambda_j with wavelengths
lambda_j = radix^(L-1-j): the coarsest wavelength covers the whole range, and each
finer level multiplies resolution by `radix`. Each frequency's phase gives the residue
n mod lambda_j; decoding goes coarse -> fine, each level using the running estimate to
resolve the next level's wrap (round to the nearest multiple). Because the homomorphism
a(x) (x) a(y) = a(x+y) holds for ANY frequency set, this ladder supports the same exact
+/- bind -- only the READOUT changes, from an O(range) scan to an O(K) unwrap.

Scope: non-negative results (addition / magnitudes). Subtraction's sign is a one-bit
add-on. The float64 additive ceiling (~2^53) still applies -- a faster decoder lifts
COST, not precision.
"""
import time, math
import numpy as np


class LadderCode:
    def __init__(self, max_value, radix=64):
        self.radix = radix
        self.L = int(math.ceil(math.log(max(max_value, 2), radix))) + 2   # # frequencies
        self.lam = np.array([float(radix) ** (self.L - 1 - j) for j in range(self.L)])  # coarse->fine, finest=1
        self.w = 2 * np.pi / self.lam
        self.max_value = max_value

    def code(self, n):
        a = self.w * float(n)
        return np.concatenate([np.cos(a), np.sin(a)])

    def bind(self, A, B, sub=False):                 # exact +/- via elementwise complex product
        K = self.L
        ar, ai = A[:K], A[K:]; br, bi = B[:K], B[K:]
        if sub: bi = -bi
        return np.concatenate([ar * br - ai * bi, ar * bi + ai * br])

    def decode_hier(self, C):                        # O(K) coarse-to-fine phase unwrap
        K = self.L
        th = np.arctan2(C[K:], C[:K])
        th = np.where(th < 0, th + 2 * np.pi, th)    # [0, 2pi)
        res = th / (2 * np.pi) * self.lam            # residue_j = n mod lambda_j
        est = res[0]                                 # coarse: lambda_0 >= range -> n mod lambda_0 = n
        for j in range(1, K):
            k = round((est - res[j]) / self.lam[j])  # resolve this level's wrap from the running estimate
            est = k * self.lam[j] + res[j]
        return int(round(est))

    def decode_brute(self, C, lo, hi):               # the current O(range) matched filter, for comparison
        K = self.L
        th = np.arctan2(C[K:], C[:K])
        c = np.arange(lo, hi + 1)
        s = np.cos(th[None, :] - self.w[None, :] * c[:, None]).sum(1)
        return int(c[int(np.argmax(s))])


def exactness_and_speed():
    print("--- hierarchical decode: exactness + speed vs brute force ---")
    rng = np.random.default_rng(0)
    for R in [10**3, 10**6, 10**9, 10**12]:
        L = LadderCode(R)
        ns = rng.integers(0, R, 2000).tolist()
        t0 = time.time()
        ok = sum(L.decode_hier(L.code(n)) == n for n in ns)
        dt = (time.time() - t0) / len(ns)
        brute = 1.06e-6 * R                          # measured anchor: 1e6 window = 1.06 s, linear
        speedup = brute / max(dt, 1e-12)             # dimensionless ratio
        print(f"  range 10^{int(round(math.log10(R))):<2}  K={L.L} freqs  "
              f"exact={ok}/{len(ns)}  hier decode={dt*1e6:6.1f} us/call  "
              f"| brute force ~{_fmt(brute)}/call  -> ~{speedup:.1e}x faster")


def demo_big_addition():
    print("\n--- exact arithmetic on large numbers, decoded in microseconds ---")
    cases = [(123456789, 987654321, '+'), (10**11, 7 * 10**11, '+'),
             (54321, 12345, '+'), (999_999_999, 1, '+')]
    for a, b, op in cases:
        L = LadderCode(a + b + 1)
        C = L.bind(L.code(a), L.code(b))
        t0 = time.time(); dec = L.decode_hier(C); us = (time.time() - t0) * 1e6
        print(f"  {a} + {b} = {dec}   ({'exact' if dec == a + b else 'WRONG'}, "
              f"{L.L} freqs, {us:.1f} us)")


def precision_ceiling():
    print("\n--- float64 precision ceiling of the hierarchical decoder (additive) ---")
    rng = np.random.default_rng(1)
    for e in [12, 13, 14, 15, 16]:
        R = 10**e
        L = LadderCode(R)
        ns = [int(rng.integers(0, R)) for _ in range(500)]
        ok = sum(L.decode_hier(L.code(n)) == n for n in ns)
        print(f"  range 10^{e}: exact={ok}/500   ({'OK' if ok == 500 else 'precision-limited'})")


def _fmt(s):
    if s < 1e-3: return f"{s*1e6:.1f} us"
    if s < 1: return f"{s*1e3:.1f} ms"
    if s < 60: return f"{s:.1f} s"
    if s < 3600: return f"{s/60:.1f} min"
    if s < 86400*365: return f"{s/3600:.1f} h"
    yr = s / (3600*24*365)
    return f"{yr:.1f} yr" if yr < 1e4 else f"{yr:.1e} yr"


if __name__ == "__main__":
    exactness_and_speed()
    demo_big_addition()
    precision_ceiling()
