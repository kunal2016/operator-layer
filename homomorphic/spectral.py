"""
Spectral number codes and the Homomorphic Operator Layer (HOL).

Goal: embeddings that *store mathematical structure* — so that the vector of 9
combined with the vector of 9 becomes the vector of 18, and of 81.

We store a number in a generic **spectral phasor code** (a bank of complex
exponentials — a Fourier code, not a place-value or residue scheme):

    additive code   a(n) = [ e^{i ω_k n}   ]_k     (for +, -)
    logarithmic code m(n) = [ e^{i ν_k ln n} ]_k    (for x, /)

Both are deterministic and invertible (decode by a matched filter over the range).

The two useful homomorphisms fall straight out of the phase:

    a(x+y) = a(x) ⊙ a(y)     (adding numbers  = phases add  = complex product)
    m(x·y) = m(x) ⊙ m(y)     (multiplying     = log-phases add = complex product)

so a fixed *bind* (elementwise complex product) computes the operation in code
space, exactly, for ANY inputs.

The contribution of this repo is the **Homomorphic Operator Layer**: rather than
hope a network learns arithmetic from these features (it does not — it memorises
the training range and fails to extrapolate), we make the bind an *architectural
primitive*. A tiny gate learns only the easy part — mapping a natural-language
query to which operation — and the exact bind handles the numbers. The result is
a model whose arithmetic **generalises to magnitudes far outside training**.
"""
from __future__ import annotations
import numpy as np


class SpectralNumbers:
    def __init__(self, k_add=64, add_fmax=2.6, k_log=48, log_fmax=6.0):
        self.Ka = k_add
        self.Km = k_log
        self.wa = np.linspace(0.02, add_fmax, k_add)      # additive frequencies
        self.wm = np.linspace(0.25, log_fmax, k_log)      # log frequencies
        self.add_dim = 2 * k_add
        self.log_dim = 2 * k_log

    # ---- codes ------------------------------------------------------------
    def add_code(self, n):
        a = self.wa * float(n)
        return np.concatenate([np.cos(a), np.sin(a)])

    def log_code(self, n):
        a = self.wm * np.log(max(float(n), 1.0))
        return np.concatenate([np.cos(a), np.sin(a)])

    # ---- the bind (elementwise complex product) --------------------------
    @staticmethod
    def _cmul(A, B, K, conj=False):
        ar, ai = A[:K], A[K:]
        br, bi = B[:K], B[K:]
        if conj:
            bi = -bi
        return np.concatenate([ar * br - ai * bi, ar * bi + ai * br])

    def bind_add(self, a, b): return self._cmul(self.add_code(a), self.add_code(b), self.Ka)
    def bind_sub(self, a, b): return self._cmul(self.add_code(a), self.add_code(b), self.Ka, conj=True)
    def bind_mul(self, a, b): return self._cmul(self.log_code(a), self.log_code(b), self.Km)
    def bind_div(self, a, b): return self._cmul(self.log_code(a), self.log_code(b), self.Km, conj=True)

    # ---- decode (matched filter over the candidate range) ----------------
    @staticmethod
    def _mf(th, freqs, values, logs=False):
        """Memory-safe matched filter: argmax over `values` in chunks."""
        best_v, best_s = None, -1e18
        vals = np.asarray(values, dtype=np.float64)
        for i in range(0, len(vals), 40000):
            v = vals[i:i + 40000]
            arg = np.log(v) if logs else v
            s = np.cos(th[None, :] - freqs[None, :] * arg[:, None]).sum(1)
            j = int(np.argmax(s))
            if s[j] > best_s:
                best_s, best_v = s[j], int(v[j])
        return best_v

    def dec_add(self, code, cands):
        th = np.arctan2(code[self.Ka:], code[:self.Ka])
        return self._mf(th, self.wa, list(cands), logs=False)

    def dec_log(self, code, cands):
        th = np.arctan2(code[self.Km:], code[:self.Km])
        return self._mf(th, self.wm, list(cands), logs=True)

    # ---- convenience: apply an operation entirely in code space ----------
    def compute(self, a, b, op, hi):
        """op in {'+','-','*','/'}; hi bounds the decode search."""
        if op == '+':
            return self.dec_add(self.bind_add(a, b), range(0, 2 * hi + 1))
        if op == '-':
            code = self.bind_sub(a, b)
            th = np.arctan2(code[self.Ka:], code[:self.Ka])
            c = np.arange(-hi, hi + 1)
            s = np.cos(th[None, :] - self.wa[None, :] * c[:, None]).sum(1)
            return int(c[int(np.argmax(s))])
        if op == '*':
            return self.dec_log(self.bind_mul(a, b), range(1, min(hi * hi, 2_000_000) + 1))
        if op == '/':
            return self.dec_log(self.bind_div(a, b), range(1, hi + 1))
        raise ValueError(op)


if __name__ == "__main__":
    S = SpectralNumbers()
    for a, b, op in [(9, 9, '+'), (9, 9, '*'), (123, 456, '+'), (48, 79, '*'), (1517, 41, '/')]:
        print(f"  {a} {op} {b} = {S.compute(a, b, op, 4000)}")
