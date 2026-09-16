# What are the right mathematical factors?

*Experiment 2 — a factorization ablation (`experiments/factorization.py`).*

The Kronecker-embedding insight is that a **structured factorization** —
`byte ⊗ position` — beats an arbitrary learned token ID. The natural question for
numbers is not "can we copy that?" (merely copying it is not enough) but:

> **What are the right factors?**

Different factorizations are right for different jobs:

- `byte ⊗ position` — right for **identity / spelling** (indexing a symbol).
- `digit ⊗ place` — right for **decimal notation / magnitude** (reading and writing),
  but addition does **not** factor through it cleanly, because carries *couple* the
  place-planes.
- **character / irrep basis** (Fourier `e^{iω n}` for `+`, log-Fourier for `×`) —
  right for the **operation**, because the operation becomes *pointwise*.

The unifying principle: **the right factors are the characters (irreducible
representations) of the algebraic structure whose operation you want to be exact and
cheap** — i.e. the basis in which that operation diagonalizes. Representation theory
answers the question.

## The ablation (operator *admissibility*)

We factor an integer three ways and exhibit each factorization's *natural* operator,
checking that it is exact across magnitude bands (n = 60–120 random pairs per band).
This measures which factorizations **admit an exact operator**, and of what form — it
is *not* a learned-model generalization comparison (that controlled test, holding the
representation fixed and varying only the operator, is `end2end.py`).

| factorization | natural operator | exact across magnitudes? | admits an exact operator? |
|---|---|---|---|
| **Fourier characters** `e^{iω n}` | pointwise complex product (0 params) | **100%** (bands to 2×10⁵) | ✓ pointwise — but a *bounded* range |
| **digit ⊗ place** (base 10) | digit-wise add + **carry** | **100%**, *unbounded* (any #places) | ✓ but *sequential*, not pointwise |
| **arbitrary learned ID** | none (a learned MLP must memorise) | train **100%**, unseen **0%** | ✗ none |

*(The learned-ID row only illustrates "no operator → memorisation"; its 0% is a
memorisation failure, and with a fixed answer-class head the test-range answers are
also unseen labels. The rigorous learned comparison is the `end2end.py` ablation.)*

## What it shows

1. **Usefulness tracks operator-admissibility, not representability.** All three
   factorizations can *represent* any number, but only the two that **admit an exact
   operator** compute it across magnitudes. The arbitrary learned ID has *no*
   operator, so a model can only memorise and then fails on unseen inputs. This is the
   Kronecker lesson, sharpened: structure helps *because it admits an operator*, not
   merely because it is structured. (That a *learned* model inherits this — the
   operator, not the features, generalizes — is shown rigorously by the controlled
   `end2end.py` ablation, not by this admissibility check.)

2. **Among structured factors there is a range ↔ operation tradeoff.** The Fourier
   (character) basis diagonalizes addition, so its operator is a single **pointwise**
   product — elegant and parameter-free — but the code has a finite unambiguous range
   and its decode is a search whose cost grows with magnitude. The `digit ⊗ place`
   basis does **not** diagonalize addition, so its operator must **propagate carries**
   (sequential, not pointwise) — but it is exact and **unbounded** at O(#places) cost.
   The "right factors" for full arithmetic may need to *combine* a notational factor
   (for range) with a character factor (for a clean operation).

3. **A ring has no single diagonalizing basis.** Addition and multiplication do not
   share an eigenbasis — which is exactly why the main method uses *two* bands
   (additive-Fourier and log-Fourier). Finding a joint representation, or a cheap
   change-of-basis between operation-bases, is the deeper open direction.

**Takeaway.** The right mathematical factors are **operation-relative**: pick the
basis whose characters diagonalize the operation you care about. `byte ⊗ position`
answers "what symbol, where"; the character basis answers "what operation, exactly" —
and it is the latter that makes arithmetic *generalize*.
