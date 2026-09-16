"""Fast, numpy-only checks that the spectral codes compute exact arithmetic."""
import random
from homomorphic import SpectralNumbers

S = SpectralNumbers()


def test_headline_examples():
    assert S.compute(9, 9, '+', 4000) == 18
    assert S.compute(9, 9, '*', 4000) == 81
    assert S.compute(123, 456, '+', 4000) == 579
    assert S.compute(48, 79, '*', 4000) == 3792
    assert S.compute(1517, 41, '/', 4000) == 37


def test_addition_exact():
    rng = random.Random(0)
    for _ in range(300):
        a, b = rng.randrange(4000), rng.randrange(4000)
        assert S.compute(a, b, '+', 4000) == a + b


def test_subtraction_exact():
    rng = random.Random(1)
    for _ in range(300):
        a, b = rng.randrange(2000), rng.randrange(2000)
        assert S.compute(a, b, '-', 2000) == a - b


def test_multiplication_exact():
    rng = random.Random(2)
    for _ in range(200):
        a, b = rng.randint(1, 300), rng.randint(1, 300)
        assert S.compute(a, b, '*', max(a, b)) == a * b


def test_chained_addition_no_drift():
    rng = random.Random(3)
    for _ in range(100):
        xs = [rng.randrange(300) for _ in range(16)]
        code = S.add_code(xs[0])
        for x in xs[1:]:
            code = S._cmul(code, S.add_code(x), S.Ka)
        assert S.dec_add(code, range(0, 16 * 300 + 1)) == sum(xs)
