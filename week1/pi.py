import random


def estimate_pi(n, seed):
    rng = random.Random(seed)
    inside = 0
    for _ in range(n):
        x = rng.random()
        y = rng.random()
        if x * x + y * y <= 1:
            inside += 1
    return 4.0 * inside / n
