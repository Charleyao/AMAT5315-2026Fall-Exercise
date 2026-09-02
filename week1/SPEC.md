`estimate_pi(n, seed)` throws n random darts uniformly in the unit square from
(0,0) to (1,1), counts how many of those points land within distance 1 of the
origin, and returns four times that fraction. The seed must make the result
repeatable, so that the same seed always reproduces the same output. As the
definition of correct: `abs(estimate_pi(1_000_000, seed=2026) - math.pi) < 1e-2`.
