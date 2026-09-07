"""PB-native shell sampling with the Exact solver (M06).

The shell S_eps(x) = {y != x : |E_int(y) - E_int(x)| <= S*eps} is posed
directly as pseudo-Boolean constraints — binary y_i, linearized products
p_ij (p <= y_i, p <= y_j, p >= y_i + y_j - 1), and one two-sided linear
window constraint — with no CNF translation. M05 showed this removes the
CDCL-over-adder-circuits wall that dominated the CNF route on deep shells.

Sampling strategy: deep shells are small (M05: 10^3 -> 10^0 members below
T_c), so enumerate them completely (projected on y via invalidateLastSol)
and pick uniformly — exactly uniform, no hashing, no Hastings correction
beyond the |S(x)|/|S(y)| ratio the oracle already needs (and now has for
free, since |S(x)| is enumerated). Shells above the cap are large and
easy; they fall back to the CNF hashing sampler.
"""

from __future__ import annotations

import time

import numpy as np

from tunnelvision.dynamics.shellhash import ShellHashSampler


class ExactShellSampler:
    def __init__(self, J, h, eps: float = 2.0, scale: int = 1000,
                 cap: int = 2000, seed: int = 0, timeout_s: float = 300.0):
        self.base = ShellHashSampler(J, h, eps=eps, scale=scale, seed=seed)
        self.n = self.base.n
        self.cap = int(cap)
        self.timeout_s = float(timeout_s)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- model
    def _model(self, x: np.ndarray):
        import exact

        sh = self.base
        n = self.n
        s = exact.Exact()
        ynames = [f"y{i}" for i in range(n)]
        for nm in ynames:
            s.addVariable(nm, 0, 1)
        terms = []
        for i in range(n):
            a_i = (2 * int(sh.Jint[i].sum() - sh.Jint[i, i])
                   - 2 * int(sh.hint[i]))
            if a_i:
                terms.append((a_i, f"y{i}"))
        for i in range(n):
            for j in range(i + 1, n):
                if sh.Jint[i, j] == 0:
                    continue
                p = f"p{i}_{j}"
                s.addVariable(p, 0, 1)
                s.addConstraint([(1, p), (-1, f"y{i}")], False, 0, True, 0)
                s.addConstraint([(1, p), (-1, f"y{j}")], False, 0, True, 0)
                s.addConstraint([(1, p), (-1, f"y{i}"), (-1, f"y{j}")],
                                True, -1, False, 0)
                terms.append((-4 * int(sh.Jint[i, j]), p))
        e0 = sh.energy_int(x) - sh.C0
        w = round(sh.scale * sh.eps)
        s.addConstraint(terms, True, e0 - w, True, e0 + w)
        diff = [(1 if x[i] == 0 else -1, f"y{i}") for i in range(n)]
        s.addConstraint(diff, True, 1 - int(x.sum()), False, 0)
        return s, ynames

    # ------------------------------------------------------------ enumerate
    def enumerate(self, x: np.ndarray, cap: int | None = None,
                  timeout_s: float | None = None):
        """Up to cap+1 shell members projected on y. Returns (members,
        status) with status in {'complete', 'capped', 'timeout'}."""
        cap = self.cap if cap is None else cap
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        s, ynames = self._model(x)
        t0 = time.perf_counter()
        out = []
        while len(out) <= cap:
            left = timeout_s - (time.perf_counter() - t0)
            if left <= 0:
                return out, "timeout"
            r = s.runOnce(left)
            if r == "PAUSED":
                continue
            if r in ("UNSAT", "INCONSISTENT"):
                return out, "complete"        # check BEFORE reading: on
            if r == "SAT":                    # UNSAT hasSolution() is stale
                vals = s.getLastSolutionFor(ynames)
                out.append(np.array(vals, dtype=np.uint8))
                s.invalidateLastSol(ynames)
            else:
                return out, f"stop:{r}"
        return out, "capped"

    # ---------------------------------------------------------------- sample
    def sample(self, x: np.ndarray):
        """One shell proposal. Exactly uniform when the shell fits under the
        cap; hashing fallback otherwise. Returns (y | None, info)."""
        t0 = time.perf_counter()
        members, status = self.enumerate(x)
        if status == "complete":
            if not members:
                return None, dict(seconds=time.perf_counter() - t0,
                                  mode="empty", size=0, status=status)
            y = members[int(self.rng.integers(len(members)))]
            return y, dict(seconds=time.perf_counter() - t0, mode="enum",
                           size=len(members), status=status)
        if status == "capped":
            y, info = self.base.sample(x)
            return y, dict(seconds=time.perf_counter() - t0, mode="hash",
                           size=-1, status="large",
                           hash_info=info)
        return None, dict(seconds=time.perf_counter() - t0, mode="abort",
                          size=len(members), status=status)


__all__ = ["ExactShellSampler"]
