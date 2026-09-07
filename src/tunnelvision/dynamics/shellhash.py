"""SHELLHASH (M03) — near-uniform energy-shell sampling via SAT + XOR hashing.

The shell S_eps(x) = {y != x : |E(y) - E(x)| <= eps} of an Ising target is
encoded in CNF and sampled with random-parity hashing in the style of
XORSample / UniGen (Gomes-Sabharwal-Selman 2006; Ermon et al. UAI'12;
Chakraborty-Meel-Vardi). This is the de-novo classical answer to the open
question the paper leaves: uniform-on-shell at distance, without
enumeration, without a replica, without stored history.

Encoding
--------
Integerize the Hamiltonian: Jint = round(S*J), hint = round(S*h) with
scale S (default 1000). With s = 2y - 1,

    E_int(y) = C0 + sum_i a_i y_i + sum_{i<j} c_ij p_ij,
    a_i  = 2*sum_{j != i} Jint_ij - 2*hint_i,
    c_ij = -4*Jint_ij,
    C0   = -sum_{i<j} Jint_ij + sum_i hint_i,

where p_ij = y_i AND y_j are Tseitin product variables (3 clauses each).
The shell becomes two integer pseudo-Boolean bounds on a linear form
(pysat.pb.PBEnc, which accepts negative weights). All correctness claims
are made against E_int; the quantization gap to the real-J shell is
measured, not assumed away (see exact_shell_indices vs. real-J shells).

Sampling (UniGen-lite)
----------------------
Draw m random XOR constraints over the y-variables (each includes each
variable w.p. 1/2, random parity bit); enumerate the surviving cell up to
a cap via blocking clauses; return a uniform choice within the cell.
Adaptive m: too many solutions -> add XORs; empty -> drop one. XORs are
encoded solver-agnostically by chaining 3-variable parity chunks through
auxiliary variables (linear size), so any pysat solver works. This is the
practical variant of UniGen: guarantees are the literature's, uniformity
here is validated empirically at enumerable n (tests + M03a).
"""

from __future__ import annotations

import time

import numpy as np

from tunnelvision.bits import all_binary_states


class ShellHashSampler:
    def __init__(self, J, h, eps: float = 2.0, scale: int = 1000,
                 min_dh: int = 0, solver: str = "auto",
                 cell_cap: int = 16, seed: int = 0,
                 conf_budget: int | None = None):
        from pysat.formula import IDPool

        self.J = np.asarray(J, dtype=np.float64)
        self.h = np.asarray(h, dtype=np.float64)
        self.n = self.J.shape[0]
        self.eps = float(eps)
        self.scale = int(scale)
        self.min_dh = int(min_dh)
        if solver == "auto":
            try:
                import pycryptosat  # noqa: F401
                solver = "cms"
            except ImportError:
                solver = "glucose4"
        self.solver_name = solver
        self.cell_cap = int(cell_cap)
        self.conf_budget = conf_budget
        self.rng = np.random.default_rng(seed)
        self.Jint = np.rint(self.scale * self.J).astype(np.int64)
        self.hint = np.rint(self.scale * self.h).astype(np.int64)

        self.pool = IDPool()
        self.yvar = [self.pool.id(f"y{i}") for i in range(self.n)]
        self.pvar = {}
        self.base_clauses: list[list[int]] = []
        lits: list[int] = []
        wts: list[int] = []
        for i in range(self.n):
            a_i = 2 * int(self.Jint[i].sum() - self.Jint[i, i]) - 2 * int(self.hint[i])
            if a_i != 0:
                lits.append(self.yvar[i]); wts.append(a_i)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.Jint[i, j] == 0:
                    continue
                p = self.pool.id(f"p{i}_{j}")
                self.pvar[(i, j)] = p
                yi, yj = self.yvar[i], self.yvar[j]
                # p <-> yi & yj
                self.base_clauses += [[-p, yi], [-p, yj], [p, -yi, -yj]]
                lits.append(p); wts.append(-4 * int(self.Jint[i, j]))
        self.C0 = -int(np.triu(self.Jint, 1).sum()) + int(self.hint.sum())
        self.lin_lits, self.lin_wts = lits, wts

    # ------------------------------------------------------------ energies
    def energy_int(self, y: np.ndarray) -> int:
        s = 2 * y.astype(np.int64) - 1
        return int(-(np.triu(self.Jint, 1) * np.outer(s, s)).sum()
                   - (self.hint * s).sum())

    def exact_shell_indices(self, x: np.ndarray) -> np.ndarray:
        """Numpy-enumerated integer shell around x (n <= ~24). Excludes x."""
        S = all_binary_states(self.n)
        sp = 2 * S.astype(np.int64) - 1
        Ju = np.triu(self.Jint, 1)
        E = -np.einsum("ki,ij,kj->k", sp, Ju, sp) - sp @ self.hint
        e0 = self.energy_int(x)
        ok = np.abs(E - e0) <= round(self.scale * self.eps)
        if self.min_dh > 0:
            dh = (S != x).sum(1)
            ok &= dh >= self.min_dh
        pw = 1 << np.arange(self.n)
        ok[int(x @ pw)] = False
        return np.flatnonzero(ok)

    # ------------------------------------------------------------ formula
    def _pb_cnf(self, lo: int, hi: int):
        """Cached clauses for lo <= linear_form <= hi (the expensive part).

        The PB encodings dominate construction cost (~97% at n=8); they
        depend only on the integer bounds, so in a chain they amortize —
        exactly across the energy bins that make the proposal symmetric.
        """
        if not hasattr(self, "_pb_cache"):
            self._pb_cache = {}
        key = (lo, hi)
        if key not in self._pb_cache:
            top = self.pool.top + 1
            cl_hi, top = self._atmost_norm(self.lin_lits, self.lin_wts,
                                           hi, top)
            # atleast(w, b) == atmost(-w, -b), then normalized below.
            cl_lo, top = self._atmost_norm(self.lin_lits,
                                           [-wt for wt in self.lin_wts],
                                           -lo, top)
            self._pb_cache[key] = (cl_hi + cl_lo, top)
        return self._pb_cache[key]

    @staticmethod
    def _atmost_norm(lits, wts, bound, top):
        """PBEnc.atmost with manual negative-weight normalization.

        pysat rejects any raw negative bound before applying its own
        normalization, so do it here: w*l with w<0 becomes |w|*(not l)
        with the bound shifted by |w|. Trivially-true and trivially-false
        constraints short-circuit without an encoder call.
        """
        from pysat.pb import EncType, PBEnc

        nl, nw = [], []
        b = bound
        for l, w in zip(lits, wts):
            if w == 0:
                continue
            if w < 0:
                nl.append(-l); nw.append(-w); b += -w
            else:
                nl.append(l); nw.append(w)
        if b < 0:
            return [[]], top                      # unsatisfiable
        if b >= sum(nw):
            return [], top                        # trivially true
        # EncType.adder is forced: the default 'best' picks a BDD-style
        # encoding that is pseudo-polynomial in sum|w| (~5e5 at n=16,
        # scale=1000) and OOMs/stalls on some instances (n=16 seed=3 took
        # >4 min to ENCODE; adder: 0.01 s, 14k clauses — measured, M03).
        enc = PBEnc.atmost(lits=nl, weights=nw, bound=b, top_id=top,
                           encoding=EncType.adder)
        return enc.clauses, max(top, enc.nv) + 1

    def _shell_cnf(self, x: np.ndarray, mode: str = "centered"):
        """Clauses for the shell around x. mode='centered': the M01 oracle's
        |E - E(x)| <= eps. mode='bin': the fixed energy grid cell containing
        E(x) — exactly symmetric as a Metropolis proposal (y lands in the
        same bin, so the reverse proposal is the same uniform cell) and
        maximally cache-friendly."""
        from pysat.card import CardEnc

        e0 = self.energy_int(x) - self.C0          # bound on the linear form
        w = round(self.scale * self.eps)
        if mode == "bin":
            b = e0 // (2 * w)                       # bin width = 2*eps
            lo, hi = b * 2 * w, (b + 1) * 2 * w - 1
        else:
            lo, hi = e0 - w, e0 + w
        pb_clauses, top = self._pb_cnf(lo, hi)
        clauses = list(self.base_clauses) + list(pb_clauses)
        diff_lits = [self.yvar[i] if x[i] == 0 else -self.yvar[i]
                     for i in range(self.n)]
        if self.min_dh > 1:
            card = CardEnc.atleast(lits=diff_lits, bound=self.min_dh,
                                   top_id=top)
            top = max(top, card.nv) + 1
            clauses += card.clauses
        else:
            clauses.append(diff_lits)              # y != x
        return clauses, top

    def _xor_clauses(self, var_subset: list[int], parity: int, top: int):
        """CNF for XOR(subset) = parity via 3-chunk chaining (any solver)."""
        clauses: list[list[int]] = []
        cur = list(var_subset)
        while len(cur) > 3:
            aux = top; top += 1
            a, b = cur[0], cur[1]
            # aux <-> a XOR b
            clauses += [[-aux, a, b], [-aux, -a, -b], [aux, -a, b], [aux, a, -b]]
            cur = [aux] + cur[2:]
        k = len(cur)
        if k == 0:
            if parity == 1:
                clauses.append([])                 # unsatisfiable
            return clauses, top
        for bits in range(1 << k):
            ones = bin(bits).count("1")
            if ones % 2 != parity:                 # forbid wrong-parity rows
                clauses.append([cur[i] if not (bits >> i) & 1 else -cur[i]
                                for i in range(k)])
        return clauses, top

    # ------------------------------------------------------------ sampling
    def _enumerate_cell(self, clauses, cap: int, xors=None):
        """Solutions up to cap+1 via blocking clauses. Returns (sols,
        aborted): aborted=True means the conflict budget ran out mid-cell,
        so the cell content (and any sample drawn from it) is untrusted.

        xors: list of (var_subset, parity). With solver='cms'
        (CryptoMiniSat via pycryptosat) they are added NATIVELY —
        Gaussian elimination handles the hash constraints that make
        CDCL-on-CNF-parities blow up (the reason UniGen is built on
        CryptoMiniSat; measured: glucose4 needs >5 min/sample at n=16 on
        large shells, M03c). Other solvers get the chunked CNF encoding.
        """
        if self.solver_name == "cms":
            return self._enumerate_cell_cms(clauses, cap, xors or [])
        if xors:
            top = self._xor_top
            clauses = list(clauses)
            for subset, parity in xors:
                xc, top = self._xor_clauses(subset, parity, top)
                clauses += xc
        from pysat.solvers import Solver

        sols = []
        aborted = False
        with Solver(name=self.solver_name, bootstrap_with=clauses,
                    use_timer=False) as sv:
            while len(sols) <= cap:
                if self.conf_budget is not None:
                    sv.conf_budget(self.conf_budget)
                    res = sv.solve_limited(expect_interrupt=False)
                    if res is None:
                        aborted = True
                        break
                else:
                    res = sv.solve()
                if not res:
                    break
                model = sv.get_model()
                pos = {abs(l) for l in model if l > 0}
                y = np.array([1 if v in pos else 0 for v in self.yvar],
                             dtype=np.uint8)
                sols.append(y)
                sv.add_clause([-v if v in pos else v for v in self.yvar])
        return sols, aborted

    def _enumerate_cell_cms(self, clauses, cap: int, xors):
        from pycryptosat import Solver as CMSolver

        kw = {}
        if self.conf_budget is not None:
            kw["confl_limit"] = int(self.conf_budget)
        sv = CMSolver(**kw)
        for c in clauses:
            if not c:
                return [], False                   # empty clause: UNSAT
            sv.add_clause(c)
        for subset, parity in xors:
            sv.add_xor_clause(subset, bool(parity))
        sols = []
        while len(sols) <= cap:
            sat, model = sv.solve()
            if sat is None:
                return sols, True                  # budget ran out
            if not sat:
                break
            y = np.array([1 if model[v] else 0 for v in self.yvar],
                         dtype=np.uint8)
            sols.append(y)
            sv.add_clause([-v if model[v] else v for v in self.yvar])
        return sols, False

    def sample(self, x: np.ndarray, m_init: int | None = None,
               max_tries: int = 40, mode: str = "centered"):
        """One near-uniform shell sample. Returns (y | None, info dict).

        info: wall-clock seconds, SAT calls (cells enumerated), final m,
        cell size. y is None iff the shell is empty or tries ran out.
        """
        t0 = time.perf_counter()
        shell_clauses, top0 = self._shell_cnf(x, mode=mode)
        t_enc = time.perf_counter() - t0
        m = self.m_hint if m_init is None and hasattr(self, "m_hint") \
            else (m_init or 0)
        calls = 0
        for _ in range(max_tries):
            self._xor_top = top0
            xors = []
            for _k in range(m):
                subset = [v for v in self.yvar if self.rng.random() < 0.5]
                if not subset:
                    continue
                xors.append((subset, int(self.rng.integers(2))))
            sols, aborted = self._enumerate_cell(shell_clauses, self.cell_cap,
                                                 xors=xors)
            calls += 1
            if aborted:
                return None, dict(seconds=time.perf_counter() - t0,
                                  encode_s=t_enc, sat_calls=calls, m=m,
                                  cell=len(sols), aborted=True)
            if len(sols) > self.cell_cap:
                m += 2
                continue
            if not sols:
                if m == 0:
                    break                          # true empty shell
                m -= 1
                continue
            self.m_hint = m                        # warm start next call
            y = sols[int(self.rng.integers(len(sols)))]
            return y, dict(seconds=time.perf_counter() - t0, encode_s=t_enc,
                           sat_calls=calls, m=m, cell=len(sols),
                           aborted=False)
        return None, dict(seconds=time.perf_counter() - t0, encode_s=t_enc,
                          sat_calls=calls, m=m, cell=0, aborted=False)


__all__ = ["ShellHashSampler"]
