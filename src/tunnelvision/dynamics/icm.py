"""Houdayer isoenergetic cluster moves (ICM) on a two-replica space — M02.

Houdayer (2001); Zhu, Ochoa & Katzgraber, PRL 115, 077201 (2015). Two
replicas (x1, x2) target pi (x) pi (y). The move: pick a seed site in the
disagreement set D = {i : x1_i != x2_i}, grow a cluster C inside D along
coupling-graph edges, and flip C in BOTH replicas.

Structural facts this module encodes (and the tests certify):

* D is invariant under the move (disagreeing sites still disagree after
  the double flip), so with a deterministic cluster rule given (D, J,
  seed), the proposal is exactly symmetric: q(pair'|pair) = q(pair|pair').
* If C is a full connected component of D's coupling subgraph, the PAIR
  energy E(y1)+E(y2) - E(x1)-E(x2) is exactly zero — couplings crossing
  the component boundary vanish by definition, couplings inside cancel
  pairwise between replicas, and field terms cancel because s2 = -s1 on D.
  Random fields do NOT break the move (they cancel across replicas).
* On an all-to-all instance with continuous couplings every |J_ij| > 0,
  so the full component IS the whole of D, and flipping all of D swaps
  x1 <-> x2 exactly: faithful Houdayer degenerates to a statistical
  no-op. That percolation degeneracy is the known dense-graph failure.

Because of the last fact, the honest test on dense graphs needs
strengthened cluster rules, still symmetric by D-invariance:

* theta > 0: only traverse edges with |J_ij| >= theta (bond-threshold
  restriction; deterministic).
* size_cap k: best-first growth from the seed, expanding the strongest
  |J| edge first (deterministic tie-break by site index), stop at k
  sites. Uses coupling information — an "informed" cluster.

Partial clusters no longer conserve the pair energy, so the move is
Metropolis-accepted on the pair target with Delta(lp1 + lp2); the
proposal stays symmetric, so log_q terms cancel (returned as 0.0).
"""

from __future__ import annotations

import heapq

import numpy as np

from tunnelvision.bits import all_binary_states


class HoudayerICM:
    """Two-replica ICM proposal with threshold/size-cap cluster growth.

    Not a ``kernels.base.Kernel``: it acts on the pair space. The engine
    contract is honoured in spirit — the proposal never looks at target
    log-probs (cluster growth reads only D and |J|); acceptance is the
    caller's job via :meth:`pair_log_accept`.
    """

    def __init__(self, target, theta: float = 0.0, size_cap: int | None = None):
        self.target = target
        self.theta = float(theta)
        self.size_cap = size_cap
        # Edge rule: strictly-positive coupling AND |J| above theta.
        # Strict inequality also excludes absent (zero) couplings at
        # theta = 0, which is what 'faithful Houdayer' means on sparse
        # topologies; for continuous J the >= / > distinction is null.
        self.theta_edge = max(float(theta), 0.0)
        self.absJ = np.abs(np.asarray(target.J, dtype=np.float64))
        cap = "" if size_cap is None else f",cap={size_cap}"
        self.name = f"icm(theta={theta:g}{cap})"

    # ---------------------------------------------------------------- cluster
    def cluster(self, disagree: np.ndarray, seed_site: int) -> np.ndarray:
        """Deterministic cluster containing ``seed_site``.

        ``disagree``: sorted array of site indices in D. Returns a sorted
        array of cluster sites. Depends only on (D, |J|, theta, cap, seed):
        the same on both sides of the move, hence proposal symmetry.
        """
        d_set = set(int(i) for i in disagree)
        if int(seed_site) not in d_set:
            raise ValueError("seed_site must lie in the disagreement set")
        if self.size_cap is None:
            # Plain BFS over edges |J| >= theta restricted to D.
            todo = [int(seed_site)]
            out = {int(seed_site)}
            while todo:
                i = todo.pop()
                for j in d_set:
                    if j not in out and self.absJ[i, j] > self.theta_edge:
                        out.add(j)
                        todo.append(j)
            return np.array(sorted(out), dtype=np.int64)
        # Best-first: expand strongest frontier edge first, deterministic.
        out = {int(seed_site)}
        frontier: list[tuple[float, int]] = []
        def push_from(i: int) -> None:
            for j in d_set:
                if j not in out and self.absJ[i, j] > self.theta_edge:
                    heapq.heappush(frontier, (-float(self.absJ[i, j]), int(j)))
        push_from(int(seed_site))
        while frontier and len(out) < self.size_cap:
            _, j = heapq.heappop(frontier)
            if j in out:
                continue
            out.add(j)
            push_from(j)
        return np.array(sorted(out), dtype=np.int64)

    # ---------------------------------------------------------------- propose
    def propose_pair(
        self, x1: np.ndarray, x2: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One ICM proposal. Returns (y1, y2, cluster). Identity if D empty.

        Proposal is symmetric (see module docstring); log_q_fwd - log_q_rev
        = 0 always, so no Hastings term is returned.
        """
        disagree = np.flatnonzero(x1 != x2)
        if disagree.size == 0:
            return x1.copy(), x2.copy(), disagree
        seed_site = int(disagree[int(rng.integers(0, disagree.size))])
        C = self.cluster(disagree, seed_site)
        y1, y2 = x1.copy(), x2.copy()
        y1[C] = 1 - y1[C]
        y2[C] = 1 - y2[C]
        return y1, y2, C

    def pair_log_accept(self, x1, x2, y1, y2) -> float:
        """log acceptance ratio on the pair target (symmetric proposal)."""
        t = self.target
        return float(
            t.log_prob(y1) + t.log_prob(y2) - t.log_prob(x1) - t.log_prob(x2)
        )

    # ------------------------------------------------------- exact pair chain
    def pair_transition_matrix(self, n: int, w_icm: float = 0.5):
        """Sparse exact transition matrix of the mixture chain on pair space.

        P = w_icm * P_ICM + (1 - w_icm) * P_SF, both Metropolized against
        pi x pi; P_SF flips one uniformly chosen bit of one uniformly
        chosen replica. States indexed i1 * 2^n + i2. Returns
        (P_csr, pi_pair). Feasible at n <= 10 (4^10 ~ 1e6 states).
        """
        from scipy.sparse import coo_matrix

        S = all_binary_states(n)
        lp = self.target.log_prob_batch(S)
        N = 1 << n
        NN = N * N
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        pw = 1 << np.arange(n)
        w_sf = 1.0 - w_icm
        sf_mass = w_sf / (2 * n)
        for i1 in range(N):
            x1 = S[i1]
            for i2 in range(N):
                x2 = S[i2]
                s = i1 * N + i2
                acc_diag = 0.0
                # --- single-flip half: flip bit b of replica r
                for r, base in ((0, i1), (1, i2)):
                    for b in range(n):
                        nb = base ^ (1 << b)
                        t_idx = (nb * N + i2) if r == 0 else (i1 * N + nb)
                        dlp = lp[nb] - lp[base]
                        a = min(1.0, float(np.exp(dlp)))
                        if a > 0.0:
                            rows.append(s); cols.append(t_idx); vals.append(sf_mass * a)
                        acc_diag += sf_mass * (1.0 - a)
                # --- ICM half
                disagree = np.flatnonzero(x1 != x2)
                if disagree.size == 0:
                    acc_diag += w_icm
                else:
                    p_seed = w_icm / disagree.size
                    for seed_site in disagree:
                        C = self.cluster(disagree, int(seed_site))
                        mask = int(np.sum(pw[C]))
                        j1, j2 = i1 ^ mask, i2 ^ mask
                        dlp = (lp[j1] + lp[j2]) - (lp[i1] + lp[i2])
                        a = min(1.0, float(np.exp(dlp)))
                        t_idx = j1 * N + j2
                        if a > 0.0:
                            rows.append(s); cols.append(t_idx); vals.append(p_seed * a)
                        acc_diag += p_seed * (1.0 - a)
                if acc_diag > 0.0:
                    rows.append(s); cols.append(s); vals.append(acc_diag)
        P = coo_matrix((vals, (rows, cols)), shape=(NN, NN)).tocsr()
        pi = np.exp(lp - lp.max())
        pi /= pi.sum()
        pi_pair = np.kron(pi, pi)
        return P, pi_pair


def sparse_pair_gap(P, pi_pair, tol: float = 1e-10) -> float:
    """1 - |lambda_2| for a reversible sparse chain via symmetric similarity.

    The top eigenvector of the symmetrized chain is known exactly
    (sqrt(pi)); deflating it lets Lanczos target lambda_2 directly, which
    converges orders of magnitude faster than resolving the near-degenerate
    {lambda_1, lambda_2} pair at machine precision when the gap is tiny.
    Matches diagnostics.spectral_gap's 1 - |lambda_2| convention (the SA
    end is checked too).
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import LinearOperator, eigsh

    d = np.sqrt(pi_pair)
    Sym = diags(d) @ P @ diags(1.0 / d)
    Sym = (Sym + Sym.T) / 2.0  # kill float asymmetry; P must be reversible
    dv = d / np.linalg.norm(d)
    op = LinearOperator(
        Sym.shape, matvec=lambda v: Sym @ v - dv * (dv @ v), dtype=np.float64
    )
    lam2 = float(eigsh(op, k=1, which="LA", return_eigenvectors=False,
                       tol=tol, maxiter=200_000)[0])
    lam_min = float(eigsh(Sym, k=1, which="SA", return_eigenvectors=False,
                          tol=max(tol, 1e-8), maxiter=200_000)[0])
    return 1.0 - max(lam2, abs(lam_min))


__all__ = ["HoudayerICM", "sparse_pair_gap"]
