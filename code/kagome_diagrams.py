#!/usr/bin/env python3
"""
Diagrammatic (Wick / Pfaffian) perturbation theory around a TZ Gaussian reference state of the 6-Majorana representation of the spin-1/2 Heisenberg model on
an ARBITRARY graph (kagome lattice tori, rings with J2, small test graphs),   H = sum_b J_b S_i . S_j  (b = (i, j)),   orders 2, 3 and 4.

This is the graph version of heisenberg_diagrams.py (which is the special case "ring").  The algebra of vertices, labels, colour sums and cumulants is the same
and is imported from that file; what changes is that nothing is translation invariant any more:
  * the reference is an arbitrary real N x N matrix K  (H0 = i sum_a sum_ij K_ij rho^a_i eta^a_j), not a (mu, t, kappa) triple;
  * every bond b has its own vertex mean c_b = <S^a_i S^a_j>_0 (one colour) in the cumulants, since the reference need not be homogeneous;
  * the first vertex of a diagram runs over one representative bond of every orbit of the bond set under the lattice symmetries that leave K invariant
    (up to the gauge sign s_i = +-1 of a site; a lattice translation or a point-group operation), weighted with the orbit size; the other n-1 vertices run
    over ALL bonds.

1. Reference, as in heisenberg_diagrams.py:  chi = (rho_1..rho_N, eta_1..eta_N), h = [[0, 2iK], [-2iK^T, 0]], covariance C = <chi chi^T> = 2 U U^dagger (U = the N positive-energy
   eigenvectors, energies lambda_m = 2 sigma_m(K) > 0), propagator G(tau) = 2 U e^{-lambda tau} U^dagger.  <rho_i eta_j> = -i p_ij with P = -(orthogonal polar factor of K) for a pure
   aufbau state: this is the convention of kagome_gaussian_S1.py (the self-test checks it).  A reference with a zero mode (singular K) is rejected.
2. Series, as in heisenberg_diagrams.py:  P_n = -(-1)^n 3 sum_{b_1 in reps} |orbit(b_1)| J_{b_1} sum_{b_2..b_n} J_{b_2}..J_{b_n}  int_{t_1>..>t_n=0} kappa_n(v''_{b_1}, .., v''_{b_n}) dt,
   v'' = v - c_b (mean-removed vertex, first colour fixed to x, the others summed), E_n = sum_{j=0}^{n-2} C(n-2, j) P_{n-j}, E_0 + E_1 = <H>_0 = 3 sum_b J_b c_b.
3. THREE EXACT SPEED-UPS that did not exist for the ring (each is switched off by an option and the self-test shows that it changes nothing):
   (a) LOCALITY.  If the Majoranas of the reference split into decoupled classes (connected components of h; for a dimer covering every dimer carries two classes),
       the Pfaffian factorises over classes, and a configuration whose vertices split into two groups that share no class has an exactly vanishing cumulant (independent
       groups).  Only configurations that are connected through shared classes are summed (prune=True).
   (b) FLAT BAND.  If all quasi-particle energies are equal (every dimer covering of a uniform-coupling graph: lambda = 3/4), G(tau) = e^{-lambda tau} C, every Wick term
       is a monomial prod_k zeta_k^{m_k}, zeta_k = e^{-lambda tau_k}, where m_k = number of lines crossing the k-th cut (even, at most 8 for the cuts that separate one vertex,
       16 for the middle cut at n = 4), and every time integral is exactly  int_0^inf zeta^m dtau = 1/(m lambda).  The interval integrals are therefore done with an EXACT
       rule of q = M/2 nodes (4 for M = 8, 8 for M = 16; FlatRule) instead of the ~20 nodes of the Laplace quadrature, and with no quadrature error (quad="flat").
   (c) IDENTICAL INTEGRANDS.  For a flat band the integrand of a configuration depends only on the partition of its 4n legs into classes (the "pattern") and on the entries
       C[leg, leg'] inside the classes; for a dimer covering these are the same numbers (0, 1, +-i) for every dimer, so all configurations with the same pattern are the same
       number and are evaluated ONCE with their coupling weights added (collapse=True).  The number of distinct evaluations then saturates with N (kagome dimer coverings,
       any N from 12 to 36: about 500 at order 3 and 10-20 thousand at order 4) while the number of configurations grows like N.
   Other references (K with unequal quasi-particle energies, e.g. dimers of unequal couplings, or any non-dimer K) use the Laplace quadrature of heisenberg_diagrams.laplace_nodes
   (relative accuracy tol per interval) and evaluate every configuration.
4. Cost.  Measured here on one core: about 0.5-0.6 ms per (distinct evaluation, time-node tuple) at order 4 (80 node tuples after the time-reversal halving), a few times
   less at order 3 (10 node tuples).  Kagome N = 12, best covering: order 3 about 1 s, order 4 about 11 min (13.5 thousand evaluations x 80 nodes).  A reference with a single class
   (any non-dimer K on a lattice with triangles) uses the dense path with 2^(4n)-subset Pfaffians: only n <= 3 is practical there.
5. Dimer coverings.  enumerate_matchings / all_coverings list every perfect covering of the graph by dimers up to the lattice symmetries (kagome tori: 2, 6, 12, 19 inequivalent
   coverings for N = 12, 18a, 18b, 24); covering_table scores them by E_1 + E_2 and, among those that tie for the lowest E_1 + E_2 (N >= 24: all of them, E_2 = -17/768 per site for every covering), by E_1 + E_2 + E_3; dimer_K builds the reference K = -(3/8) J_ij on every dimer.  Checks:
   at the Majumdar-Ghosh point (ring, J2 = J1/2) the dimer product is exact and every E_n with n >= 2 vanishes to 1e-14.
6. Scale of the reference Hamiltonian.  The state of the reference is fixed by the mean-field minimum, but the scale of H0 (K -> s K, all quasi-particle energies times s) is not: the
   exact sum of the series does not depend on it, the truncated sums do.  rescaled_partial_sums(P, x = 1/s) gives S_n(x) = P_1 + sum_j C(n-1, j-1) P_j x^(j-1) exactly (self-test 8);
   the program itself uses s = 1 (K = gradient of the energy / 3, i.e. the mean-field Hamiltonian of the state).

Requires numpy, scipy and, in the same directory, heisenberg_diagrams.py, heisenberg_gaussian_S1.py, heisenberg_pt_checks.py, kagome_gaussian_S1.py, kagome_lattice.py, kagome_pt_checks.py.
Usage:  python3 kagome_diagrams.py kagome 12 --order 3              # dimer-covering reference: lowest E_1 + E_2 among all inequivalent coverings (ties: lowest E_1 + E_2 + E_3)
        python3 kagome_diagrams.py kagome 18a --order 4 --keep 2    # orders 1-4 for the two best coverings
        python3 kagome_diagrams.py ring 8 --J2 0.3 --order 3         # the J1-J2 ring around its lowest Gaussian state (a dimer covering; NOT the translation-invariant
                                                                     # reference of the chain code, which the self-test uses for its regression against heisenberg_diagrams.py)
        python3 kagome_diagrams.py selftest [--quick]
"""
import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heisenberg_diagrams as hd
import kagome_gaussian_S1 as kg
import kagome_lattice as kl

pf_all, contract_vertices, LAB_MASK = hd.pf_all, hd.contract_vertices, hd.LAB_MASK


# ================================================================================================ 1. reference
class GraphReference:
    """Gaussian reference H0 = i sum_a sum_ij K_ij rho^a_i eta^a_j of the six-Majorana representation on N sites (K real N x N, entries below clean*max|K| are set to 0)."""

    def __init__(self, K, clean=1e-9):
        K = np.array(K, float)
        K[np.abs(K) < clean * max(np.abs(K).max(), 1e-300)] = 0.0
        N = K.shape[0]
        self.N, self.K = N, K
        h = np.block([[np.zeros((N, N)), 2j * K], [-2j * K.T, np.zeros((N, N))]])
        lam, U = np.linalg.eigh(h)
        self.gap = float(np.abs(lam).min())
        if self.gap < 1e-9:
            raise ValueError("the reference has zero modes (singular K): perturbation theory around a degenerate reference is undefined")
        pos = lam > 0
        assert pos.sum() == N
        self.lam, self.U = lam[pos], U[:, pos]
        self.C = 2 * self.U @ self.U.conj().T                       # <chi_r chi_c>, row index = species * N + site
        self.ncls, self.cls = connected_components(np.abs(h) > 1e-12, directed=False)
        assert all(np.abs(self.C[np.ix_(self.cls == a, self.cls == b)]).max() < 1e-9 for a in range(self.ncls) for b in range(a + 1, self.ncls))
        self.factorised = bool(self.ncls > 1)
        self.flat = bool(self.lam.max() - self.lam.min() <= 1e-9 * self.lam.max())
        self.lam0 = float(self.lam.mean())

    def G(self, taus):
        """G(tau) = <chi(tau) chi(0)> for an array of tau > 0: shape (len(taus), 2N, 2N)."""
        E = np.exp(-np.outer(np.asarray(taus, float).ravel(), self.lam))
        return 2 * ((self.U[None] * E[:, None, :]) @ self.U.conj().T)

    def pmatrix(self):
        """p_ij = i <rho_i eta_j>."""
        return (1j * self.C[:self.N, self.N:]).real


def dimer_K(N, pairs, weights_of):
    """K of a product of dimers: K_ij = K_ji = -(3/8) J_ij on every dimer (i, j) (the gauge in which p_ij = p_ji = 1: each dimer is a singlet)."""
    K = np.zeros((N, N))
    for i, j in pairs:
        K[i, j] = K[j, i] = -0.375 * weights_of(i, j)
    return K


def bond_rows(N, bonds):
    """Rows (species * N + site) of the four legs (rho_i, eta_i, rho_j, eta_j) of every bond: shape (nb, 4)."""
    b = np.asarray(bonds, int)
    return np.stack([b[:, 0], N + b[:, 0], b[:, 1], N + b[:, 1]], axis=1)


def vertex_means(ref, rows):
    """c_b = <S^a_i S^a_j>_0 for one colour a (the mean-field bond energy per colour), for every bond (rows from bond_rows)."""
    nb = len(rows)
    A = np.zeros((nb, 4, 4), complex)
    for u in range(4):
        for v in range(u + 1, 4):
            A[:, u, v] = ref.C[rows[:, u], rows[:, v]]
            A[:, v, u] = -A[:, u, v]
    Phi = pf_all(A)[:, LAB_MASK]
    Psi = contract_vertices(Phi, 1)
    return (0.25 * sum(Psi[:, s] * Psi[:, 4 + s] ** 2 for s in range(4))).real


def first_order(ref, bonds, weights):
    """E_0 + E_1 = <H>_0 = 3 sum_b J_b c_b from the vertex algebra."""
    return float(3 * (np.asarray(weights, float) * vertex_means(ref, bond_rows(ref.N, bonds))).sum())


# ================================================================================================ 2. symmetry orbits of the bonds
def _preserves(K, p, tol=1e-8):
    """True if the site permutation p (site i -> p[i]) maps K onto itself up to a gauge sign s_i = +-1 of every site: s_a s_b K[pinv a, pinv b] = K[a, b]."""
    N = K.shape[0]
    pinv = np.empty_like(p)
    pinv[p] = np.arange(N)
    M = K[np.ix_(pinv, pinv)]
    if np.abs(np.abs(M) - np.abs(K)).max() > tol:
        return False
    nz = (np.abs(M) > tol) | (np.abs(M.T) > tol)
    s = np.zeros(N, int)
    for root in range(N):
        if s[root]:
            continue
        s[root] = 1
        stack = [root]
        while stack:
            a = stack.pop()
            for b in np.nonzero(nz[a])[0]:
                if b == a:
                    continue
                if abs(M[a, b]) > tol:
                    want = s[a] * int(np.sign(K[a, b]) * np.sign(M[a, b]))
                else:
                    want = s[a] * int(np.sign(K[b, a]) * np.sign(M[b, a]))
                if s[b] == 0:
                    s[b] = want
                    stack.append(b)
                elif s[b] != want:
                    return False
    return bool(np.abs(np.outer(s, s) * M - K).max() <= tol)


def symmetry_orbits(K, bonds, weights, perms):
    """Orbits of the bonds under those lattice symmetries (site permutations) that are automorphisms of the weighted graph and leave K invariant up to gauge signs.
       Returns (reps, orbit sizes, number of symmetries used)."""
    bonds = np.asarray(bonds, int)
    nb = len(bonds)
    bidx = {(int(i), int(j)): k for k, (i, j) in enumerate(bonds)}
    parent = list(range(nb))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    used = 0
    for p in perms or []:
        p = np.asarray(p, int)
        img = []
        for k, (i, j) in enumerate(bonds):
            a, b = int(p[i]), int(p[j])
            kk = bidx.get((min(a, b), max(a, b)))
            if kk is None or abs(weights[kk] - weights[k]) > 1e-12:
                break
            img.append(kk)
        else:
            if _preserves(K, p):
                used += 1
                for k, kk in enumerate(img):
                    parent[find(k)] = find(kk)
    roots = np.array([find(k) for k in range(nb)])
    reps, mult = [], []
    for r in np.unique(roots):
        mem = np.nonzero(roots == r)[0]
        reps.append(int(mem[0]))
        mult.append(len(mem))
    return np.array(reps), np.array(mult, float), used


# ================================================================================================ 3. time quadrature
class FlatRule:
    """Exact quadrature for a flat band: nodes zeta_j = exp(-lambda tau_j) and weights w_j with sum_j w_j zeta_j^m = 1/(m lambda) for m = 2, 4, .., M
       (cut energies m lambda; Chebyshev nodes in u = zeta^2 keep the weights O(1/lambda))."""

    def __init__(self, lam, M, ulo=0.02):
        q = M // 2
        j = np.arange(q)
        u = ulo + (1 - ulo) * 0.5 * (1 + np.cos(np.pi * (2 * j + 1) / (2 * q)))
        V = np.array([[uj ** (l + 1) for uj in u] for l in range(q)])
        self.w = np.linalg.solve(V, np.array([1.0 / (2 * (l + 1) * lam) for l in range(q)]))
        self.zeta = np.sqrt(u)
        self.alpha = -np.log(self.zeta) / lam
        self.M = M


class TimeRule:
    """Quadrature over the n-1 time intervals: per interval a node array and weights; flat (exact) or Laplace (continuous cut energies)."""

    def __init__(self, ref, n, tol=1e-6, quad="auto"):
        self.flat = bool(ref.flat) and quad in ("auto", "flat")
        if quad == "flat" and not ref.flat:
            raise ValueError("flat quadrature needs a reference with equal quasi-particle energies")
        if self.flat:
            Ms = {2: (8,), 3: (8, 8), 4: (8, 16, 8)}[n]
            cache = {}
            rules = [cache.setdefault(M, FlatRule(ref.lam0, M)) for M in Ms]
            self.alpha = [r.alpha for r in rules]
            self.zeta = [r.zeta for r in rules]
            self.w = [r.w for r in rules]
            self.err = 0.0
        else:
            lam = ref.lam
            Omin = 2 * lam.min()
            Omax = min((16 if n == 4 else 8) * lam.max(), 3 * lam.sum())
            a, w, self.err = hd.laplace_nodes(Omin, max(Omax, 1.01 * Omin), tol)
            self.alpha, self.w, self.zeta = [a] * (n - 1), [w] * (n - 1), None
        self.sizes = [len(a) for a in self.alpha]

    def nodes(self, symmetric=True):
        """(index tuples, weights) of the product rule; the reversal symmetry I(tau_1..tau_{n-1})^* = I(tau_{n-1}..tau_1) of the summed integrand halves the grid."""
        idx = np.array(list(itertools.product(*[range(s) for s in self.sizes])))
        mult = np.ones(len(idx))
        n1 = idx.shape[1]
        if symmetric and n1 >= 2 and self.sizes[0] == self.sizes[-1]:
            lo, hi = idx[:, 0], idx[:, -1]
            keep = lo <= hi
            mult = np.where(lo < hi, 2.0, 1.0)[keep]
            idx = idx[keep]
        wts = np.prod([self.w[k][idx[:, k]] for k in range(n1)], axis=0) * mult
        return idx, wts


class NodeChunk:
    """Propagators for a batch of time-node tuples: pair(p, q, rows_p, rows_q) = G(t_p - t_q)[rows_p, rows_q] for vertices p < q, shape (g, m)."""

    def __init__(self, ref, rule, idx):
        self.ref, self.g = ref, len(idx)
        n1 = idx.shape[1]
        self.flat = rule.flat
        if rule.flat:
            z = np.stack([rule.zeta[k][idx[:, k]] for k in range(n1)], axis=1)
            self.Z = {(p, q): z[:, p:q].prod(axis=1) for p in range(n1 + 1) for q in range(p + 1, n1 + 1)}
        else:
            tau = np.stack([rule.alpha[k][idx[:, k]] for k in range(n1)], axis=1)
            self.Gp = {(p, q): ref.G(tau[:, p:q].sum(axis=1)) for p in range(n1 + 1) for q in range(p + 1, n1 + 1)}

    def pair(self, p, q, ra, rb):
        if self.flat:
            return self.Z[(p, q)][:, None] * self.ref.C[ra, rb][None, :]
        return self.Gp[(p, q)][:, ra, rb]


# ================================================================================================ 4. configurations, patterns, integrand
def _canonical_patterns(cls_legs):
    """Group configurations by the partition of their 4n legs into decoupled classes.  cls_legs: (M, 4n) class ids.
       Returns a list of (pats, ncls, member indices) with pats = per-vertex tuples of canonical (first-appearance) class labels."""
    M, L = cls_legs.shape
    eq = cls_legs[:, :, None] == cls_legs[:, None, :]
    first = eq.argmax(axis=2)
    uniq, inv = np.unique(first, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    order = np.argsort(inv, kind="stable")
    bounds = np.searchsorted(inv[order], np.arange(len(uniq) + 1))
    out = []
    for u, row in enumerate(uniq):
        labels, ids = {}, []
        for t in range(L):
            f = int(row[t])
            if f not in labels:
                labels[f] = len(labels)
            ids.append(labels[f])
        pats = tuple(tuple(ids[4 * p:4 * p + 4]) for p in range(L // 4))
        out.append((pats, len(labels), order[bounds[u]:bounds[u + 1]]))
    return out


def class_maps_cached(pats, ncls):
    if len(hd._CM) > 2000:
        hd._CM.clear()
    return hd.class_maps(pats, ncls)


class GraphDiagrams:
    """P_n (the plain-cumulant energy of order n, n = 2, 3, 4) for the reference `ref` on the weighted graph (bonds, weights).
       perms: lattice symmetries (site permutations) used to reduce the first vertex (None: no reduction)."""

    def __init__(self, ref, bonds, weights, n, perms=None, tol=1e-6, prune=True, symmetric=True, dense=None, quad="auto", verbose=False, budget=2.5e8, collapse=True):
        if n not in (2, 3, 4):
            raise ValueError("order n must be 2, 3 or 4")
        self.ref, self.n, self.verbose, self.budget = ref, n, verbose, budget
        self.bonds = np.asarray(bonds, int)
        self.J = np.asarray(weights, float)
        self.nb = len(self.bonds)
        self.dense = (not ref.factorised) if dense is None else dense
        if not self.dense and not ref.factorised:
            raise ValueError("the factorised path needs a reference with decoupled classes; use dense=True")
        self.brows = bond_rows(ref.N, self.bonds)
        self.cbar = vertex_means(ref, self.brows)
        self.reps, self.mult, self.nsym = symmetry_orbits(ref.K, self.bonds, self.J, perms) if perms is not None else (np.arange(self.nb), np.ones(self.nb), 1)
        # bonds that share a decoupled class (all pairs if the reference has one class or pruning is off)
        if prune and not self.dense:
            inc = np.zeros((self.nb, ref.ncls), bool)
            for b in range(self.nb):
                inc[b, ref.cls[self.brows[b]]] = True
            self.adj = (inc.astype(int) @ inc.astype(int).T) > 0
        else:
            self.adj = np.ones((self.nb, self.nb), bool)
        self.rule = TimeRule(ref, n, tol, quad)
        self.nidx, self.nwts = self.rule.nodes(symmetric)
        self.cs = hd.ColourSums(None)
        self._kw = {}
        self.collapse = bool(ref.flat) and collapse
        self.groups, self.nconf = self._build()

    # ---- configurations: first vertex = orbit representative, the others all bonds, restricted to connected configurations;
    #      grouped by the partition of the 4n legs into decoupled classes ("pattern").  For a flat band the integrand of a configuration is a function of the pattern
    #      and of the contraction entries C[leg, leg'] inside the classes only, so configurations with identical entries are evaluated ONCE with their weights added.
    def _connected(self, b1, grid):
        n = self.n
        vert = [np.full(len(grid), b1)] + [grid[:, k] for k in range(n - 1)]
        edge = lambda p, q: self.adj[vert[p], vert[q]] if p < q else self.adj[vert[q], vert[p]]
        r = [np.ones(len(grid), bool)] + [edge(0, q) for q in range(1, n)]
        for _ in range(n - 2):
            for q in range(1, n):
                for p in range(1, n):
                    if p != q:
                        r[q] = r[q] | (r[p] & edge(p, q))
        ok = np.ones(len(grid), bool)
        for q in range(1, n):
            ok &= r[q]
        return np.concatenate([np.full((int(ok.sum()), 1), b1), grid[ok]], axis=1)

    def _fingerprint(self, pats, cfg):
        """Everything the flat-band integrand depends on besides the pattern: C between the legs of one class, and the vertex means."""
        n = self.n
        cl = [c for pat in pats for c in pat]
        R = self.brows[cfg].reshape(len(cfg), 4 * n)
        cols = []
        for a in range(4 * n):
            for b in range(a + 1, 4 * n):
                if cl[a] == cl[b]:
                    v = self.ref.C[R[:, a], R[:, b]]
                    cols += [v.real, v.imag]
        cm = self.cbar[cfg]
        cols += [cm[:, p] for p in range(n)]
        return np.stack(cols, axis=1)

    def _build(self):
        n, nb, ref = self.n, self.nb, self.ref
        grid = np.indices((nb,) * (n - 1)).reshape(n - 1, -1).T
        pats0 = tuple((0, 0, 0, 0) for _ in range(n))
        acc, plain, nconf = {}, {}, 0
        for b1, mu in zip(self.reps, self.mult):
            cfg = self._connected(b1, grid)
            wf = mu * self.J[cfg].prod(axis=1)
            nconf += len(cfg)
            sets = [(pats0, 1, np.arange(len(cfg)))] if self.dense else _canonical_patterns(ref.cls[self.brows[cfg]].reshape(len(cfg), -1))
            for pats, ncls, mem in sets:
                if self.collapse:
                    fp = np.round(self._fingerprint(pats, cfg[mem]), 9) + 0.0
                    uq, first, inv = np.unique(fp, axis=0, return_index=True, return_inverse=True)
                    wsum = np.bincount(np.asarray(inv).reshape(-1), weights=wf[mem], minlength=len(uq))
                    for u in range(len(uq)):
                        key = (pats, uq[u].tobytes())
                        if key in acc:
                            acc[key][1] += wsum[u]
                        else:
                            acc[key] = [cfg[mem[first[u]]], wsum[u]]
                else:
                    e = plain.setdefault(pats, ([], []))
                    e[0].append(cfg[mem])
                    e[1].append(wf[mem])
        if self.collapse:
            tmp = {}
            for (pats, _), (c, w) in acc.items():
                g = tmp.setdefault(pats, ([], []))
                g[0].append(c)
                g[1].append(w)
            plain = {p: ([np.array(c)], [np.array(w)]) for p, (c, w) in tmp.items()}
        groups = [(pats, max(max(pt) for pt in pats) + 1, np.concatenate(cs), np.concatenate(ws)) for pats, (cs, ws) in plain.items()]
        return groups, nconf

    def counts(self):
        return dict(orbits=len(self.reps), symmetries=self.nsym, configurations=self.nconf, patterns=len(self.groups),
                    evaluations=sum(len(g[2]) for g in self.groups), nodes=len(self.nidx),
                    quadrature="flat/exact" if self.rule.flat else f"Laplace ({self.rule.sizes[0]} nodes per interval)")

    # ---- colour sums with per-vertex means
    def _kind_weights(self, m, c):
        kinds, _, _ = self.cs.tables(m, True)
        if m not in self._kw:
            self._kw[m] = np.array(kinds) == 4
        W = np.ones((c.shape[0], len(kinds)))
        for p in range(m):
            W *= np.where(self._kw[m][None, :, p], -c[:, p][:, None], 0.25)
        return W

    def _colour_sum(self, Psi, c, m, fixed):
        _, mult, (i0, i1, i2) = self.cs.tables(m, fixed)
        T = Psi[:, i0]
        T *= Psi[:, i1]
        T *= Psi[:, i2]
        Tm = np.tensordot(T.reshape(Psi.shape[0], len(mult), -1), mult, axes=([1], [0]))
        return (Tm * self._kind_weights(m, c)).sum(axis=1)

    def _integrand(self, R, cb, pats, ncls, maps, nd):
        """kappa_n of the mean-removed vertices, colour-summed (first colour fixed), for every node tuple of `nd` and every configuration of R: shape (g, m)."""
        n, ref = self.n, self.ref
        g, m = nd.g, R.shape[0]
        masks, sign, _ = maps
        Phi = np.empty((g, m, 8 ** n), complex)
        Phi[:] = sign
        for c in range(ncls):
            legs = [(p, l) for p in range(n) for l in range(4) if pats[p][l] == c]
            sz = len(legs)
            A = np.zeros((g, m, sz, sz), complex)
            for a in range(sz):
                pa, la = legs[a]
                ra = R[:, pa, la]
                for b in range(a + 1, sz):
                    pb, lb = legs[b]
                    rb = R[:, pb, lb]
                    v = ref.C[ra, rb][None, :] if pa == pb else nd.pair(pa, pb, ra, rb)
                    A[:, :, a, b] = v
                    A[:, :, b, a] = -v
            P = pf_all(A.reshape(g * m, sz, sz)).reshape(g, m, 1 << sz)
            Phi *= P[:, :, masks[c]]
        Psi = contract_vertices(Phi.reshape(g * m, 8 ** n), n)
        cm = np.broadcast_to(cb[None], (g, m, n)).reshape(g * m, n)
        if n <= 3:
            res = self._colour_sum(Psi, cm, n, True)
        else:
            res = self._colour_sum(Psi, cm, 4, True)
            pair = lambda p, q: self._colour_sum(hd._pair_slice(Psi, 4, p, q), cm[:, [p, q]], 2, p == 0)
            res = res - pair(0, 1) * pair(2, 3) - pair(0, 2) * pair(1, 3) - pair(0, 3) * pair(1, 2)
        return res.reshape(g, m)

    def _rows_cap(self, maps):
        n = self.n
        _, mult, _ = self.cs.tables(n, True)
        nk = len(self.cs.tables(n, True)[0])
        per_row = 16.0 * (2.0 ** max(maps[2]) + 3 * 8 ** n + 3 * 9 ** n + 4 * len(mult) * nk)
        return int(min(20000, max(1, self.budget / per_row)))

    def energy(self):
        """P_n = -(-1)^n 3 sum (orbit size) (couplings) integral kappa_n  (real)."""
        n = self.n
        total, done, t0 = 0j, 0, time.time()
        gmax = 64 if self.rule.flat else 32
        chunks = [(s, NodeChunk(self.ref, self.rule, self.nidx[s:s + gmax])) for s in range(0, len(self.nidx), gmax)] if self.rule.flat or n < 4 else None
        nrows = sum(len(g[2]) for g in self.groups) * len(self.nidx)
        for pats, ncls, cfgs, wts in self.groups:
            maps = class_maps_cached(pats, ncls)
            cap = self._rows_cap(maps)
            it = chunks if chunks is not None else ((s, NodeChunk(self.ref, self.rule, self.nidx[s:s + gmax])) for s in range(0, len(self.nidx), gmax))
            for s, nd in it:
                wch = self.nwts[s:s + nd.g]
                msub = max(1, cap // nd.g)
                for s2 in range(0, len(cfgs), msub):
                    cfg = cfgs[s2:s2 + msub]
                    I = self._integrand(self.brows[cfg], self.cbar[cfg], pats, ncls, maps, nd)
                    total += (wch[:, None] * wts[s2:s2 + msub][None, :] * I).sum()
                    done += nd.g * len(cfg)
            if self.verbose and (done // max(1, nrows // 20)) != ((done - len(cfgs) * len(self.nidx)) // max(1, nrows // 20)):
                print(f"      n={n}: {done}/{nrows} rows  {time.time() - t0:.0f} s", flush=True)
        return float((-(-1) ** n * 3 * total).real)


# ================================================================================================ 5. series
def energies(P, nmax):
    """Rayleigh-Schroedinger E_n from the plain-cumulant energies P_n (P[1] = <H>_0):  E_n = sum_j C(n-2, j) P_{n-j}."""
    E = {1: P[1]}
    for n in range(2, nmax + 1):
        E[n] = sum(math.comb(n - 2, j) * P[n - j] for j in range(n - 1))
    return E


def rescaled_partial_sums(P, x):
    """Partial sums S_1..S_nmax (list) of the series when the reference Hamiltonian is multiplied by s = 1/x (H0 -> s H0: same reference state, all quasi-particle
       energies times s).  The family (1 - xi) s H0 + xi H is the family (1 - eta) H0 + eta H with eta = xi / (s + (1 - s) xi), multiplied by s + (1 - s) xi, which gives exactly
           S_n(x) = P_1 + sum_{j=2}^{n} C(n-1, j-1) P_j x^(j-1)          (each P_j contains j - 1 time integrals, i.e. scales like s^-(j-1)).
       x = 1 is the series of this program; P = series(...)["P"].  Checked against the Fock-space series of s H0 in the self-test (section 8)."""
    return [P[1] + sum(math.comb(n - 1, j - 1) * P[j] * x ** (j - 1) for j in range(2, n + 1)) for n in range(1, max(P) + 1)]


def series(ref, bonds, weights, nmax=3, perms=None, tol=1e-6, verbose=False, **kw):
    """Orders 1..nmax: dict(P, E, counts)."""
    P = {1: first_order(ref, bonds, weights)}
    counts = {}
    for n in range(2, nmax + 1):
        t0 = time.time()
        d = GraphDiagrams(ref, bonds, weights, n, perms, tol, verbose=verbose, **kw)
        counts[n] = d.counts()
        P[n] = d.energy()
        counts[n]["seconds"] = time.time() - t0
    return dict(P=P, E=energies(P, nmax), counts=counts)


# ================================================================================================ 6. dimer coverings
def random_matching(N, bonds, rng):
    """A random perfect matching of the graph (randomised backtracking), as a sorted tuple of pairs; None if there is none."""
    nbr = [[] for _ in range(N)]
    for i, j in np.asarray(bonds, int):
        nbr[int(i)].append(int(j))
        nbr[int(j)].append(int(i))
    free, pairs = np.ones(N, bool), []

    def rec():
        if not free.any():
            return True
        cand = np.nonzero(free)[0]
        i = int(min(cand, key=lambda s: (sum(free[t] for t in nbr[s]), rng.random())))
        opts = [t for t in nbr[i] if free[t]]
        rng.shuffle(opts)
        for t in opts:
            free[i] = free[t] = False
            pairs.append((min(i, t), max(i, t)))
            if rec():
                return True
            pairs.pop()
            free[i] = free[t] = True
        return False
    return tuple(sorted(pairs)) if rec() else None


def canonical_matching(pairs, perms):
    """Representative of a matching under the lattice symmetries (the lexicographically smallest image)."""
    best = None
    for p in perms:
        img = tuple(sorted((min(int(p[i]), int(p[j])), max(int(p[i]), int(p[j]))) for i, j in pairs))
        if best is None or img < best:
            best = img
    return best


def sample_coverings(N, bonds, perms, count, seed=0, tries=20):
    """Up to `count` inequivalent random perfect matchings."""
    rng = np.random.default_rng(seed)
    seen, out = set(), []
    for _ in range(count * tries):
        m = random_matching(N, bonds, rng)
        if m is None:
            break
        key = canonical_matching(m, perms)
        if key not in seen:
            seen.add(key)
            out.append(m)
            if len(out) >= count:
                break
    return out


def enumerate_matchings(N, bonds, limit=200000):
    """All perfect matchings of the graph as sorted tuples of pairs; None if there are more than `limit` of them."""
    nbr = [[] for _ in range(N)]
    for i, j in np.asarray(bonds, int):
        nbr[int(i)].append(int(j))
        nbr[int(j)].append(int(i))
    out, cur, free = [], [], [True] * N

    def rec():
        if len(out) > limit:
            return
        i = next((s for s in range(N) if free[s]), None)
        if i is None:
            out.append(tuple(sorted(cur)))
            return
        free[i] = False
        for t in nbr[i]:
            if free[t]:
                free[t] = False
                cur.append((min(i, t), max(i, t)))
                rec()
                cur.pop()
                free[t] = True
        free[i] = True
    rec()
    return None if len(out) > limit else out


def all_coverings(N, bonds, perms, limit=200000):
    """All perfect dimer coverings up to the lattice symmetries (one representative each), or None if there are more than `limit` perfect matchings."""
    ms = enumerate_matchings(N, bonds, limit)
    if ms is None:
        return None
    seen = {}
    for m in ms:
        seen.setdefault(canonical_matching(m, perms), m)
    return [seen[k] for k in sorted(seen)]


def _quantise(x, tie=1e-9):
    """Bin index of x on a grid of width `tie` (grid shifted off the simple rationals that occur), so that energies equal up to rounding noise fall into the same bin."""
    return int(np.floor(x / tie + 0.3141592653))


def covering_table(N, bonds, weights, perms, ncover=24, seed=0, tol=1e-6, max_sample=200, verbose=False, tie=1e-9):
    """Every inequivalent dimer covering (all of them if there are at most 2e5 perfect matchings, else `max_sample` random ones) scored by P_1 and P_2.
       P_3 is then computed for every covering whose E_1 + E_2 equals the best one within `tie` (for N >= 24 that is ALL of them: the second-order energy is the same
       bulk value -17/768 per site for every covering) and, if there are fewer than `ncover` of those, for the next best ones up to `ncover`.
       Returns (rows, True if the enumeration was exhaustive).  A row is dict(pairs=[[i, j], ..], P=[P_1, P_2, P_3 or None]); the rows are sorted by E_1 + E_2,
       coverings with equal E_1 + E_2 (within `tie`) by E_1 + E_2 + E_3 (lowest first), equal values in the order of enumeration; rows[0] is the reference used everywhere."""
    covers = all_coverings(N, bonds, perms)
    exhaustive = covers is not None
    if covers is None:
        covers = sample_coverings(N, bonds, perms, max_sample, seed)
    if not covers:
        raise ValueError("no perfect dimer covering exists (odd N): the dimer reference needs an even number of sites")
    rows, t0 = [], time.time()
    for pairs in covers:
        ref = covering_reference(N, bonds, weights, pairs)
        r = series(ref, bonds, weights, 2, perms, tol)
        rows.append({"pairs": [list(p) for p in pairs], "P": [r["P"][1], r["P"][2], None]})
    q12 = [_quantise(r["P"][0] + r["P"][1], tie) for r in rows]
    rows = [rows[i] for i in sorted(range(len(rows)), key=lambda i: (q12[i], i))]
    q12 = sorted(q12)
    if verbose:
        print(f"   {len(rows)} inequivalent coverings ({'exhaustive' if exhaustive else 'random sample'}) scored at order 2 ({time.time() - t0:.0f} s)", flush=True)
    ntie = sum(1 for q in q12 if q == q12[0])
    n3 = max(ncover, ntie)
    t0 = time.time()
    for row in rows[:n3]:
        ref = covering_reference(N, bonds, weights, [tuple(p) for p in row["pairs"]])
        row["P"][2] = series(ref, bonds, weights, 3, perms, tol)["P"][3]
    q123 = [_quantise(r["P"][0] + 2 * r["P"][1] + r["P"][2], tie) if r["P"][2] is not None else 10 ** 18 for r in rows]
    rows = [rows[i] for i in sorted(range(len(rows)), key=lambda i: (q12[i], q123[i], i))]
    if verbose:
        print(f"   order 3 for {min(n3, len(rows))} coverings ({ntie} of them tie for the lowest E_1 + E_2) ({time.time() - t0:.0f} s)", flush=True)
    return rows, exhaustive


def covering_reference(N, bonds, weights, pairs):
    w = {(int(i), int(j)): float(x) for (i, j), x in zip(bonds, weights)}
    return GraphReference(dimer_K(N, pairs, lambda i, j: w[(min(i, j), max(i, j))]))


# ================================================================================================ 7. drivers
def run_kagome(name, nmax, ncover, tol, seed, ed, verbose, keep=2):
    T = kl.torus_from_name(name)
    N, bonds, J, perms = T.N, T.bonds, T.couplings(), T.all_symmetries()
    print(f"kagome torus {T.name} (N = {N}): {len(bonds)} bonds, {len(perms)} lattice symmetries")
    Eex = None
    if ed and N <= 24:
        t0 = time.time()
        Eex = kg.exact_energy(N, bonds, J)[0]
        print(f"exact diagonalisation: E_0 = {Eex:.10f}  ({Eex / N:.6f} per site)  [{time.time() - t0:.0f} s]")
    elif name in kg.LITERATURE_ED:
        Eex = kg.LITERATURE_ED[name][0]
        print(f"literature ED total energy (unverified here): {Eex:.9f}  ({Eex / N:.6f} per site)")
    rows, exhaustive = covering_table(N, bonds, J, perms, ncover, seed, tol, verbose=True)
    e12 = np.array([r["P"][0] + r["P"][1] for r in rows])
    e123 = np.array([r["P"][0] + r["P"][1] + r["P"][2] + r["P"][1] for r in rows if r["P"][2] is not None])
    print(f"   E_1 + E_2 over the {len(rows)} coverings: best {e12.min():.8f}, median {np.median(e12):.8f}, worst {e12.max():.8f}   (mean field {-0.375 * N:.8f})")
    print(f"   E_1 + E_2 + E_3 over the {len(e123)} coverings for which it was computed: best {e123.min():.8f}, median {np.median(e123):.8f}, worst {e123.max():.8f}")
    for rank, row in enumerate(rows[:keep]):
        pairs = [tuple(p) for p in row["pairs"]]
        ref = covering_reference(N, bonds, J, pairs)
        t0 = time.time()
        res = series(ref, bonds, J, nmax, perms, tol, verbose)
        print(f"\ncovering #{rank + 1} (lowest E_1 + E_2, ties: lowest E_1 + E_2 + E_3):  dimers {pairs}")
        report(res, N, Eex)
        print(f"   {time.time() - t0:.0f} s;  " + "; ".join(f"n={n}: {c['configurations']} configs -> {c['evaluations']} evaluations x {c['nodes']} nodes, {c['quadrature']}, {c['seconds']:.0f} s" for n, c in res["counts"].items()))
    return rows


def report(res, N, Eex):
    S = 0.0
    nmax = max(res["E"])
    print(f"   {'order':>5} {'E_n':>14} {'partial sum':>14} {'per site':>10}" + (f" {'from exact':>11}" if Eex is not None else ""))
    for n in range(1, nmax + 1):
        S += res["E"][n]
        label = f"{n}" + (" (MF)" if n == 1 else "")
        print(f"   {label:>5} {res['E'][n]:>+14.8f} {S:>14.8f} {S / N:>10.6f}" + (f" {100 * (S - Eex) / abs(Eex):>+10.2f}%" if Eex is not None else ""))


def run_ring(Ns, nmax, J2s, tol, verbose):
    """The J1-J2 ring through the graph engine, around the lowest Gaussian state that kagome_gaussian_S1.minimise finds on the ring graph (a dimer covering, as for the kagome clusters;
       not the translation-invariant reference of heisenberg_gaussian_S1)."""
    for N in Ns:
        for J2 in J2s:
            bonds, w = kg.ring_graph(N, J2)
            r = kg.minimise(N, bonds, w, starts=24, seed=0)
            ref = GraphReference(r["K"])
            perms = [(np.arange(N) + s) % N for s in range(N)] + [(-np.arange(N) + s) % N for s in range(N)]
            Eex = kg.exact_energy(N, bonds, w)[0]
            res = series(ref, bonds, w, nmax, perms, tol, verbose)
            print(f"\nring N = {N}, J2 = {J2:g}: reference = lowest Gaussian state (E = {r['E']:.8f}, gap {r['gap']:.4f}, {'flat band' if ref.flat else 'dispersive'}, {ref.ncls} classes);  ED = {Eex:.10f}")
            report(res, N, Eex)


# ================================================================================================ 8. self-test
def _fock_context(N, bonds, weights, K):
    import heisenberg_pt_checks as pc
    import kagome_pt_checks as kp
    gam = pc.majoranas(3 * N)
    Sl = [pc.spin_6(gam[6 * i:6 * i + 6]) for i in range(N)]
    H = kp.graph_hamiltonian(Sl, bonds, weights)
    H0 = kp.mean_field_K(gam, K)
    return pc, kp, gam, Sl, H, H0


def selftest(quick=False):
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
    bad = 0

    def verdict(ok):
        nonlocal bad
        bad += 0 if ok else 1
        return "ok" if ok else "FAILED"

    rng = np.random.default_rng(3)
    tri = (3, np.array([[0, 1], [0, 2], [1, 2]]), np.array([1.0, 1.3, 0.7]))
    k4 = (4, np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]), np.array([1.0, 0.6, 0.8, 1.2, 0.5, 1.0]))
    bow = (5, np.array([[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4]]), np.ones(6))
    import kagome_pt_checks as kp

    print("1. conventions and first order: <H>_0 = 3 sum_b J_b c_b (vertex algebra)  vs  the energy of kagome_gaussian_S1.minimise; p = i<rho eta> vs the minimiser's P")
    for label, (N, bonds, w) in [("K4 (weighted)", k4), ("ring 6, J2 = 0.3", (6, *kg.ring_graph(6, 0.3))), ("kagome 12", (12, kl.torus_from_name("12").bonds, kl.torus_from_name("12").couplings()))]:
        r = kg.minimise(N, bonds, w, starts=12, seed=1)
        if r["gap"] < 1e-6:
            print(f"   {label}: minimiser state has zero modes - skipped")
            continue
        ref = GraphReference(r["K"])
        d1 = abs(first_order(ref, bonds, w) - r["E"])
        d2 = np.abs(ref.pmatrix() - r["P"]).max() if r["aufbau"] else np.nan
        print(f"   {label}: <H>_0 = {first_order(ref, bonds, w):+.10f}  (minimiser {r['E']:+.10f})  |diff| = {d1:.1e};  max|p - P| = {d2:.1e}  {verdict(d1 < 1e-8 and (np.isnan(d2) or d2 < 1e-8))}")

    print("2. integrands (n = 2, 3, 4)  vs  exact Fock-space imaginary-time correlators of the cumulants (random bonds, random times)")
    cases = [("triangle, random K (dense path)", tri, kp.random_reference(3, tri[1], rng, 0.8)),
             ("K4, two dimers (flat, factorised)", k4, dimer_K(4, [(0, 1), (2, 3)], lambda i, j: 1.0)),
             ("K4, random K (dense path)", k4, kp.random_reference(4, k4[1], rng, 0.7))]
    if not quick:
        cases.append(("bowtie, random K (dense path)", bow, kp.random_reference(5, bow[1], rng, 0.7)))
    for label, (N, bonds, w), K in cases:
        ref = GraphReference(K)
        pc, _, gam, Sl, H, H0 = _fock_context(N, bonds, w, K)
        wv, v = pc.lowest_two(H0)
        psi0, E0 = v[:, 0], wv[0]
        eye = sp.identity(H0.shape[0], format="csr")
        brows = bond_rows(N, bonds)
        cb = vertex_means(ref, brows)
        worst, scale = 0.0, 0.0
        for n in (2, 3, 4):
            for trial in range(2 if n == 4 else 3):
                for dense in ([True] if not ref.factorised else [True, False]):
                    d = GraphDiagrams(ref, bonds, w, n, None, 1e-3, prune=False, dense=dense, symmetric=False)
                    cfg = np.concatenate([[rng.integers(0, len(bonds))], rng.integers(0, len(bonds), n - 1)])
                    times = np.sort(rng.uniform(0.2, 3.0, n))[::-1]
                    times = times - times[-1]
                    # propagator data for these times through the generic (Laplace) NodeChunk interface
                    nd = NodeChunk.__new__(NodeChunk)
                    nd.ref, nd.g, nd.flat = ref, 1, False
                    nd.Gp = {(p, q): ref.G([times[p] - times[q]]) for p in range(n) for q in range(p + 1, n)}
                    pats, ncls = ((tuple((0, 0, 0, 0) for _ in range(n)), 1) if dense else (None, None))
                    if not dense:
                        pats, ncls, _ = _canonical_patterns(ref.cls[brows[cfg]].reshape(1, -1))[0]
                    maps = class_maps_cached(pats, ncls)
                    got = d._integrand(brows[cfg][None], cb[cfg][None], pats, ncls, maps, nd)[0, 0]

                    def corr(idx, colours):
                        vec = None
                        for k in range(len(idx) - 1, -1, -1):
                            p = idx[k]
                            i, j = bonds[cfg[p]]
                            V = Sl[i][colours[p]] @ Sl[j][colours[p]]
                            ww = psi0 if vec is None else vec
                            vec = V @ ww - cb[cfg[p]] * ww
                            if k > 0:
                                vec = expm_multiply(-(H0 - E0 * eye) * (times[idx[k - 1]] - times[p]), vec)
                        return np.vdot(psi0, vec)
                    ref_val = 0.0
                    for cols in itertools.product(range(3), repeat=n - 1):
                        cl = (0,) + cols
                        if n < 4:
                            ref_val += corr(list(range(n)), cl)
                        else:
                            ref_val += (corr([0, 1, 2, 3], cl) - corr([0, 1], cl) * corr([2, 3], cl) - corr([0, 2], cl) * corr([1, 3], cl) - corr([0, 3], cl) * corr([1, 2], cl))
                    worst, scale = max(worst, abs(got - ref_val)), max(scale, abs(ref_val))
        print(f"   {label}: max |diagram - Fock| = {worst:.1e} (largest value {scale:.1e})  {verdict(worst < 1e-10 * max(scale, 1e-3) + 1e-12)}")

    print("3. series (orders 2-4)  vs  the brute-force Fock-space Rayleigh-Schroedinger series (kagome_pt_checks)")
    cases = [("triangle, random K (dense path, Laplace quadrature)", tri, kp.random_reference(3, tri[1], np.random.default_rng(5), 0.8), 3, 1e-6),
             ("K4, two dimers (flat band, exact quadrature, factorised)", k4, dimer_K(4, [(0, 1), (2, 3)], lambda i, j: 1.0), 3 if quick else 4, 1e-6),
             ("K4, two dimers with unequal couplings (Laplace quadrature, factorised)", k4, dimer_K(4, [(0, 3), (1, 2)], lambda i, j: {(0, 3): 0.8, (1, 2): 1.2}[(i, j)]), 3, 1e-5)]
    if not quick:
        cases += [("bowtie, random K (dense path)", bow, kp.random_reference(5, bow[1], np.random.default_rng(6), 0.7), 3, 1e-5)]
    for label, (N, bonds, w), K, nmax, tol in cases:
        t0 = time.time()
        ref = GraphReference(K)
        res = series(ref, bonds, w, nmax, None, tol)
        fock = kp.fock_series(N, bonds, w, K, nmax)["E"]
        diffs = [abs(res["E"][n] - fock[n]) for n in range(2, nmax + 1)]
        scale = max(abs(fock[n]) for n in range(2, nmax + 1))
        print(f"   {label}: " + "  ".join(f"E_{n} = {res['E'][n]:+.8f} (Fock {fock[n]:+.8f})" for n in range(2, nmax + 1)) +
              f"   max|diff| = {max(diffs):.1e}  {verdict(max(diffs) < 20 * tol * scale + 1e-9)}  ({time.time() - t0:.0f} s)")

    print("4. exact reductions: pruning of disconnected configurations, first-vertex symmetry orbits, exact flat quadrature vs Laplace quadrature, dense vs factorised tables")
    T = kl.torus_from_name("2x1")
    N, bonds, w, perms = T.N, T.bonds, T.couplings(), T.all_symmetries()
    cov = sample_coverings(N, bonds, perms, 1, 0)[0]
    ref = covering_reference(N, bonds, w, cov)
    base = series(ref, bonds, w, 3, None, 1e-8, prune=False)["E"]
    for label, kw, perm in [("pruning of disconnected configurations", dict(), None), ("first-vertex symmetry orbits", dict(prune=False), perms)]:
        e = series(ref, bonds, w, 3, perm, 1e-8, **kw)["E"]
        dE = max(abs(e[n] - base[n]) for n in (2, 3))
        print(f"   kagome 2x1 torus (N = {N}, merged bonds, covering {list(cov)}, {'flat' if ref.flat else 'dispersive'} reference): {label} on: max|E_n - E_n(plain sum over all configurations)| = {dE:.1e}  {verdict(dE < 1e-9)}")
    refd = GraphReference(dimer_K(4, [(0, 1), (2, 3)], lambda i, j: 1.0))
    e_x = series(refd, k4[1], k4[2], 3, None, 1e-9, quad="flat")["E"]
    e_l = series(refd, k4[1], k4[2], 3, None, 1e-9, quad="laplace")["E"]
    dE = max(abs(e_x[n] - e_l[n]) for n in (2, 3))
    print(f"   K4 two dimers: exact flat-band quadrature vs Laplace quadrature (tol 1e-9): max|diff| = {dE:.1e}  {verdict(dE < 1e-7)}")
    e_c = series(refd, k4[1], k4[2], 3, None, 1e-6, collapse=False)["E"]
    dE = max(abs(e_x[n] - e_c[n]) for n in (2, 3))
    print(f"   K4 two dimers: collapsing identical integrands vs evaluating every configuration: max|diff| = {dE:.1e}  {verdict(dE < 1e-12)}")
    refd = GraphReference(dimer_K(4, [(0, 1), (2, 3)], lambda i, j: 1.0))
    e_f = series(refd, k4[1], k4[2], 4, None, 1e-6, dense=False)["E"]
    e_d = series(refd, k4[1], k4[2], 3, None, 1e-6, dense=True)["E"]
    dE = max(abs(e_f[n] - e_d[n]) for n in (2, 3))
    print(f"   K4 two dimers: dense 16-leg path vs factorised: max|diff| = {dE:.1e}  {verdict(dE < 1e-10)}")

    print("5. regression against the chain engine heisenberg_diagrams.py (same reference K, ring N = 6, J2 = 0 and 0.3; translation orbits on)")
    for J2 in (0.0, 0.3):
        N = 6
        bonds, w = kg.ring_graph(N, J2)
        ref_old = hd.Reference(N, twist=None, J2=J2)
        K = np.zeros((N, N))
        for i in range(N):
            K[i, i] += ref_old.mu
            for r, (t, ka) in enumerate(((ref_old.tR, ref_old.kappa), (ref_old.tR2, ref_old.kappa2)), start=1):
                j = (i + r) % N
                s = ref_old.seam if i + r >= N else 1.0
                K[i, j] += s * (t + ka)
                K[j, i] += s * (t - ka)
        ref = GraphReference(K)
        perms = [(np.arange(N) + s) % N for s in range(N)]
        new = series(ref, bonds, w, 3, perms, 1e-6)["E"]                          # the TI reference has a single class: dense path, order 3 at most
        old = hd.series(ref_old, 3, 1e-6, None, J2=J2)["E"]
        dE = max(abs(new[n] - old[n]) for n in new)
        print(f"   J2 = {J2:g}: " + "  ".join(f"E_{n}: {new[n]:+.8f} / {old[n]:+.8f}" for n in new) + f"   max|new - old| = {dE:.1e}  {verdict(dE < 2e-5)}")
    print("6. Majumdar-Ghosh point: at J2 = J1/2 the dimer product is the exact ground state, so every E_n with n >= 2 must vanish (ring, dimer reference, J2 bonds included)")
    for N in ((6,) if quick else (6, 8)):
        bonds, w = kg.ring_graph(N, 0.5)
        ref = GraphReference(dimer_K(N, [(2 * i, 2 * i + 1) for i in range(N // 2)], lambda i, j: 1.0))
        perms = [(np.arange(N) + s) % N for s in range(N)] + [(-np.arange(N) + s) % N for s in range(N)]
        res = series(ref, bonds, w, 3 if quick else 4, perms, 1e-6)
        worst = max(abs(res["E"][n]) for n in res["E"] if n >= 2)
        print(f"   ring N = {N}, J2 = 0.5: <H>_0 = {res['E'][1]:.10f} (-3N/8 = {-3 * N / 8:.10f});  max_n>=2 |E_n| = {worst:.1e}  {verdict(worst < 1e-9 and abs(res['E'][1] + 3 * N / 8) < 1e-9)}")
    print("7. dimer coverings: number of inequivalent coverings against Burnside's lemma, and the ordering of the covering table (lowest E_1 + E_2 first, ties by E_1 + E_2 + E_3)")
    for name in (("12",) if quick else ("12", "18a", "18b")):
        T = kl.torus_from_name(name)
        N, bonds, w, perms = T.N, T.bonds, T.couplings(), T.all_symmetries()
        sets = [frozenset(tuple(sorted(p)) for p in m) for m in enumerate_matchings(N, bonds)]
        fixed = sum(1 for g in perms for s in sets if frozenset(tuple(sorted((int(g[a]), int(g[b])))) for a, b in s) == s)
        n_burnside, n_cov = fixed / len(perms), len(all_coverings(N, bonds, perms))
        rows, _ = covering_table(N, bonds, w, perms, ncover=2)
        e12 = [r["P"][0] + r["P"][1] for r in rows]
        top = [r for r, e in zip(rows, e12) if e <= e12[0] + 1e-9]
        e123 = [r["P"][0] + 2 * r["P"][1] + r["P"][2] for r in top]
        ok = (abs(n_burnside - n_cov) < 1e-9 and len(rows) == n_cov and all(e12[k] <= e12[k + 1] + 1e-9 for k in range(len(rows) - 1))
              and all(x is not None for x in [r["P"][2] for r in top]) and all(e123[0] <= e + 1e-9 for e in e123))
        print(f"   kagome {name} (N = {N}, {len(sets)} perfect matchings, {len(perms)} symmetries): {n_cov} inequivalent coverings (Burnside {n_burnside:.0f}); {len(top)} tie for the lowest E_1 + E_2 = {e12[0]:.8f}, "
              f"all have E_3;  rows[0] has the lowest E_1 + E_2 + E_3 = {e123[0]:.8f} among them  {verdict(ok)}")
    print("8. rescaling of the reference Hamiltonian H0 -> s H0: partial sums from rescaled_partial_sums(P, 1/s) of the s = 1 diagram series  vs  the Fock-space series built on s H0")
    for label, (N, bonds, w), K in [("K4, two dimers", k4, dimer_K(4, [(0, 1), (2, 3)], lambda i, j: 1.0)), ("K4, random K", k4, kp.random_reference(4, k4[1], np.random.default_rng(7), 0.7))]:
        d = series(GraphReference(K), bonds, w, 4 if (not quick and 'dimers' in label) else 3, None, 1e-7)     # order 4 only for the factorised (dimer) reference
        for s_ in (0.6, 1.7):
            S = rescaled_partial_sums(d["P"], 1.0 / s_)
            E = kp.fock_series(N, bonds, w, s_ * K, len(S))["E"]
            worst = max(abs(S[n - 1] - sum(E[:n + 1])) for n in range(2, len(S) + 1))
            print(f"   {label}, s = {s_}: S_2..S_{len(S)} = " + " ".join(f"{x_:+.8f}" for x_ in S[1:]) + f"   max|diagram - Fock| = {worst:.1e}  {verdict(worst < 1e-6)}")
    print("\nALL CHECKS PASSED" if bad == 0 else f"\n{bad} CHECK(S) FAILED")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("kagome", help="orders 1-nmax on a kagome torus around dimer-covering references")
    p.add_argument("name", help="cluster name (12, 18a, 18b, 24, 27a, 27b, 30, 36a-36d) or LxM")
    p.add_argument("--order", type=int, default=3, choices=[2, 3, 4])
    p.add_argument("--coverings", type=int, default=24, help="how many of the best coverings (by E_1 + E_2) also get their third-order energy (all coverings that tie for the lowest E_1 + E_2 get it anyway)")
    p.add_argument("--keep", type=int, default=1, help="how many of the best coverings are carried to the full order")
    p.add_argument("--tol", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-ed", action="store_true", help="skip exact diagonalisation (N <= 24 only)")
    p.add_argument("--verbose", action="store_true")
    q = sub.add_parser("ring", help="the J1-J2 ring around its lowest Gaussian state (a dimer covering)")
    q.add_argument("N", nargs="+", type=int)
    q.add_argument("--J2", type=float, nargs="+", default=[0.0])
    q.add_argument("--order", type=int, default=3, choices=[2, 3, 4])
    q.add_argument("--tol", type=float, default=1e-6)
    q.add_argument("--verbose", action="store_true")
    s = sub.add_parser("selftest", help="validation against the Fock-space series and the chain engine")
    s.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(selftest(a.quick))
    if a.cmd == "ring":
        run_ring(a.N, a.order, a.J2, a.tol, a.verbose)
    else:
        run_kagome(a.name, a.order, a.coverings, a.tol, a.seed, not a.no_ed, a.verbose, a.keep)


if __name__ == "__main__":
    main()
