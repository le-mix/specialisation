#!/usr/bin/env python3
"""
Diagrammatic (Wick / Pfaffian) perturbation theory around the TZ Gaussian state of the 6-Majorana representation of the spin-1/2
Heisenberg ring     H = sum_{j=1}^{N} S_j . S_{j+1}     (J = 1, periodic, N sites),     orders 2, 3 and 4.

This file contains the 6-Majorana TZ computation only.  Its reference state is the one found by heisenberg_gaussian_S1.py, and every number
can be checked against the brute-force Fock-space series of heisenberg_pt_checks.py (N <= 6) and against exact diagonalisation.

--------------------------------------------------------------------------------------------------------------------------------------
1. Reference.  Each site carries six Majoranas (rho^a, eta^a), a = x, y, z (colours), (rho^a)^2 = (eta^a)^2 = 1, so the local space has dimension 8.
   H0 = sum_a sum_i [ mu i rho^a_i eta^a_i + (tR+kappa) i rho^a_i eta^a_{i+1} + (tR-kappa) i rho^a_{i+1} eta^a_i ]   (seam x (-1)^twist),
   is the Gaussian reference (the three colours are independent copies).  chi = (rho_1..rho_N, eta_1..eta_N) has covariance
   C = <chi chi^T> = 2 U U^dagger  and imaginary-time propagator  G(tau) = <chi(tau) chi(0)> = 2 U e^{-lambda tau} U^dagger  (tau > 0),
   U = the N positive-energy eigenvectors of the 2N x 2N matrix h = [[0, 2iK], [-2iK^T, 0]], lambda_m > 0 their quasi-particle energies.

2. Vertex.  S^a = -(i/2) [ Pi^a_+ rho^b rho^c + Pi^a_- eta^b eta^c ],   Pi^a_pm = (1 pm i rho^a eta^a)/2,   (a, b, c) cyclic, so that
       v_{j,a} = S^a_j S^a_{j+1} = (1/4) sum_{s,s'=rho,eta} Y^a_{s s'} B^b_{s s'} B^c_{s s'},
       Y^a_{s s'} = Pi_s(j) Pi_s'(j+1) (acts on colour a),   B^g_{s s'} = chi^{g,s}_j chi^{g,s'}_{j+1} (colour g = b, c);
   the species (s, s') is shared by the three colour factors.  H = sum_{j,a} v_{j,a}.  Every vertex has an even number of legs in every colour.

3. Series.  H = H0 + V.  E(xi) = lowest eigenvalue of H0 + xi V = sum_n E_n xi^n.  E_0 + E_1 = <H> (mean-field energy).
   With H(tau) = e^{H0 tau} H e^{-H0 tau} and the joint (connected) cumulants kappa_n of the time-ordered correlators, define
       P_n = -(-1)^n  int_{t_1 > ... > t_n = 0}  kappa_n( H(t_1), ..., H(t_n) ) dt_1 .. dt_{n-1}          (P_1 = <H>)
   with the PLAIN bond operators as vertices (no normal ordering, no counter-term).  Because H = H0 + V, the generating function obeys
       -(1/beta) ln < T exp(-xi int H(tau) dtau) > = (1 + xi) E( xi/(1 + xi) ) - E_0 ,
   i.e. the insertions of H0 at interior times are total time derivatives, and comparing powers of xi gives the exact relation
       E_n = sum_{j=0}^{n-2} binomial(n-2, j) P_{n-j}          (E_2 = P_2, E_3 = P_3 + P_2, E_4 = P_4 + 2 P_3 + P_2).
   The cumulants are formed from the mean-removed vertices v'' = v - <v>:  kappa_2 = <12>,  kappa_3 = <123>,
   kappa_4 = <1234> - <12><34> - <13><24> - <14><23>.   The first vertex is fixed on bond 0 with colour x (result x 3N by translation and
   colour rotation); the other n-1 vertices run over all N bonds and 3 colours.

4. Wick.  Each colour is a Gaussian state, so for fixed vertex species the correlator factorises into three Pfaffians (one per colour).  For one
   colour and one vertex the four legs are (rho_L, eta_L, rho_R, eta_R) and the operator content is one of the 8 even monomials
   (1, e01, e02, e03, e12, e13, e23, e0123).  All 8^n label combinations Phi[m_1..m_n] = Pf(contraction matrix of the chosen legs) are
   evaluated once per time node and bond positions (the contraction matrix has G(t_p - t_q) between vertices p < q and the equal-time C
   inside a vertex); the vertex algebra (Y and B vectors, identity, species sums, colour assignments) is then a contraction of this table.
   For even N with mu = 0 the Majoranas split into two decoupled sublattices, E = {rho_even, eta_odd} and O = {rho_odd, eta_even}, that never
   correlate, so every Pfaffian factorises, Pf = (sign) Pf_E Pf_O, with two 8-leg Pfaffians instead of one 16-leg one ("E/O path", the
   production path).  Odd N (mu != 0) uses the dense 16-leg tables ("dense path": validated, but far slower).

5. Time integrals.  Cut energies Omega (sums of quasi-particle energies of the lines crossing a cut) satisfy
   2 lambda_min <= Omega <= min(16 lambda_max, 3 sum lambda) for connected terms, so each interval integral int_0^inf e^{-Omega tau} dtau = 1/Omega
   is replaced by sum_k w_k e^{-alpha_k Omega} with positive, bounded weights (laplace_nodes, relative accuracy tol on that interval).  The
   integrand is I(tau_1, .., tau_{n-1}) with the time-reversal property I(tau_1, .., tau_{n-1})^* = I(tau_{n-1}, .., tau_1), used to halve the grid.

Checks (python3 heisenberg_diagrams.py selftest): first order from the vertex algebra, orders 2-4 vs the Fock-space series, dense vs E/O.
Requires numpy, scipy and heisenberg_gaussian_S1.py (and heisenberg_pt_checks.py for the checks) in the same directory.

Usage:  python3 heisenberg_diagrams.py ring 4 6 8                 # orders 1-4 for each N, lower-mean-field seam sign, vs exact diagonalisation
        python3 heisenberg_diagrams.py ring 8 --twist both --tol 1e-6 --raw
        python3 heisenberg_diagrams.py ring 5 --dense --tol 1e-4  # odd N: dense path (slow)
        python3 heisenberg_diagrams.py selftest [--quick]
"""
import argparse
import itertools
import math
import os
import sys
import time

import numpy as np
from scipy.optimize import nnls

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heisenberg_gaussian_S1 as hg


# ================================================================================================ 1. TZ reference
class Reference:
    """TZ Gaussian reference state of the N-site ring (seam sign (-1)^twist).  twist=None picks the seam sign with the lower mean-field energy."""

    def __init__(self, N, twist=None, starts=24, seed=0):
        self.N = N
        if twist is None:
            outs = [hg.minimise(N, t, starts=starts, seed=seed) for t in (0, 1)]
            twist = 0 if outs[0]["E"] <= outs[1]["E"] + 1e-12 else 1
            out = outs[twist]
        else:
            out = hg.minimise(N, twist, starts=starts, seed=seed)
        self.twist, self.seam = twist, (1.0 if twist == 0 else -1.0)
        self.mu, self.tR, self.kappa, self.E_min = out["mu"], out["tR"], out["kappa"], out["E"]
        if N % 2 == 0 and abs(self.mu) < 1e-7:
            self.mu = 0.0                                  # optimum has <i rho eta> = 0 on a site: make the sublattice structure exact
        K = np.zeros((N, N))
        for i in range(N):
            j = (i + 1) % N
            s = self.seam if j == 0 else 1.0
            K[i, i] += self.mu
            K[i, j] += s * (self.tR + self.kappa)
            K[j, i] += s * (self.tR - self.kappa)
        h = np.block([[np.zeros((N, N)), 2j * K], [-2j * K.T, np.zeros((N, N))]])
        lam, U = np.linalg.eigh(h)
        self.gap = np.abs(lam).min()
        if self.gap < 1e-9:
            raise ValueError(f"the TZ reference for N={N}, twist={twist} has zero modes (degenerate reference): perturbation theory is undefined")
        pos = lam > 0
        assert pos.sum() == N
        self.lam, self.U = lam[pos], U[:, pos]
        self.C = 2 * self.U @ self.U.conj().T              # <chi_r chi_c>, row index = species * N + site
        idx = np.arange(2 * N)
        spec, site = idx // N, idx % N
        in_E = ((spec == 0) & (site % 2 == 0)) | ((spec == 1) & (site % 2 == 1))      # E = {rho_even, eta_odd}, O = {rho_odd, eta_even}
        self.bipartite = bool(N % 2 == 0 and np.abs(self.C[np.ix_(in_E, ~in_E)]).max() < 1e-9)

    def G(self, taus):
        """G(tau) = <chi(tau) chi(0)> for an array of tau > 0: shape (len(taus), 2N, 2N)."""
        E = np.exp(-np.outer(np.asarray(taus, float).ravel(), self.lam))
        return 2 * ((self.U[None] * E[:, None, :]) @ self.U.conj().T)


# ================================================================================================ 2. Pfaffians of all principal submatrices
_PLANS = {}


def _pf_plan(nl):
    if nl in _PLANS:
        return _PLANS[nl]
    pcnt = np.array([bin(m).count("1") for m in range(1 << nl)])
    layers = []
    for size in range(2, nl + 1, 2):
        Sl, starts, ii, jj, sp, sg = [], [], [], [], [], []
        for S in np.nonzero(pcnt == size)[0]:
            S = int(S)
            bits = [b for b in range(nl) if (S >> b) & 1]
            starts.append(len(ii))
            Sl.append(S)
            b0 = bits[0]
            for k in range(1, size):                          # expansion along the first row: Pf(S) = sum_k (-1)^(k-1) A[b0,bk] Pf(S - {b0,bk})
                bk = bits[k]
                ii.append(b0)
                jj.append(bk)
                sp.append(S ^ (1 << b0) ^ (1 << bk))
                sg.append(1.0 if (k - 1) % 2 == 0 else -1.0)
        layers.append(tuple(np.array(x) for x in (Sl, starts, ii, jj, sp, sg)))
    _PLANS[nl] = layers
    return layers


def pf_all(A):
    """A: (g, nl, nl) complex antisymmetric.  Pfaffians of ALL principal submatrices, indexed by subset bitmask: shape (g, 2**nl) (odd subsets: 0)."""
    g, nl, _ = A.shape
    P = np.zeros((g, 1 << nl), complex)
    P[:, 0] = 1.0
    for Sl, starts, ii, jj, sp, sg in _pf_plan(nl):
        P[:, Sl] = np.add.reduceat(A[:, ii, jj] * P[:, sp] * sg, starts, axis=1)
    return P


# ================================================================================================ 3. vertex algebra: labels and vertex vectors
# legs of one colour on a bond (j, j+1):  0 = rho_L, 1 = eta_L, 2 = rho_R, 3 = eta_R;  the 8 even monomials ("labels"):
LABELS = [(), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (0, 1, 2, 3)]
LAB_INDEX = {m: i for i, m in enumerate(LABELS)}
LAB_MASK = np.array([sum(1 << l for l in m) for m in LABELS])
ID_VEC = 8


def _vertex_vectors():
    """V[label, nu] with nu = 0..3: Y_{s s'}  (index 2 s + s', s = 0 rho / 1 eta; Pi_s Pi_s' = (1/4)[1 + sigma i e01 + sigma' i e23 - sigma sigma' e0123]),
       nu = 4..7: B_{s s'} = chi^s_L chi^s'_R,   nu = 8: the identity."""
    V = np.zeros((8, 9), complex)
    for s0, s1 in itertools.product((0, 1), repeat=2):
        sp_ = 2 * s0 + s1
        a0, a1 = (1.0 if s0 == 0 else -1.0), (1.0 if s1 == 0 else -1.0)
        V[LAB_INDEX[()], sp_] = 0.25
        V[LAB_INDEX[(0, 1)], sp_] = 0.25 * a0 * 1j
        V[LAB_INDEX[(2, 3)], sp_] = 0.25 * a1 * 1j
        V[LAB_INDEX[(0, 1, 2, 3)], sp_] = 0.25 * (-a0 * a1)
        V[LAB_INDEX[(s0, 2 + s1)], 4 + sp_] = 1.0
    V[LAB_INDEX[()], ID_VEC] = 1.0
    return V


VV = _vertex_vectors()


def contract_vertices(Phi, n):
    """Psi[g, nu_1..nu_n] = sum_m Phi[g, m_1..m_n] prod_p V[m_p, nu_p]   (Phi: (g, 8**n) -> Psi: (g, 9**n))."""
    g = Phi.shape[0]
    T = Phi.reshape(g, *(8,) * n)
    for p in range(n - 1, -1, -1):
        T = np.moveaxis(np.tensordot(T, VV, axes=([p + 1], [0])), -1, p + 1)
    return T.reshape(g, 9 ** n)


def _label_tuples(n):
    return np.array(list(itertools.product(range(8), repeat=n)))


_DM = {}


def dense_masks(n):
    if n not in _DM:
        lt = _label_tuples(n)
        _DM[n] = sum(LAB_MASK[lt[:, p]] << (4 * p) for p in range(n))
    return _DM[n]


_EO = {}


def eo_maps(n, par):
    """Even/odd split of the label tables for vertex parities par (par_p = bond position x_p mod 2).
       Legs of a bond starting on an EVEN site: E = (rho_L, eta_R) = legs (0, 3), O = (eta_L, rho_R) = legs (1, 2); on an ODD site the two sets are swapped.
       Returns (iE, iO, sign): for each label tuple (8^n of them) the E-subset bitmask, the O-subset bitmask (bit 2p+loc = loc-th leg of the set at vertex p)
       and the sign of the permutation that sorts the global leg sequence into (all E legs, all O legs): Pf = sign * Pf_E * Pf_O."""
    key = (n, tuple(par))
    if key in _EO:
        return _EO[key]
    legmap = {0: {0: ("E", 0), 1: ("O", 0), 2: ("O", 1), 3: ("E", 1)},
              1: {0: ("O", 0), 1: ("E", 0), 2: ("E", 1), 3: ("O", 1)}}
    lt = _label_tuples(n)
    iE = np.zeros(len(lt), int)
    iO = np.zeros(len(lt), int)
    sg = np.ones(len(lt))
    for r, tup in enumerate(lt):
        seq = []
        for p in range(n):
            for leg in LABELS[tup[p]]:
                s, loc = legmap[par[p]][leg]
                seq.append(s)
                if s == "E":
                    iE[r] |= 1 << (2 * p + loc)
                else:
                    iO[r] |= 1 << (2 * p + loc)
        inv = sum(1 for a in range(len(seq)) if seq[a] == "O" for b in range(a + 1, len(seq)) if seq[b] == "E")
        sg[r] = -1.0 if inv % 2 else 1.0
    _EO[key] = (iE, iO, sg)
    return _EO[key]


ELEG = {0: (0, 3), 1: (1, 2)}          # legs of a bond vertex that belong to set E, for a bond starting on an even / odd site
OLEG = {0: (1, 2), 1: (0, 3)}
ALLEG = {0: (0, 1, 2, 3), 1: (0, 1, 2, 3)}


def leg_rows(N, x):
    """Rows (species * N + site) of the four legs (rho_L, eta_L, rho_R, eta_R) of the bond (x, x+1 mod N)."""
    xl, xr = x % N, (x + 1) % N
    return np.array([xl, N + xl, xr, N + xr])


def build_A(ref, positions, Gp, legsel):
    """Antisymmetric contraction matrix (g, L n, L n) of the selected legs of the n vertices at the bonds `positions`;  Gp[(p, q)] = G(t_p - t_q) (p < q)."""
    N, n = ref.N, len(positions)
    g = next(iter(Gp.values())).shape[0]
    L = len(legsel[0])
    A = np.zeros((g, L * n, L * n), complex)
    rows = [leg_rows(N, x)[list(legsel[x % 2])] for x in positions]
    for p in range(n):
        Cu = np.triu(ref.C[np.ix_(rows[p], rows[p])], 1)                         # equal-time contractions inside the vertex
        A[:, L * p:L * p + L, L * p:L * p + L] = (Cu - Cu.T)[None]
        for q in range(p + 1, n):
            blk = Gp[(p, q)][:, rows[p][:, None], rows[q][None, :]]
            A[:, L * p:L * p + L, L * q:L * q + L] = blk
            A[:, L * q:L * q + L, L * p:L * p + L] = -blk.transpose(0, 2, 1)
    return A


def phi_table(ref, positions, Gp, dense):
    """Phi[g, 8^n] = Pfaffian of the legs selected by each label tuple."""
    n = len(positions)
    if dense:
        return pf_all(build_A(ref, positions, Gp, ALLEG))[:, dense_masks(n)]
    iE, iO, sg = eo_maps(n, tuple(x % 2 for x in positions))
    PE = pf_all(build_A(ref, positions, Gp, ELEG))
    PO = pf_all(build_A(ref, positions, Gp, OLEG))
    return PE[:, iE] * PO[:, iO] * sg


# ================================================================================================ 4. colour assignments, species sums, cumulants
def _rgs(m):
    """Colour assignments of m vertices up to relabelling of the three colours (restricted growth strings; the first vertex has colour 0)."""
    out = []

    def rec(prefix, mx):
        if len(prefix) == m:
            out.append(tuple(prefix))
            return
        for v in range(min(mx + 2, 3)):
            rec(prefix + [v], max(mx, v))
    rec([0], 0)
    return out


MULT_FIXED = {1: 1, 2: 2, 3: 2}        # number of colour assignments of a class with b colour blocks, first vertex colour fixed
MULT_FREE = {1: 3, 2: 6, 3: 6}         # ... first vertex colour free


class ColourSums:
    """sum over colour assignments a_p and vertex kinds k_p (0..3 = species with weight 1/4, 4 = the shift -cbar of the mean removal) of
       prod_{colour g} Psi[nu^g],  nu^g_p = k_p if a_p = g else 4 + k_p  (identity index 8 for the shift)."""

    def __init__(self, cbar):
        self.cbar, self.cache = cbar, {}

    def tables(self, m, fixed):
        key = (m, fixed)
        if key not in self.cache:
            classes = _rgs(m)
            kinds = list(itertools.product(range(5), repeat=m))
            kidx = np.zeros((3, len(classes), len(kinds)), int)
            for ic, a in enumerate(classes):
                for ik, k in enumerate(kinds):
                    for g in range(3):
                        f = 0
                        for p in range(m):
                            nu = ID_VEC if k[p] == 4 else (k[p] if a[p] == g else 4 + k[p])
                            f = f * 9 + nu
                        kidx[g, ic, ik] = f
            wk = np.array([[(-self.cbar if kp == 4 else 0.25) for kp in k] for k in kinds]).prod(axis=1)
            mult = np.array([(MULT_FIXED if fixed else MULT_FREE)[max(a) + 1] for a in classes], float)
            self.cache[key] = ((mult[:, None] * wk[None, :]).ravel(), tuple(kidx[g].ravel() for g in range(3)))
        return self.cache[key]

    def __call__(self, Psi, m, fixed):
        W, (i0, i1, i2) = self.tables(m, fixed)
        return (Psi[:, i0] * Psi[:, i1] * Psi[:, i2]) @ W


def _pair_slice(Psi, n, p, q):
    idx = [slice(None)] + [slice(None) if i in (p, q) else ID_VEC for i in range(n)]
    sub = Psi.reshape((Psi.shape[0],) + (9,) * n)[tuple(idx)]
    return sub.reshape(sub.shape[0], 81)


def vertex_mean(ref):
    """c = <v_{j,a}> (one bond, one colour): the mean-field bond energy per colour; <H> = 3 N c."""
    N = ref.N
    idx = [0, N, 1, N + 1]                                    # rho_0, eta_0, rho_1, eta_1
    A = np.zeros((1, 4, 4), complex)
    for u in range(4):
        for v in range(u + 1, 4):
            A[0, u, v] = ref.C[idx[u], idx[v]]
            A[0, v, u] = -A[0, u, v]
    Phi = pf_all(A)[:, LAB_MASK]
    Psi = contract_vertices(Phi, 1)[0]
    return float((0.25 * sum(Psi[s] * Psi[4 + s] ** 2 for s in range(4))).real)


def first_order(ref):
    """E_0 + E_1 = <H>_0 = 3 N <v> from the vertex algebra."""
    return 3 * ref.N * vertex_mean(ref)


def integrand(ref, positions, Gp, cs, dense):
    """Colour-summed connected correlator kappa_n of the mean-removed vertices at the bonds `positions` (the first with colour x), for each time node."""
    n = len(positions)
    Psi = contract_vertices(phi_table(ref, positions, Gp, dense), n)
    if n <= 3:
        return cs(Psi, n, True)
    pair = lambda p, q: cs(_pair_slice(Psi, 4, p, q), 2, p == 0)
    return cs(Psi, 4, True) - pair(0, 1) * pair(2, 3) - pair(0, 2) * pair(1, 3) - pair(0, 3) * pair(1, 2)


# ================================================================================================ 5. time quadrature
def laplace_nodes(Omin, Omax, tol, wmax=25.0):
    """Positive nodes alpha_k and weights w_k with |x sum_k w_k exp(-alpha_k x) - 1| <= tol for all x = Omega/Omin in [1, Omax/Omin], so that
       int_0^inf dtau exp(-Omega tau) = 1/Omega  ~  sum_k w_k exp(-alpha_k Omega).  The weights are kept bounded (max w <= wmax/Omin): the cumulants
       cancel disconnected pieces only up to rounding noise, which unbounded weights would amplify."""
    R = max(Omax / Omin, 1.0 + 1e-9)
    x = np.exp(np.linspace(0.0, np.log(R), 3000))
    amax = np.log(1.0 / tol) + 6.0
    for hc in np.arange(1.0, 0.099, -0.05):
        for off in (0.0, 0.25, 0.5, 0.75):
            a = np.exp(np.arange(np.log(0.3 / R) + off * hc, np.log(amax) + 1e-12, hc))
            M = np.exp(-np.outer(x, a)) * x[:, None]
            w, _ = nnls(M, np.ones_like(x), maxiter=200000)
            keep = w > 0
            err = float(np.max(np.abs(M[:, keep] @ w[keep] - 1.0)))
            if err <= tol and w.max() <= wmax:
                return a[keep] / Omin, w[keep] / Omin, err
    raise RuntimeError("no well-conditioned quadrature found")


# ================================================================================================ 6. the n-th order sum
class Diagrams:
    """P_n = -(-1)^n int kappa_n(H(t_1) .. H(t_n)) for the TZ reference `ref`, n = 2, 3, 4 (plain bond vertices)."""

    def __init__(self, ref, n, tol=1e-6, dense=None, chunk=128, symmetric=True, verbose=False):
        if n not in (2, 3, 4):
            raise ValueError("order n must be 2, 3 or 4")
        self.ref, self.n, self.chunk, self.verbose = ref, n, chunk, verbose
        self.dense = (not ref.bipartite) if dense is None else dense
        if not self.dense and not ref.bipartite:
            raise ValueError("the E/O path needs even N and mu = 0; use dense=True (slow)")
        lam = ref.lam
        Omin = 2 * lam.min()
        Omax = min((16 if n == 4 else 8) * lam.max(), 3 * lam.sum())
        self.alpha, self.w, self.err = laplace_nodes(Omin, max(Omax, 1.01 * Omin), tol)
        self.K = len(self.alpha)
        self.cs = ColourSums(vertex_mean(ref))
        nodes = np.array(list(itertools.product(range(self.K), repeat=n - 1)))
        mult = np.ones(len(nodes))
        if symmetric and n >= 3:                                         # I(tau_1..tau_{n-1})^* = I(tau_{n-1}..tau_1)
            lo, hi = nodes[:, 0], nodes[:, -1]
            keep = lo <= hi
            mult = np.where(lo < hi, 2.0, 1.0)[keep]
            nodes = nodes[keep]
        self.nodes, self.wts = nodes, np.prod(self.w[nodes], axis=1) * mult

    def energy(self):
        ref, n, N = self.ref, self.n, self.ref.N
        total, t0 = 0.0, time.time()
        positions = list(itertools.product(range(N), repeat=n - 1))
        nch = -(-len(self.nodes) // self.chunk)
        for c in range(nch):
            sl = slice(c * self.chunk, (c + 1) * self.chunk)
            nd, wch = self.nodes[sl], self.wts[sl]
            Gp = {(p, q): ref.G(self.alpha[nd[:, p:q]].sum(axis=1)) for p in range(n) for q in range(p + 1, n)}     # t_p - t_q = tau_p + .. + tau_{q-1}
            for xs in positions:
                total += (wch * integrand(ref, (0,) + xs, Gp, self.cs, self.dense)).sum()
            if self.verbose:
                print(f"      n={n}: chunk {c + 1}/{nch}  {time.time() - t0:.0f} s", flush=True)
        return float((-(-1) ** n * 3 * N * total).real)


def series(ref, nmax=4, tol=1e-6, dense=None, chunk=128, symmetric=True, verbose=False):
    """Orders 1..nmax: dict with the plain-cumulant energies P_n, the energies E_n of the Rayleigh-Schroedinger series, and the quadrature sizes."""
    P = {1: first_order(ref).real}
    K = {}
    for n in range(2, nmax + 1):
        d = Diagrams(ref, n, tol, dense, chunk, symmetric, verbose)
        P[n] = d.energy()
        K[n] = d.K
    E = {1: P[1]}
    for n in range(2, nmax + 1):
        E[n] = sum(math.comb(n - 2, j) * P[n - j] for j in range(n - 1))
    return dict(P=P, E=E, K=K)


# ================================================================================================ 7. drivers
def run_ring(Ns, nmax, tol, twists, dense, raw, chunk, symmetric, verbose):
    for N in Ns:
        Eex = hg.exact_energy(N)
        for tw in twists:
            t0 = time.time()
            ref = Reference(N, twist=tw)
            use_dense = dense or not ref.bipartite
            print(f"\nN = {N}, seam sign (-1)^twist, twist = {ref.twist}{' (lower mean-field energy)' if tw is None else ''};  "
                  f"mu = {ref.mu:.4g}, tR = {ref.tR:.5f}, kappa = {ref.kappa:.5f};  quasi-particle energies in [{ref.lam.min():.4f}, {ref.lam.max():.4f}]")
            print(f"   path: {'dense 16-leg Pfaffians' if use_dense else 'E/O-split 8-leg Pfaffians'};  exact ground-state energy (ED) = {Eex:.10f}")
            res = series(ref, nmax, tol, use_dense, chunk, symmetric, verbose)
            S = 0.0
            print(f"   {'order':>5} {'E_n':>14} {'partial sum':>14} {'from exact':>11} {'sum/N':>10}" + (f" {'P_n':>14}" if raw else ""))
            for n in range(1, nmax + 1):
                S += res["E"][n]
                label = f"{n}" + (" (MF)" if n == 1 else "")
                print(f"   {label:>5} {res['E'][n]:>+14.8f} {S:>14.8f} {100 * (S - Eex) / abs(Eex):>+10.2f}% {S / N:>10.6f}" + (f" {res['P'][n]:>+14.8f}" if raw else ""))
            print(f"   quadrature nodes per time interval: {res['K']}, relative tolerance {tol:g};  {time.time() - t0:.0f} s")
            sys.stdout.flush()


def selftest(quick=False):
    import heisenberg_pt_checks as pc
    bad = 0

    def verdict(ok):
        nonlocal bad
        bad += 0 if ok else 1
        return "ok" if ok else "FAILED"

    print("1. first order: <H>_0 = 3 N <v> from the vertex algebra  vs  the minimiser's energy")
    for N in (3, 4, 5, 6):
        for tw in (0, 1):
            ref = Reference(N, twist=tw)
            d = abs(first_order(ref).real - ref.E_min)
            print(f"   N={N} twist={tw}: {first_order(ref).real:+.10f}  (minimiser {ref.E_min:+.10f})  |diff| = {d:.1e}  {verdict(d < 1e-8)}")
    print("2. orders 2-4  vs  the brute-force Fock-space Rayleigh-Schroedinger series (heisenberg_pt_checks)")
    cases = [(4, 0, 1e-5, False), (4, 1, 1e-5, False), (3, 0, 1e-4, True)]
    for N, tw, tol, dense in cases:
        t0 = time.time()
        ref = Reference(N, twist=tw)
        nmax = 3 if (dense and quick) else 4
        res = series(ref, nmax, tol, dense)
        fock = pc.tz_series(N, tw, nmax)["E"]
        diffs = [abs(res["E"][n] - fock[n]) for n in range(2, nmax + 1)]
        scale = max(abs(fock[n]) for n in range(2, nmax + 1))
        print(f"   N={N} twist={tw} ({'dense' if dense else 'E/O'}, tol {tol:g}): " +
              "  ".join(f"E_{n} = {res['E'][n]:+.8f} (Fock {fock[n]:+.8f})" for n in range(2, nmax + 1)) +
              f"   max|diff| = {max(diffs):.1e}  {verdict(max(diffs) < 20 * tol * scale)}  ({time.time() - t0:.0f} s)")
    print("3. dense 16-leg Pfaffian tables  vs  E/O-split tables (N=4, random time nodes and bond positions)")
    ref = Reference(4, twist=1)
    d = Diagrams(ref, 4, 1e-4, dense=False)
    rng = np.random.default_rng(1)
    nd = d.nodes[rng.integers(0, len(d.nodes), 6)]
    Gp = {(p, q): ref.G(d.alpha[nd[:, p:q]].sum(axis=1)) for p in range(4) for q in range(p + 1, 4)}
    worst = 0.0
    for xs in [(1, 2, 3), (0, 0, 0), (2, 2, 1), (3, 1, 0)]:
        pos = (0,) + xs
        worst = max(worst, np.abs(phi_table(ref, pos, Gp, True) - phi_table(ref, pos, Gp, False)).max(),
                    np.abs(integrand(ref, pos, Gp, d.cs, True) - integrand(ref, pos, Gp, d.cs, False)).max())
    print(f"   max |dense - E/O| over tables and integrands = {worst:.1e}  {verdict(worst < 1e-12)}")
    print("\nALL CHECKS PASSED" if bad == 0 else f"\n{bad} CHECK(S) FAILED")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ring", help="orders 1-4 of the series for ring sizes N, compared with exact diagonalisation")
    p.add_argument("N", nargs="+", type=int)
    p.add_argument("--order", type=int, default=4, choices=[2, 3, 4], help="highest order (default 4)")
    p.add_argument("--twist", choices=["best", "0", "1", "both"], default="best", help="seam sign of the reference (default: lower mean-field energy)")
    p.add_argument("--tol", type=float, default=1e-6, help="relative accuracy of each time-interval quadrature (default 1e-6)")
    p.add_argument("--dense", action="store_true", help="force the dense 16-leg path (automatic for odd N; slow)")
    p.add_argument("--raw", action="store_true", help="also print the plain-cumulant energies P_n")
    p.add_argument("--chunk", type=int, default=128, help="time nodes per batch (memory/speed trade-off)")
    p.add_argument("--no-sym", action="store_true", help="do not use the time-reversal symmetry of the integrand (2x slower)")
    p.add_argument("--verbose", action="store_true")
    q = sub.add_parser("selftest", help="validation against the Fock-space series")
    q.add_argument("--quick", action="store_true", help="skip the slow dense order-4 check")
    a = ap.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(selftest(a.quick))
    twists = {"best": [None], "0": [0], "1": [1], "both": [0, 1]}[a.twist]
    run_ring(a.N, a.order, a.tol, twists, a.dense, a.raw, a.chunk, not a.no_sym, a.verbose)


if __name__ == "__main__":
    main()