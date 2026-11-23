#!/usr/bin/env python3
"""
Diagrammatic (Wick / Pfaffian) perturbation theory around the TZ Gaussian state of the 6-Majorana representation of the spin-1/2
J1-J2 Heisenberg ring
        H = sum_{j=1}^{N} S_j . S_{j+1}  +  J2 sum_{j=1}^{N} S_j . S_{j+2}          (J1 = 1, periodic, N sites),     orders 2, 3 and 4.
J2 = 0 is the nearest-neighbour ring of the previous version (same numbers).

This file contains the 6-Majorana TZ computation only.  Its reference state is the one found by heisenberg_gaussian_S1.py, and every number
can be checked against the brute-force Fock-space series of heisenberg_pt_checks.py (N <= 6) and against exact diagonalisation.

--------------------------------------------------------------------------------------------------------------------------------------
1. Reference.  Each site carries six Majoranas (rho^a, eta^a), a = x, y, z (colours), (rho^a)^2 = (eta^a)^2 = 1, so the local space has dimension 8.
   H0 = sum_a sum_i [ mu i rho^a_i eta^a_i + sum_{r=1,2} ( (t_r+kappa_r) i rho^a_i eta^a_{i+r} + (t_r-kappa_r) i rho^a_{i+r} eta^a_i ) ]
   (every bond that crosses the seam x (-1)^twist) is the Gaussian reference (the three colours are independent copies).  chi = (rho_1..rho_N,
   eta_1..eta_N) has covariance C = <chi chi^T> = 2 U U^dagger  and imaginary-time propagator  G(tau) = <chi(tau) chi(0)> = 2 U e^{-lambda tau} U^dagger
   (tau > 0),  U = the N positive-energy eigenvectors of the 2N x 2N matrix h = [[0, 2iK], [-2iK^T, 0]], lambda_m > 0 their quasi-particle energies.
   For J2 < J1 the translation-invariant optimum has t_2 = kappa_2 = 0 (no next-nearest-neighbour correlations): the reference is the
   nearest-neighbour one, <S_i . S_{i+2}> = 0 in it, and J2 enters only through V.  For J2 > J1 the optimum has t_1 = kappa_1 = 0: two decoupled
   next-nearest-neighbour chains (at finite N the switch can occur slightly below J2 = J1, depending on the parity of N/2).

2. Vertices.  S^a = -(i/2) [ Pi^a_+ rho^b rho^c + Pi^a_- eta^b eta^c ],   Pi^a_pm = (1 pm i rho^a eta^a)/2,   (a, b, c) cyclic.  A bond vertex joins a left
   site L = j and a right site R = j + r (distance r = 1 or 2; r is the "type" of the vertex), the formulas do not depend on r:
       v_{j,r,a} = S^a_j S^a_{j+r} = (1/4) sum_{s,s'=rho,eta} Y^a_{s s'} B^b_{s s'} B^c_{s s'},
       Y^a_{s s'} = Pi_s(L) Pi_s'(R) (acts on colour a),   B^g_{s s'} = chi^{g,s}_L chi^{g,s'}_R (colour g = b, c);
   the species (s, s') is shared by the three colour factors.  H = sum_{j,a} v_{j,1,a} + J2 sum_{j,a} v_{j,2,a}.  Every vertex has an even number of
   legs in every colour.

3. Series.  H = H0 + V.  E(xi) = lowest eigenvalue of H0 + xi V = sum_n E_n xi^n.  E_0 + E_1 = <H> (mean-field energy).
   With H(tau) = e^{H0 tau} H e^{-H0 tau} and the joint (connected) cumulants kappa_n of the time-ordered correlators, define
       P_n = -(-1)^n  int_{t_1 > ... > t_n = 0}  kappa_n( H(t_1), ..., H(t_n) ) dt_1 .. dt_{n-1}          (P_1 = <H>)
   with the PLAIN bond operators as vertices (no normal ordering, no counter-term).  Because H = H0 + V, the generating function obeys
       -(1/beta) ln < T exp(-xi int H(tau) dtau) > = (1 + xi) E( xi/(1 + xi) ) - E_0 ,
   i.e. the insertions of H0 at interior times are total time derivatives, and comparing powers of xi gives the exact relation
       E_n = sum_{j=0}^{n-2} binomial(n-2, j) P_{n-j}          (E_2 = P_2, E_3 = P_3 + P_2, E_4 = P_4 + 2 P_3 + P_2).
   The cumulants are formed from the mean-removed vertices v'' = v - <v>_r (one mean per type):  kappa_2 = <12>,  kappa_3 = <123>,
   kappa_4 = <1234> - <12><34> - <13><24> - <14><23>.   The first vertex is fixed on bond 0 (colour x) with all types (result x 3N by translation and
   colour rotation); the other n-1 vertices run over all N bonds, both types and 3 colours.
   J2 DEPENDENCE.  P_n is a homogeneous polynomial of degree n in (J1, J2): with J1 = 1,  P_n = sum_{m=0}^{n} T_{n,m} J2^m, where T_{n,m} is the sum of the
   diagrams with exactly m distance-2 vertices.  The program computes the T_{n,m} once for a given reference and evaluates every J2 that has the same
   reference (all J2 up to about 0.9 for even N; --poly prints the coefficients of E_n(J2)).

4. Wick.  Each colour is a Gaussian state, so for fixed vertex species the correlator factorises into three Pfaffians (one per colour).  For one
   colour and one vertex the four legs are (rho_L, eta_L, rho_R, eta_R) and the operator content is one of the 8 even monomials
   (1, e01, e02, e03, e12, e13, e23, e0123).  All 8^n label combinations Phi[m_1..m_n] = Pf(contraction matrix of the chosen legs) are
   evaluated once per time node and bond positions (the contraction matrix has G(t_p - t_q) between vertices p < q and the equal-time C
   inside a vertex); the vertex algebra (Y and B vectors, identity, species sums, colour assignments) is then a contraction of this table.
   DECOUPLED CLASSES.  If the Majoranas of the reference split into decoupled classes (connected components of h), the Pfaffian factorises into one
   Pfaffian per class, Pf = (sign) prod_c Pf_c, each class carrying only its own legs.  For even N and mu = 0 (all J2 < J1) the classes are
   E = {rho_even, eta_odd} and O = {rho_odd, eta_even}: two 8-leg Pfaffians instead of one 16-leg one (production path); for the decoupled
   next-nearest-neighbour chains of J2 > J1 the even and odd sites separate (and each chain splits again when N/2 is even: up to four classes).
   A reference with a single class (odd N, or mu != 0 coupling the two Majorana species on a site) uses the dense 16-leg tables ("dense path":
   validated, but about 30 times slower per configuration).
   COST.  A configuration is a choice of the vertex types and of the positions of vertices 2..n: 2^n N^(n-1) of them (N^(n-1) for J2 = 0).

5. Time integrals.  Cut energies Omega (sums of quasi-particle energies of the lines crossing a cut) satisfy
   2 lambda_min <= Omega <= min(16 lambda_max, 3 sum lambda) for connected terms, so each interval integral int_0^inf e^{-Omega tau} dtau = 1/Omega
   is replaced by sum_k w_k e^{-alpha_k Omega} with positive, bounded weights (laplace_nodes, relative accuracy tol on that interval).  The
   integrand is I(tau_1, .., tau_{n-1}) with the time-reversal property I(tau_1, .., tau_{n-1})^* = I(tau_{n-1}, .., tau_1), used to halve the grid.

Checks (python3 heisenberg_diagrams.py selftest): first order from the vertex algebra, the n = 2, 3, 4 integrands against exact Fock-space
imaginary-time correlators, orders 2-4 vs the Fock-space series, dense vs class-factorised tables, symmetry-equivalent references.
Requires numpy, scipy and heisenberg_gaussian_S1.py (and heisenberg_pt_checks.py for the checks) in the same directory.

Usage:  python3 heisenberg_diagrams.py ring 4 6 8                       # orders 1-4 for each N, lower-mean-field seam sign, vs exact diagonalisation
        python3 heisenberg_diagrams.py ring 8 --J2 0 0.3 0.5            # J1-J2 ring: one set of diagrams, evaluated for several J2
        python3 heisenberg_diagrams.py ring 8 --twist both --tol 1e-6 --raw --poly
        python3 heisenberg_diagrams.py ring 5 --dense --order 3 --tol 1e-4    # odd N: dense path (slow)
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
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heisenberg_gaussian_S1 as hg


# ================================================================================================ 1. TZ reference
class Reference:
    """TZ Gaussian reference state of the N-site ring (seam sign (-1)^twist) found by heisenberg_gaussian_S1.minimise for the coupling J2.
       twist=None picks the seam sign with the lower mean-field energy."""

    def __init__(self, N, twist=None, starts=24, seed=0, J2=0.0):
        self.N, self.J2_ref = N, J2
        if twist is None:
            outs = [hg.minimise(N, t, starts=starts, seed=seed, J2=J2) for t in (0, 1)]
            twist = 0 if outs[0]["E"] <= outs[1]["E"] + 1e-12 else 1
            out = outs[twist]
        else:
            out = hg.minimise(N, twist, starts=starts, seed=seed, J2=J2)
        self.twist, self.seam = twist, (1.0 if twist == 0 else -1.0)
        self.mu, self.E_min = out["mu"], out["E"]
        (self.tR, self.tR2), (self.kappa, self.kappa2) = out["t"], out["kap"]
        if N % 2 == 0 and abs(self.mu) < 1e-7:
            self.mu = 0.0                                  # optimum has <i rho eta> = 0 on a site: make the sublattice structure exact
        if abs(self.tR) < 1e-7 and abs(self.kappa) < 1e-7:
            self.tR = self.kappa = 0.0                     # no nearest-neighbour correlations in the optimum (J2 > J1)
        if abs(self.tR2) < 1e-7 and abs(self.kappa2) < 1e-7:
            self.tR2 = self.kappa2 = 0.0                   # no distance-2 correlations in the optimum (J2 < J1)
        K = np.zeros((N, N))
        for i in range(N):
            K[i, i] += self.mu
            for r, (t, ka) in enumerate(((self.tR, self.kappa), (self.tR2, self.kappa2)), start=1):
                j = (i + r) % N
                s = self.seam if i + r >= N else 1.0
                K[i, j] += s * (t + ka)
                K[j, i] += s * (t - ka)
        h = np.block([[np.zeros((N, N)), 2j * K], [-2j * K.T, np.zeros((N, N))]])
        lam, U = np.linalg.eigh(h)
        self.gap = np.abs(lam).min()
        if self.gap < 1e-9:
            raise ValueError(f"the TZ reference for N={N}, twist={twist}, J2={J2} has zero modes (degenerate reference): perturbation theory is undefined")
        pos = lam > 0
        assert pos.sum() == N
        self.lam, self.U = lam[pos], U[:, pos]
        self.C = 2 * self.U @ self.U.conj().T              # <chi_r chi_c>, row index = species * N + site
        # decoupled classes of the 2N Majoranas = connected components of the graph of h
        self.ncls, self.cls = connected_components(np.abs(h) > 1e-12, directed=False)
        assert all(np.abs(self.C[np.ix_(self.cls == a, self.cls == b)]).max() < 1e-9
                   for a in range(self.ncls) for b in range(a + 1, self.ncls))
        self.factorised = bool(self.ncls > 1)

    def orbit(self):
        """The (t_1, kappa_1, t_2, kappa_2) of all references related to this one by an exact symmetry of H.  The minimiser returns an arbitrary
           member of such an orbit (e.g. t -> -t at J2 = 0.1 and J2 = 0.2), and each member would otherwise cost a full set of diagrams.
             reflection i -> N-1-i                      : kappa_r -> -kappa_r                          (any N, both seam signs)
             sublattice sign (rho_i, eta_i) -> (-1)^i   : (t_1, kappa_1) -> -(t_1, kappa_1)            (even N)
             sign pattern + + - - + + - - ...           : (t_2, kappa_2) -> -(t_2, kappa_2)            (N divisible by 4, no distance-1 terms)
           Flipping the sign of all six Majoranas of a site leaves the spin operators of that site unchanged, so these maps leave H, and hence
           every coefficient of the series, unchanged; only H0 is relabelled."""
        t1, k1, t2, k2 = self.tR, self.kappa, self.tR2, self.kappa2
        orb = {(t1, k1, t2, k2), (t1, -k1, t2, -k2)}
        if self.N % 2 == 0:
            orb |= {(-t1, -k1, t2, k2), (-t1, k1, t2, -k2)}
        if self.N % 4 == 0 and t1 == 0.0 and k1 == 0.0:
            orb |= {(t1, k1, -t2, -k2), (t1, k1, -t2, k2), (t1, k1, t2, -k2)}
        return orb

    def equivalent(self, other, tol=1e-6):
        """True if both references give identical diagram sums: same N, seam sign, mu, and parameters equal within tol up to the symmetries above."""
        mine = (other.tR, other.kappa, other.tR2, other.kappa2)
        return ((self.N, self.twist) == (other.N, other.twist) and abs(self.mu - other.mu) < tol
                and any(max(abs(a - b) for a, b in zip(o, mine)) < tol for o in self.orbit()))

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
# legs of one colour on a bond (L, R):  0 = rho_L, 1 = eta_L, 2 = rho_R, 3 = eta_R;  the 8 even monomials ("labels"):
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


# ---- class maps: which legs of a label tuple belong to which decoupled class, and the sign of the reordering
_CM = {}


def class_maps(pats, ncls):
    """pats: for each of the n vertices the tuple (class of rho_L, eta_L, rho_R, eta_R).  For every label tuple (8^n of them, C order) returns
       masks[c]: bitmask of the class-c legs it selects (bit = position among all class-c legs, vertices in order, legs in order),
       sign: sign of the permutation that sorts the selected legs (vertex-major) into class-major order, so that Pf = sign prod_c Pf_c,
       and sizes[c]: the number of class-c legs of the configuration."""
    key = (pats, ncls)
    if key in _CM:
        return _CM[key]
    n = len(pats)
    off = [0] * ncls
    mc, cnt_v, inn = [], [], []
    for pat in pats:
        M = np.zeros((ncls, 8), np.int64)
        Cn = np.zeros((ncls, 8), np.int64)
        I = np.zeros(8, np.int64)
        loc, cnt = {}, [0] * ncls
        for leg in range(4):
            loc[leg] = cnt[pat[leg]]
            cnt[pat[leg]] += 1
        for m, lab in enumerate(LABELS):
            for leg in lab:
                c = pat[leg]
                M[c, m] |= 1 << (off[c] + loc[leg])
                Cn[c, m] += 1
            I[m] = sum(1 for a in range(len(lab)) for b in range(a + 1, len(lab)) if pat[lab[a]] > pat[lab[b]])
        mc.append(M)
        cnt_v.append(Cn)
        inn.append(I)
        for c in range(ncls):
            off[c] += cnt[c]

    def ax(p, arr):
        sh = [1] * n
        sh[p] = 8
        return arr.reshape(sh)

    masks = []
    for c in range(ncls):
        tot = np.zeros((8,) * n, np.int64)
        for p in range(n):
            tot = tot + ax(p, mc[p][c])
        masks.append(tot.ravel())
    inv = np.zeros((8,) * n, np.int64)
    for p in range(n):
        inv = inv + ax(p, inn[p])
    for p in range(n):
        for q in range(p + 1, n):
            for c in range(ncls):
                for c2 in range(c):
                    inv = inv + ax(p, cnt_v[p][c]) * ax(q, cnt_v[q][c2])
    sign = np.where(inv.ravel() % 2 == 1, -1.0, 1.0)
    _CM[key] = (masks, sign, tuple(off))
    return _CM[key]


def leg_rows(N, x, r=1):
    """Rows (species * N + site) of the four legs (rho_L, eta_L, rho_R, eta_R) of the bond (x, x + r mod N)."""
    xl, xr = x % N, (x + r) % N
    return np.array([xl, N + xl, xr, N + xr])


def build_A(ref, rows, pats, c, size, Gp):
    """Antisymmetric contraction matrix (g, size, size) of the class-c legs of the vertices with leg rows `rows`;  Gp[(p, q)] = G(t_p - t_q) (p < q)."""
    n = len(rows)
    g = next(iter(Gp.values())).shape[0]
    A = np.zeros((g, size, size), complex)
    sel = [rows[p][[l for l in range(4) if pats[p][l] == c]] for p in range(n)]
    off = np.concatenate([[0], np.cumsum([len(s) for s in sel])])
    for p in range(n):
        if len(sel[p]) == 0:
            continue
        sl = slice(off[p], off[p + 1])
        Cu = np.triu(ref.C[np.ix_(sel[p], sel[p])], 1)                           # equal-time contractions inside the vertex
        A[:, sl, sl] = (Cu - Cu.T)[None]
        for q in range(p + 1, n):
            if len(sel[q]) == 0:
                continue
            blk = Gp[(p, q)][:, sel[p][:, None], sel[q][None, :]]
            A[:, sl, off[q]:off[q + 1]] = blk
            A[:, off[q]:off[q + 1], sl] = -blk.transpose(0, 2, 1)
    return A


def phi_table(ref, positions, Gp, dense):
    """Phi[g, 8^n] = Pfaffian of the legs selected by each label tuple.  positions = ((x_1, r_1), ..., (x_n, r_n)): the bonds (x_p, x_p + r_p)."""
    N, n = ref.N, len(positions)
    rows = [leg_rows(N, x, r) for x, r in positions]
    if dense:
        ncls, cls = 1, np.zeros(2 * N, int)
    else:
        ncls, cls = ref.ncls, ref.cls
    pats = tuple(tuple(int(cls[i]) for i in rw) for rw in rows)
    masks, sign, sizes = class_maps(pats, ncls)
    Phi = sign[None, :].astype(complex)
    for c in range(ncls):
        P = pf_all(build_A(ref, rows, pats, c, sizes[c], Gp))
        Phi = Phi * P[:, masks[c]]
    return Phi


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
    """sum over colour assignments a_p and vertex kinds k_p (0..3 = species with weight 1/4, 4 = the shift -cbar_r of the mean removal, r = type of
       vertex p) of prod_{colour g} Psi[nu^g],  nu^g_p = k_p if a_p = g else 4 + k_p  (identity index 8 for the shift).  The couplings J_r are NOT
       included (they are powers of J2 applied by the caller)."""

    def __init__(self, cbar):
        self.cbar, self.cache, self.wcache = cbar, {}, {}

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
            mult = np.array([(MULT_FIXED if fixed else MULT_FREE)[max(a) + 1] for a in classes], float)
            self.cache[key] = (kinds, mult, tuple(kidx[g].ravel() for g in range(3)))
        return self.cache[key]

    def weights(self, m, fixed, ts):
        key = (m, fixed, ts)
        if key not in self.wcache:
            kinds, mult, _ = self.tables(m, fixed)
            wk = np.array([[(-self.cbar[ts[p]] if kp == 4 else 0.25) for p, kp in enumerate(k)] for k in kinds]).prod(axis=1)
            self.wcache[key] = (mult[:, None] * wk[None, :]).ravel()
        return self.wcache[key]

    def __call__(self, Psi, m, fixed, ts):
        _, _, (i0, i1, i2) = self.tables(m, fixed)
        return (Psi[:, i0] * Psi[:, i1] * Psi[:, i2]) @ self.weights(m, fixed, ts)


def _pair_slice(Psi, n, p, q):
    idx = [slice(None)] + [slice(None) if i in (p, q) else ID_VEC for i in range(n)]
    sub = Psi.reshape((Psi.shape[0],) + (9,) * n)[tuple(idx)]
    return sub.reshape(sub.shape[0], 81)


def vertex_mean(ref, r=1):
    """c_r = <v_{j,r,a}> (one bond of length r, one colour): the mean-field bond energy per colour; <H> = 3 N (c_1 + J2 c_2)."""
    N = ref.N
    idx = leg_rows(N, 0, r)                                   # rho_0, eta_0, rho_r, eta_r
    A = np.zeros((1, 4, 4), complex)
    for u in range(4):
        for v in range(u + 1, 4):
            A[0, u, v] = ref.C[idx[u], idx[v]]
            A[0, v, u] = -A[0, u, v]
    Phi = pf_all(A)[:, LAB_MASK]
    Psi = contract_vertices(Phi, 1)[0]
    return float((0.25 * sum(Psi[s] * Psi[4 + s] ** 2 for s in range(4))).real)


def vertex_means(ref):
    return {1: vertex_mean(ref, 1), 2: vertex_mean(ref, 2)}


def first_order(ref, J2=0.0):
    """E_0 + E_1 = <H>_0 = 3 N (c_1 + J2 c_2) from the vertex algebra."""
    c = vertex_means(ref)
    return 3 * ref.N * (c[1] + J2 * c[2])


def integrand(ref, positions, Gp, cs, dense):
    """Colour-summed connected correlator kappa_n of the mean-removed vertices at the bonds `positions` = ((x_p, r_p)) (the first with colour x, no
       coupling constants), for each time node."""
    n = len(positions)
    ts = tuple(r for _, r in positions)
    Psi = contract_vertices(phi_table(ref, positions, Gp, dense), n)
    if n <= 3:
        return cs(Psi, n, True, ts)
    pair = lambda p, q: cs(_pair_slice(Psi, 4, p, q), 2, p == 0, (ts[p], ts[q]))
    return cs(Psi, 4, True, ts) - pair(0, 1) * pair(2, 3) - pair(0, 2) * pair(1, 3) - pair(0, 3) * pair(1, 2)


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
    """T_{n,m}, the diagram sums of P_n = sum_m T_{n,m} J2^m (J1 = 1) for the TZ reference `ref`, n = 2, 3, 4 (plain bond vertices);
       m = number of distance-2 vertices.  types = (1,) for J2 = 0, (1, 2) otherwise."""

    def __init__(self, ref, n, tol=1e-6, dense=None, chunk=32, symmetric=True, verbose=False, types=(1,)):
        if n not in (2, 3, 4):
            raise ValueError("order n must be 2, 3 or 4")
        self.ref, self.n, self.chunk, self.verbose, self.types = ref, n, chunk, verbose, tuple(types)
        self.dense = (not ref.factorised) if dense is None else dense
        if not self.dense and not ref.factorised:
            raise ValueError("the factorised path needs a reference with decoupled classes (even N, mu = 0); use dense=True (slow)")
        lam = ref.lam
        Omin = 2 * lam.min()
        Omax = min((16 if n == 4 else 8) * lam.max(), 3 * lam.sum())
        self.alpha, self.w, self.err = laplace_nodes(Omin, max(Omax, 1.01 * Omin), tol)
        self.K = len(self.alpha)
        self.cs = ColourSums(vertex_means(ref))
        nodes = np.array(list(itertools.product(range(self.K), repeat=n - 1)))
        mult = np.ones(len(nodes))
        if symmetric and n >= 3:                                         # I(tau_1..tau_{n-1})^* = I(tau_{n-1}..tau_1)
            lo, hi = nodes[:, 0], nodes[:, -1]
            keep = lo <= hi
            mult = np.where(lo < hi, 2.0, 1.0)[keep]
            nodes = nodes[keep]
        self.nodes, self.wts = nodes, np.prod(self.w[nodes], axis=1) * mult

    def energy(self):
        """Array T[0..n]: T[m] = -(-1)^n 3N sum of the diagrams with m distance-2 vertices (J1 = 1, no J2 factors)."""
        ref, n, N = self.ref, self.n, self.ref.N
        T, t0 = np.zeros(n + 1, complex), time.time()
        positions = list(itertools.product(range(N), repeat=n - 1))
        tuples = list(itertools.product(self.types, repeat=n))
        nch = -(-len(self.nodes) // self.chunk)
        for c in range(nch):
            sl = slice(c * self.chunk, (c + 1) * self.chunk)
            nd, wch = self.nodes[sl], self.wts[sl]
            Gp = {(p, q): ref.G(self.alpha[nd[:, p:q]].sum(axis=1)) for p in range(n) for q in range(p + 1, n)}     # t_p - t_q = tau_p + .. + tau_{q-1}
            for ts in tuples:
                m = sum(1 for r in ts if r == 2)
                acc = 0.0
                for xs in positions:
                    pos = ((0, ts[0]),) + tuple(zip(xs, ts[1:]))
                    acc += (wch * integrand(ref, pos, Gp, self.cs, self.dense)).sum()
                T[m] += acc
            if self.verbose:
                print(f"      n={n}: chunk {c + 1}/{nch}  {time.time() - t0:.0f} s", flush=True)
        return (-(-1) ** n * 3 * N * T).real


def series_coefficients(ref, nmax=4, tol=1e-6, dense=None, chunk=32, symmetric=True, verbose=False, types=(1,)):
    """Polynomials in J2 (coefficient arrays, J1 = 1): P[1] = <H>, P[n] = T[n] for n = 2..nmax, plus the quadrature sizes."""
    c = vertex_means(ref)
    P = {1: np.zeros(nmax + 1)}
    P[1][0], P[1][1] = 3 * ref.N * c[1], 3 * ref.N * c[2]
    K = {}
    for n in range(2, nmax + 1):
        d = Diagrams(ref, n, tol, dense, chunk, symmetric, verbose, types)
        T = np.zeros(nmax + 1)
        T[:n + 1] = d.energy()
        P[n] = T
        K[n] = d.K
    return P, K


def energies(P, J2, nmax):
    """P_n(J2) and the Rayleigh-Schroedinger energies E_n(J2) from the coefficient polynomials."""
    Pn = {n: float(np.polyval(P[n][::-1], J2)) for n in range(1, nmax + 1)}
    E = {1: Pn[1]}
    for n in range(2, nmax + 1):
        E[n] = sum(math.comb(n - 2, j) * Pn[n - j] for j in range(n - 1))
    return Pn, E


def energy_polynomials(P, nmax):
    """Coefficients e[n][m] of E_n(J2) = sum_m e[n][m] J2^m."""
    e = {1: P[1]}
    for n in range(2, nmax + 1):
        e[n] = sum(math.comb(n - 2, j) * P[n - j] for j in range(n - 1))
    return e


def series(ref, nmax=4, tol=1e-6, dense=None, chunk=32, symmetric=True, verbose=False, J2=0.0):
    """Orders 1..nmax at one value of J2: dict with the plain-cumulant energies P_n, the energies E_n of the Rayleigh-Schroedinger series, and the quadrature sizes."""
    P, K = series_coefficients(ref, nmax, tol, dense, chunk, symmetric, verbose, (1, 2) if J2 != 0.0 else (1,))
    Pn, E = energies(P, J2, nmax)
    return dict(P=Pn, E=E, K=K)


# ================================================================================================ 7. drivers
def run_ring(Ns, nmax, tol, twists, J2s, dense, raw, chunk, symmetric, verbose, poly=False):
    for N in Ns:
        Eex = {J2: hg.exact_energy(N, 1.0, J2) for J2 in J2s}
        for tw in twists:
            groups = []                                            # diagrams are shared by all J2 with the same reference (up to symmetries)
            for J2 in J2s:
                ref = Reference(N, twist=tw, J2=J2)
                for g in groups:
                    if g[0].equivalent(ref):
                        g[1].append(J2)
                        break
                else:
                    groups.append((ref, [J2]))
            for ref, js in groups:
                t0 = time.time()
                use_dense = dense or not ref.factorised
                types = (1, 2) if any(j != 0.0 for j in js) else (1,)
                print(f"\nN = {N}, seam sign (-1)^twist, twist = {ref.twist}{' (lower mean-field energy)' if tw is None else ''};  "
                      f"mu = {ref.mu:.4g}, tR = {ref.tR:.5f}, kappa = {ref.kappa:.5f}"
                      + (f", tR2 = {ref.tR2:.5f}, kappa2 = {ref.kappa2:.5f}" if (ref.tR2 or ref.kappa2) else "")
                      + f";  quasi-particle energies in [{ref.lam.min():.4f}, {ref.lam.max():.4f}]")
                print(f"   path: {'dense 16-leg Pfaffians' if use_dense else f'class-factorised Pfaffians ({ref.ncls} decoupled classes of Majoranas)'};  "
                      f"J2 evaluated: {', '.join(f'{j:g}' for j in js)};  reference minimised at J2 = {ref.J2_ref:g}")
                P, K = series_coefficients(ref, nmax, tol, use_dense, chunk, symmetric, verbose, types)
                for J2 in js:
                    Pn, E = energies(P, J2, nmax)
                    S = 0.0
                    print(f"   J2 = {J2:g}:  exact ground-state energy (ED) = {Eex[J2]:.10f}")
                    print(f"   {'order':>5} {'E_n':>14} {'partial sum':>14} {'from exact':>11} {'sum/N':>10}" + (f" {'P_n':>14}" if raw else ""))
                    for n in range(1, nmax + 1):
                        S += E[n]
                        label = f"{n}" + (" (MF)" if n == 1 else "")
                        print(f"   {label:>5} {E[n]:>+14.8f} {S:>14.8f} {100 * (S - Eex[J2]) / abs(Eex[J2]):>+10.2f}% {S / N:>10.6f}" + (f" {Pn[n]:>+14.8f}" if raw else ""))
                if poly:
                    e = energy_polynomials(P, nmax)
                    print("   E_n(J2) = sum_m e_{n,m} J2^m   (J1 = 1; valid for this reference):")
                    for n in range(1, nmax + 1):
                        print(f"     n={n}: " + "  ".join(f"{e[n][m]:+.8f}" for m in range(n + 1)))
                print(f"   quadrature nodes per time interval: {K}, relative tolerance {tol:g};  {time.time() - t0:.0f} s")
                sys.stdout.flush()


# ================================================================================================ 8. self-test
def selftest(quick=False):
    import scipy.sparse as sp
    from scipy.sparse.linalg import expm_multiply
    import heisenberg_pt_checks as pc
    bad = 0

    def verdict(ok):
        nonlocal bad
        bad += 0 if ok else 1
        return "ok" if ok else "FAILED"

    print("1. first order: <H>_0 = 3 N (c_1 + J2 c_2) from the vertex algebra  vs  the minimiser's energy")
    for N, tw, J2 in [(3, 0, 0.0), (4, 1, 0.0), (5, 0, 0.0), (6, 1, 0.0), (4, 1, 0.4), (5, 0, 0.4), (6, 0, 0.4), (6, 1, 0.5), (6, 0, 1.5), (7, 1, 0.7)]:
        ref = Reference(N, twist=tw, J2=J2)
        d = abs(first_order(ref, J2).real - ref.E_min)
        print(f"   N={N} twist={tw} J2={J2:g}: {first_order(ref, J2).real:+.10f}  (minimiser {ref.E_min:+.10f})  |diff| = {d:.1e}  {verdict(d < 1e-8)}")

    print("2. integrands (n = 2, 3, 4)  vs  exact Fock-space imaginary-time correlators (N = 4 and 5; vertex types 1 and 2; random bonds and times)")
    rng = np.random.default_rng(7)
    for N, tw, J2 in [(4, 1, 0.4), (5, 0, 0.4)]:
        ref = Reference(N, twist=tw, J2=J2)
        gam = pc.majoranas(3 * N)
        Sl = [pc.spin_6(gam[6 * i:6 * i + 6]) for i in range(N)]
        H0 = pc.mean_field_6(gam, N, ref.mu, (ref.tR, ref.tR2), (ref.kappa, ref.kappa2), tw)
        w, v = pc.lowest_two(H0)
        psi0, E0 = v[:, 0], w[0]
        eye = sp.identity(H0.shape[0], format="csr")
        cs = ColourSums(vertex_means(ref))
        c = vertex_means(ref)
        worst, scale = 0.0, 0.0
        for n in (2, 3, 4):
            for trial in range(3 if n < 4 else 2):
                pos = tuple((int(rng.integers(0, N)), int(rng.integers(1, 3))) for _ in range(n))
                pos = ((0, pos[0][1]),) + pos[1:]
                times = np.sort(rng.uniform(0.2, 3.0, n))[::-1]
                times = times - times[-1]
                for dense in ([True] if not ref.factorised else [True, False]):
                    Gp = {(p, q): ref.G([times[p] - times[q]]) for p in range(n) for q in range(p + 1, n)}
                    got = integrand(ref, pos, Gp, cs, dense)[0]

                    def corr(idx, colours):
                        vec = None
                        for k in range(len(idx) - 1, -1, -1):
                            p = idx[k]
                            x, r = pos[p]
                            V = Sl[x % N][colours[p]] @ Sl[(x + r) % N][colours[p]]
                            ww = psi0 if vec is None else vec
                            vec = V @ ww - c[r] * ww
                            if k > 0:
                                dt = times[idx[k - 1]] - times[p]
                                vec = expm_multiply(-(H0 - E0 * eye) * dt, vec)
                        return np.vdot(psi0, vec)
                    ref_val = 0.0
                    for cols in itertools.product(range(3), repeat=n - 1):
                        cl = (0,) + cols
                        if n < 4:
                            ref_val += corr(list(range(n)), cl)
                        else:
                            ref_val += (corr([0, 1, 2, 3], cl) - corr([0, 1], cl) * corr([2, 3], cl)
                                        - corr([0, 2], cl) * corr([1, 3], cl) - corr([0, 3], cl) * corr([1, 2], cl))
                    worst, scale = max(worst, abs(got - ref_val)), max(scale, abs(ref_val))
        print(f"   N={N} twist={tw} J2={J2:g} ({'dense' if not ref.factorised else 'dense and factorised'}): max |diagram - Fock| = {worst:.1e} "
              f"(largest value {scale:.1e})  {verdict(worst < 1e-10 * max(scale, 1e-3) + 1e-12)}")

    print("3. series (orders 2-4)  vs  the brute-force Fock-space Rayleigh-Schroedinger series (heisenberg_pt_checks)")
    cases = [(4, 0, 1e-5, 0.0), (4, 1, 1e-5, 0.0), (4, 1, 1e-5, 0.4), (6, 0, 1e-4, 0.4)]
    if not quick:
        cases.append((5, 0, 1e-4, 0.4))
    for N, tw, tol, J2 in cases:
        t0 = time.time()
        ref = Reference(N, twist=tw, J2=J2)
        dense = not ref.factorised
        nmax = 3 if (dense or (N == 6 and quick)) else 4
        res = series(ref, nmax, tol, dense, J2=J2)
        fock = pc.tz_series(N, tw, nmax, J2=J2)["E"]
        diffs = [abs(res["E"][n] - fock[n]) for n in range(2, nmax + 1)]
        scale = max(abs(fock[n]) for n in range(2, nmax + 1))
        print(f"   N={N} twist={tw} J2={J2:g} ({'dense' if dense else 'factorised'}, tol {tol:g}): " +
              "  ".join(f"E_{n} = {res['E'][n]:+.8f} (Fock {fock[n]:+.8f})" for n in range(2, nmax + 1)) +
              f"   max|diff| = {max(diffs):.1e}  {verdict(max(diffs) < 20 * tol * scale)}  ({time.time() - t0:.0f} s)")

    print("4. dense 16-leg Pfaffian tables  vs  class-factorised tables (N=4, random time nodes, bond positions and types)")
    ref = Reference(4, twist=1, J2=0.4)
    d = Diagrams(ref, 4, 1e-4, dense=False, types=(1, 2))
    rng = np.random.default_rng(1)
    nd = d.nodes[rng.integers(0, len(d.nodes), 6)]
    Gp = {(p, q): ref.G(d.alpha[nd[:, p:q]].sum(axis=1)) for p in range(4) for q in range(p + 1, 4)}
    worst = 0.0
    for xs, rs in [((1, 2, 3), (1, 1, 1, 1)), ((0, 0, 0), (2, 1, 2, 2)), ((2, 2, 1), (2, 2, 1, 1)), ((3, 1, 0), (1, 2, 2, 1))]:
        pos = ((0, rs[0]),) + tuple(zip(xs, rs[1:]))
        worst = max(worst, np.abs(phi_table(ref, pos, Gp, True) - phi_table(ref, pos, Gp, False)).max(),
                    np.abs(integrand(ref, pos, Gp, d.cs, True) - integrand(ref, pos, Gp, d.cs, False)).max())
    print(f"   max |dense - factorised| over tables and integrands = {worst:.1e}  {verdict(worst < 1e-12)}")

    print("5. references that differ only by an exact symmetry (t -> -t, kappa -> -kappa, as returned by the minimiser at different J2) give the same series")
    print("   (orders 2-3 at J2 = 0.1, tol 1e-5; this is what lets run_ring evaluate all J2 below the switching point from one set of diagrams)")
    for N in ((8,) if quick else (8, 10)):
        refs = [Reference(N, twist=None, J2=j) for j in (0.0, 0.1, 0.2, 0.3)]
        base = series(refs[0], 3, 1e-5, J2=0.1)["E"]
        for r in refs[1:]:
            e = series(r, 3, 1e-5, J2=0.1)["E"]
            dE = max(abs(e[n] - base[n]) for n in (2, 3))
            same = refs[0].equivalent(r)
            print(f"   N={N}: reference minimised at J2 = {r.J2_ref:g}: (t1, kappa1) = ({r.tR:+.5f}, {r.kappa:+.5f}) vs ({refs[0].tR:+.5f}, {refs[0].kappa:+.5f}) at J2 = 0;  "
                  f"equivalent: {same};  max|E_n - E_n(J2=0 reference)| = {dE:.1e}  {verdict(same and dE < 1e-6)}")
    print("\nALL CHECKS PASSED" if bad == 0 else f"\n{bad} CHECK(S) FAILED")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ring", help="orders 1-4 of the series for ring sizes N, compared with exact diagonalisation")
    p.add_argument("N", nargs="+", type=int)
    p.add_argument("--J2", type=float, nargs="+", default=[0.0], help="next-nearest-neighbour coupling(s), J1 = 1 (default 0); several values share the diagrams")
    p.add_argument("--order", type=int, default=4, choices=[2, 3, 4], help="highest order (default 4)")
    p.add_argument("--twist", choices=["best", "0", "1", "both"], default="best", help="seam sign of the reference (default: lower mean-field energy)")
    p.add_argument("--tol", type=float, default=1e-6, help="relative accuracy of each time-interval quadrature (default 1e-6)")
    p.add_argument("--dense", action="store_true", help="force the dense 16-leg path (automatic for odd N; slow)")
    p.add_argument("--raw", action="store_true", help="also print the plain-cumulant energies P_n")
    p.add_argument("--poly", action="store_true", help="print the coefficients e_{n,m} of E_n(J2) = sum_m e_{n,m} J2^m")
    p.add_argument("--chunk", type=int, default=32, help="time nodes per batch (memory/speed trade-off)")
    p.add_argument("--no-sym", action="store_true", help="do not use the time-reversal symmetry of the integrand (2x slower)")
    p.add_argument("--verbose", action="store_true")
    q = sub.add_parser("selftest", help="validation against the Fock-space series")
    q.add_argument("--quick", action="store_true", help="skip the slow checks")
    a = ap.parse_args()
    if a.cmd == "selftest":
        raise SystemExit(selftest(a.quick))
    twists = {"best": [None], "0": [0], "1": [1], "both": [0, 1]}[a.twist]
    run_ring(a.N, a.order, a.tol, twists, a.J2, a.dense, a.raw, a.chunk, not a.no_sym, a.verbose, a.poly)


if __name__ == "__main__":
    main()