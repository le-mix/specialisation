#!/usr/bin/env python3
"""
Gaussian (6-Majorana) variational minimiser and exact diagonaliser for the spin-1/2 Heisenberg model on an arbitrary graph, in particular the
nearest-neighbour kagome antiferromagnet on a torus (kagome_lattice.py):

        H = sum_{bonds (i,j)} J_ij S_i . S_j ,       J_ij = 1 on every kagome bond by default  (--Jdown sets the down-triangle bonds: breathing kagome).

This is the kagome counterpart of heisenberg_gaussian_S1.py.  It uses the same representation, the same time-reversal class TZ and the same
bond energy; what changes is that there is no translation-invariant Fourier ansatz (the optimum turns out to be a dimer covering, which breaks the lattice
translations; see "What it finds"), so the state is minimised in real space with no symmetry assumed.

State (per colour a = x, y, z; the three colours are identical copies, so the state is rotation invariant):
    <rho_i eta_j> = -i p_ij ,   <rho_i rho_j> = <eta_i eta_j> = delta_ij ,   P = (p_ij) a real N x N matrix with ||P|| <= 1 (operator norm).
Pure states are exactly the orthogonal matrices P in O(N) (two components, det P = +-1).  The energy of one bond (i, j) depends only on the
2 x 2 block (a, b, c, d) = (p_ii, p_ij, p_ji, p_jj),
    e(a,b,c,d) = 3/16 [ -(b^2+c^2) - (a-d)(b^2-c^2) + (b^2+c^2)(a d - b c) ]            (<S_i.S_j>; identical to heisenberg_gaussian_S1.bond),
so that  E(P) = sum_bonds J_ij e(p_ii, p_ij, p_ji, p_jj).   H is invariant under the gauge transformation (rho_i, eta_i) -> -(rho_i, eta_i) (all six Majoranas
of a site: S_i is unchanged), which acts as p_ij -> s_i s_j p_ij; only signs of products of p (or K) around closed loops are physical.

Minimisation.  E is a polynomial of degree 4 in the entries of P.  The program minimises it over O(N): P = P0 exp(A), A antisymmetric, with the analytic
gradient (Frechet derivative of exp via the eigendecomposition of iA) and L-BFGS, re-centring P0 until the Riemannian gradient P^T grad E - (P^T grad E)^T
vanishes, from `starts` random starts (ground states of random mean-field Hamiltonians K with the support of the lattice); the lowest energy is kept.
At a stationary point  grad E = P S  with S symmetric, and the mean-field (Fock) Hamiltonian of the state is
    H_MF = sum_a sum_ij K_ij i rho^a_i eta^a_j ,      K = (1/3) grad E = P S / 3    (the chain code's mu = E_a/3, t = (E_b+E_c)/6, kappa = (E_b-E_c)/6, in real space).
The state is the ground state of H_MF (the aufbau condition, P = -polar(K)) exactly when all eigenvalues of S are negative; if some are not, K := -P|S|/3 is
used instead (same state, positive quasi-particle energies) and the output says so.  The quasi-particle energies are the eigenvalues of
h = [[0, 2iK], [-2iK^T, 0]], i.e. twice the singular values 2 sigma(K) = (2/3)|eig S|; zero eigenvalues of S = zero modes = degenerate reference.

What it finds.  On the clusters 12, 18a, 18b and 24 (24-48 random starts each) the lowest Gaussian state is a perfect DIMER COVERING: every row of P has a single entry
+-1 with p_ij p_ji = +1 on each dimer (a singlet), E = -3N/8 (-0.375 per site), S = -9/8 times the identity, quasi-particle gap 3/4, aufbau.  Every other stationary point found lies
higher (N = 12, 18a, 24: -0.3125, -0.3333, -0.3438 per site); most have zero modes, the others are gapped (gap 0.1-0.25) but violate the aufbau condition.  The dimer state is
reached by only a minority of the random starts (12 of 48 at N = 12, 1 of 24 at 18a in one test), so use --starts 48 or more for N >= 18.  The same happens on the J1-J2 ring (even N <= 10, J2 = 0 and 0.3): E = -3N/8, far below the translation-invariant optimum of heisenberg_gaussian_S1.py, which the
chain code was restricted to (N = 8: -3.000 against -1.827; exact -3.651).  All perfect coverings are degenerate at the mean-field level (kagome_diagrams.py lifts the
degeneracy at second order; the coverings of a cluster can be enumerated with kagome_diagrams.all_coverings).  An odd number of sites has no perfect covering: the lowest states
then have zero modes (degenerate reference) and perturbation theory around them is undefined.  That the dimer covering is the global minimum is an observation from random
starts (it is not proved) and was checked for the clusters above only.

Usage:  python3 kagome_gaussian_S1.py 12                     # the 12-site torus of the ED papers; also 18a 18b 24 27a 27b 30 36a 36b 36c 36d, or 3x3 (L1 x L2 cells)
        python3 kagome_gaussian_S1.py 18a --ed --starts 48
        python3 kagome_gaussian_S1.py ring 10 --J2 0.3 --ed  # the J1-J2 ring written as a graph (check against heisenberg_gaussian_S1)
Requires numpy, scipy, kagome_lattice.py and heisenberg_gaussian_S1.py (for the bond energy) in the same directory.
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heisenberg_gaussian_S1 as hg
import kagome_lattice as kl

# Exact ground-state energies (total, J = 1) of the periodic kagome clusters, Laeuchli, Sudan, Soerensen, PRB 83, 212401 (2011), Table I.
# Entries marked True were reproduced by this program's own diagonalisation (see the self-test); the others are quoted from the table as extracted.
LITERATURE_ED = {"12": (-5.444875216, True), "18a": (-8.064482605, True), "18b": (-8.048270773, True), "24": (-10.589965547, True),
                 "27a": (-11.793996213, False), "27b": (-11.779504985, False), "30": (-13.154318943, False),
                 "36a": (-15.787874847, False), "36b": (-15.806927756, False), "36c": (-15.814334002, False), "36d": (-15.781555118, False)}


# ----------------------------------------------------------------------------------------------------------
# Graphs
# ----------------------------------------------------------------------------------------------------------
def ring_graph(N, J2=0.0):
    """Bonds and weights of the J1-J2 ring written as a graph (J1 = 1); coinciding pairs (small N) are merged by adding their couplings."""
    w = {}
    for r, J in ((1, 1.0), (2, J2)):
        if J == 0.0:
            continue
        for i in range(N):
            j = (i + r) % N
            if i != j:
                key = (min(i, j), max(i, j))
                w[key] = w.get(key, 0.0) + J
    keys = sorted(w)
    return np.array(keys, int).reshape(-1, 2), np.array([w[k] for k in keys])


# ----------------------------------------------------------------------------------------------------------
# Energy functional
# ----------------------------------------------------------------------------------------------------------
class Functional:
    """E(P) = sum_b w_b e(p_ii, p_ij, p_ji, p_jj) on the graph `bonds` (array (nb, 2)) with couplings `weights`."""

    def __init__(self, N, bonds, weights):
        self.N = int(N)
        self.bonds = np.asarray(bonds, int).reshape(-1, 2)
        self.I, self.J = self.bonds[:, 0], self.bonds[:, 1]
        self.w = np.asarray(weights, float)
        if len(self.w) != len(self.I):
            raise ValueError("weights and bonds have different lengths")

    def energy_grad(self, P):
        I, J, w = self.I, self.J, self.w
        e, ea, eb, ec, ed = hg.bond(P[I, I], P[I, J], P[J, I], P[J, J])
        G = np.zeros((self.N, self.N))
        np.add.at(G, (I, I), w * ea)
        np.add.at(G, (J, J), w * ed)
        np.add.at(G, (I, J), w * eb)
        np.add.at(G, (J, I), w * ec)
        return float((w * e).sum()), G

    def energy(self, P):
        return self.energy_grad(P)[0]

    def bond_energies(self, P):
        """<S_i.S_j> of every bond (without the coupling)."""
        return hg.bond(P[self.I, self.I], P[self.I, self.J], P[self.J, self.I], P[self.J, self.J])[0]


def polar(K):
    U, s, Vt = np.linalg.svd(K)
    return U @ Vt


# ----------------------------------------------------------------------------------------------------------
# Minimisation over O(N)
# ----------------------------------------------------------------------------------------------------------
def _skew(x, iu, N):
    A = np.zeros((N, N))
    A[iu] = x
    return A - A.T


def _objective(x, P0, F, iu):
    """f(x) = E(P0 exp(A(x))) and its gradient (the Frechet derivative of exp through the eigendecomposition of the Hermitian matrix iA)."""
    N = F.N
    A = _skew(x, iu, N)
    w, V = np.linalg.eigh(1j * A)
    Vh = V.conj().T
    Q = ((V * np.exp(-1j * w)) @ Vh).real
    P = P0 @ Q
    E, G = F.energy_grad(P)
    Mt = Vh @ (P0.T @ G) @ V
    phi = np.exp(0.5j * (w[:, None] + w[None, :])) * np.sinc((w[None, :] - w[:, None]) / (2 * np.pi))    # int_0^1 e^{i w_k (1-s) + i w_l s} ds
    Gam = (V @ (Mt * phi) @ Vh).real
    g = (Gam - Gam.T)[iu]
    return E, g


def riemannian_residual(P, F):
    """max |M - M^T| with M = P^T grad E: zero at a stationary point of E on O(N)."""
    G = F.energy_grad(P)[1]
    M = P.T @ G
    return float(np.abs(M - M.T).max())


def refine(F, P0, tol=1e-13, maxouter=40):
    """Local minimisation on O(N) from the orthogonal matrix P0; returns (P, E, number of L-BFGS iterations)."""
    N = F.N
    iu = np.triu_indices(N, 1)
    P = P0.copy()
    E_old, nit = None, 0
    for _ in range(maxouter):
        res = minimize(_objective, np.zeros(len(iu[0])), args=(P, F, iu), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=20000, maxfun=40000, maxcor=40, ftol=1e-17, gtol=1e-12))
        nit += res.nit
        A = _skew(res.x, iu, N)
        w, V = np.linalg.eigh(1j * A)
        Q = ((V * np.exp(-1j * w)) @ V.conj().T).real
        P = polar(P @ Q)                                        # polar() removes the accumulated rounding error of orthogonality
        E = F.energy(P)
        if E_old is not None and abs(E - E_old) <= tol * max(1.0, abs(E)) and riemannian_residual(P, F) < 1e-9:
            break
        E_old = E
    return P, F.energy(P), nit


def random_start(F, rng, noise=1e-3):
    """Ground state of a random mean-field Hamiltonian K with the support of the lattice (random on-site terms and random bond amplitudes in both directions)."""
    N = F.N
    K = noise * rng.normal(size=(N, N))
    K[np.arange(N), np.arange(N)] += rng.normal(size=N)
    K[F.I, F.J] += rng.normal(size=len(F.I))
    K[F.J, F.I] += rng.normal(size=len(F.I))
    return -polar(K)


def fock_matrix(P, F, tol=1e-8):
    """Mean-field (Fock) matrix K of the state P, the symmetric matrix S of grad E = P S, and the aufbau flag."""
    G = F.energy_grad(P)[1]
    S = P.T @ G
    S = 0.5 * (S + S.T)
    ev, V = np.linalg.eigh(S)
    aufbau = bool(ev.max() < -tol)
    if aufbau:
        K = G / 3.0
    else:
        K = -(P @ ((V * np.abs(ev)) @ V.T)) / 3.0
    return K, ev, aufbau


def minimise(N, bonds, weights, starts=24, seed=0, verbose=False, tol=1e-13):
    """Best of `starts` local minimisations of E over the pure TZ Gaussian states of the graph.  Returns a dict (see `describe`)."""
    F = Functional(N, bonds, weights)
    rng = np.random.default_rng(seed)
    runs = []
    t0 = time.time()
    for s in range(starts):
        P, E, nit = refine(F, random_start(F, rng), tol=tol)
        runs.append((E, P, nit))
        if verbose:
            print(f"   start {s + 1}/{starts}: E = {E:.12f}  ({nit} iterations, {time.time() - t0:.0f} s)", flush=True)
    runs.sort(key=lambda r: r[0])
    E, P, nit = runs[0]
    K, ev, aufbau = fock_matrix(P, F)
    sv = np.linalg.svd(K, compute_uv=False)
    energies = np.array([r[0] for r in runs])
    return dict(N=N, bonds=F.bonds, weights=F.w, P=P, K=K, E=E, E_per_site=E / N, aufbau=aufbau, S_eig=ev, gap=2 * sv.min(),
                residual=riemannian_residual(P, F), detP=float(np.linalg.det(P)), energies=energies, iterations=nit,
                orthogonality=float(np.abs(P.T @ P - np.eye(N)).max()))


# ----------------------------------------------------------------------------------------------------------
# Flux (gauge-invariant loop signs) of the mean-field Hamiltonian
# ----------------------------------------------------------------------------------------------------------
def loop_signs(K, loop, tol=1e-9):
    """Signs (+1, -1, 0 if some link vanishes) of the products of K around the loop in the two orientations."""
    out = []
    for seq in (list(loop), list(loop)[::-1]):
        pr = 1.0
        for a, b in zip(seq, seq[1:] + seq[:1]):
            pr *= K[a, b]
        out.append(0 if abs(pr) < tol * 1e-3 else int(np.sign(pr)))
    return tuple(out)


def flux_summary(K, torus):
    """Counts of the sign patterns of the loop products around up triangles, down triangles and hexagons: {kind: {(sign, sign_reversed): count}}."""
    out = {"up": {}, "down": {}, "hexagon": {}}
    for kind, t in torus.triangles:
        s = loop_signs(K, t)
        out[kind][s] = out[kind].get(s, 0) + 1
    for h in torus.hexagons:
        s = loop_signs(K, h)
        out["hexagon"][s] = out["hexagon"].get(s, 0) + 1
    return out


# ----------------------------------------------------------------------------------------------------------
# Exact diagonalisation (S^z = 0 or 1/2 sector, which contains the ground state)
# ----------------------------------------------------------------------------------------------------------
def exact_energy(N, bonds, weights, k=1, max_N=24):
    """Lowest eigenvalue(s) of H = sum_b w_b S_i.S_j in the sector of smallest |S^z| (it contains every spin multiplet, hence the ground state)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    if N > max_N:
        raise ValueError(f"exact diagonalisation of N = {N} sites is not attempted (limit {max_N}: about 2.7e6 basis states and 1 GB at N = 24)")
    nup = N // 2
    s = np.arange(1 << N, dtype=np.int64)
    pc = np.zeros(1 << N, np.int8)
    t = s.copy()
    for _ in range(N):
        pc += (t & 1).astype(np.int8)
        t >>= 1
    basis = s[pc == nup]
    del s, pc, t
    dim = len(basis)
    diag = np.zeros(dim)
    rows, cols, vals = [], [], []
    for (i, j), w in zip(np.asarray(bonds, int), np.asarray(weights, float)):
        if w == 0.0:
            continue
        bi, bj = (basis >> int(i)) & 1, (basis >> int(j)) & 1
        diag += 0.25 * w * np.where(bi == bj, 1.0, -1.0)
        flip = np.nonzero(bi != bj)[0]
        new = basis[flip] ^ ((1 << int(i)) | (1 << int(j)))
        rows.append(flip.astype(np.int32))
        cols.append(np.searchsorted(basis, new).astype(np.int32))
        vals.append(np.full(len(flip), 0.5 * w))
    rows.append(np.arange(dim, dtype=np.int32))
    cols.append(np.arange(dim, dtype=np.int32))
    vals.append(diag)
    H = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim)).tocsr()
    del rows, cols, vals
    if dim <= 600:
        return np.linalg.eigvalsh(H.toarray())[:k]
    rng = np.random.default_rng(1)
    w_ = eigsh(H, k=k, which="SA", v0=rng.normal(size=dim), tol=1e-12)[0]
    return np.sort(w_)


# ----------------------------------------------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------------------------------------------
def describe(r, torus=None, ed=None):
    N = r["N"]
    print(f"N = {N}, {len(r['bonds'])} bonds:  best of {len(r['energies'])} starts")
    print(f"  E = {r['E']:.10f}   E/N = {r['E_per_site']:.10f}   (det P = {r['detP']:+.0f}; ||P^T P - 1|| = {r['orthogonality']:.1e})")
    en = np.round(r["energies"], 8)
    vals, cnt = np.unique(en, return_counts=True)
    print("  distinct local minima (energy x number of starts): " + ", ".join(f"{v:.6f} x{c}" for v, c in list(zip(vals, cnt))[:8]) + (" ..." if len(vals) > 8 else ""))
    print(f"  stationarity residual max|P^T grad E - transpose| = {r['residual']:.1e};   aufbau (all eigenvalues of S negative): {r['aufbau']}"
          f"   (largest eigenvalue of S: {r['S_eig'].max():+.4f})")
    print(f"  quasi-particle gap of the mean-field Hamiltonian, 2 min sigma(K) = {r['gap']:.6f}" + ("   <-- zero modes: the reference is degenerate" if r["gap"] < 1e-7 else ""))
    if torus is not None:
        fl = flux_summary(r["K"], torus)
        print("  loop signs of K (sign of the product around the loop in the two orientations: count):")
        for kind in ("up", "down", "hexagon"):
            print(f"    {kind:>8}: " + ", ".join(f"({a:+d},{b:+d}) x{c}" for (a, b), c in sorted(fl[kind].items())))
    if ed is not None:
        print(f"  exact ED E0 = {ed:.10f}  (E0/N = {ed / N:.10f});  mean field - exact = {r['E'] - ed:+.6e}  (must be >= 0);  MF/exact = {r['E'] / ed:.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cluster", help="kagome cluster (12, 18a, 18b, 24, 27a, 27b, 30, 36a..36d, or LxM cells) or 'ring'")
    ap.add_argument("n", nargs="?", type=int, help="ring size (with 'ring')")
    ap.add_argument("--starts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--J", type=float, default=1.0, help="coupling of the up-triangle bonds (> 0; default 1)")
    ap.add_argument("--Jdown", type=float, default=None, help="coupling of the down-triangle bonds (default: same as --J)")
    ap.add_argument("--J2", type=float, default=0.0, help="next-nearest-neighbour coupling of the ring (ring only)")
    ap.add_argument("--ed", action="store_true", help="compare with exact diagonalisation (N <= 24)")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    if a.cluster == "ring":
        if a.n is None:
            raise SystemExit("ring needs a size: kagome_gaussian_S1.py ring 10")
        N, (bonds, w), torus = a.n, ring_graph(a.n, a.J2), None
        print(f"J1-J2 ring, N = {N}, J2 = {a.J2:g}")
    else:
        torus = kl.torus_from_name(a.cluster)
        Jd = a.J if a.Jdown is None else a.Jdown
        N, bonds, w = torus.N, torus.bonds, torus.couplings(a.J, Jd)
        print(torus.summary())
    if a.J <= 0:
        raise SystemExit("J must be > 0")
    t0 = time.time()
    r = minimise(N, bonds, w, a.starts, a.seed, a.verbose)
    ed = exact_energy(N, bonds, w)[0] if a.ed else None
    describe(r, torus, ed)
    if torus is not None and a.cluster in LITERATURE_ED and a.J == 1.0 and (a.Jdown in (None, 1.0)):
        lit, ver = LITERATURE_ED[a.cluster]
        print(f"  literature ED (Laeuchli et al.): E0 = {lit:.9f}" + ("" if ver else "   (as extracted from the paper's table, not reproduced here)"))
    print(f"  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
