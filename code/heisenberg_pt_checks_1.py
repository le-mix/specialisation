#!/usr/bin/env python3
"""
Brute-force perturbation theory around the TZ Gaussian state of the 6-Majorana representation for the spin-1/2 J1-J2 Heisenberg ring
    H = sum_i S_i . S_{i+1} + J2 sum_i S_i . S_{i+2}     (J1 = 1, periodic, N sites; J2 = 0 is the nearest-neighbour ring).

Everything is done in the FULL Fock space (2^(3N) dimensions, no Wick / diagram shortcuts), so every number can be compared with exact
diagonalisation and with heisenberg_diagrams.py, which computes the same series with Wick/Pfaffian diagrams.

Reference.  heisenberg_gaussian_S1.py minimises <H> over TZ Gaussian states (one-site cell, time-reversal class "TZ"):
    H0 = sum_{alpha=x,y,z} sum_i [ mu i rho^alpha_i eta^alpha_i
           + sum_{r=1,2} ( (t_r+kappa_r) i rho^alpha_i eta^alpha_{i+r} + (t_r-kappa_r) i rho^alpha_{i+r} eta^alpha_i ) ],
with (mu, t_r, kappa_r) the optimum and the coupling of every bond that crosses the seam multiplied by (-1)^twist.  |0> is the (non-degenerate)
ground state of H0, energy E0.  (For J2 < J1 the optimum has t_2 = kappa_2 = 0: the reference is the nearest-neighbour one and J2 enters through V only.)

Series.  H(xi) = H0 + xi V,  V = H - H0,  physical point xi = 1:  E(xi) = sum_n E_n xi^n  (Rayleigh-Schroedinger, intermediate normalisation):
    E_n = <0| V |psi_{n-1}>,     (H0 - E0) |psi_n> = - Q [ V |psi_{n-1}> - sum_{k=1}^{n-1} E_k |psi_{n-k}> ],     Q = 1 - |0><0|.
E_0 + E_1 = <0|H|0> is the (variational) mean-field energy.  The reduced resolvent is applied with conjugate gradients to the Hermitian
positive-definite operator H0 - E0 + |0><0|.  The printed "check" compares the series with the exact lowest eigenvalue of H0 + 0.05 V; the
printed exact energy is the spin-ring ED (heisenberg_gaussian_S1.exact_energy).

Spin operator of one site from its six Majoranas g = (rho^x, rho^y, rho^z, eta^x, eta^y, eta^z):
    S = (J_rho + J_eta)/2 + Gamma (J_rho - J_eta)/2,   J^a = -(i/2) chi^b chi^c,   Gamma = i rho^x rho^y rho^z eta^x eta^y eta^z.

Requires numpy, scipy and heisenberg_gaussian_S1.py in the same directory.
Usage:  python3 heisenberg_pt_checks.py                    # N = 3 4 5, both seam signs, orders up to 8
        python3 heisenberg_pt_checks.py 4 6 --order 6      # N = 6 has Fock dimension 262144 (about 10-30 s per seam sign)
        python3 heisenberg_pt_checks.py 5 --twist 0
        python3 heisenberg_pt_checks.py 4 5 6 --J2 0.4     # J1-J2 ring
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heisenberg_gaussian_S1 as hg

# ------------------------------------------------------------------------------------------------ Majorana operators (Jordan-Wigner)
_X = sp.csr_matrix(np.array([[0, 1], [1, 0]], complex))
_Y = sp.csr_matrix(np.array([[0, -1j], [1j, 0]]))
_Z = sp.csr_matrix(np.diag([1.0, -1.0]).astype(complex))
_I = sp.identity(2, dtype=complex, format="csr")


def majoranas(nmodes):
    """The 2*nmodes sparse Majorana operators, {g_a, g_b} = 2 delta_ab."""
    out = []
    for j in range(nmodes):
        for P in (_X, _Y):
            m = sp.identity(1, dtype=complex, format="csr")
            for o in [_Z] * j + [P] + [_I] * (nmodes - j - 1):
                m = sp.kron(m, o, format="csr")
            out.append(m)
    return out


def spin_6(g):
    """Spin operators of one site from its six Majoranas g = (rho^x, rho^y, rho^z, eta^x, eta^y, eta^z):
       S = (J_rho + J_eta)/2 + Gamma (J_rho - J_eta)/2,   J^a = -(i/2) chi^b chi^c,   Gamma = i rho^x rho^y rho^z eta^x eta^y eta^z."""
    r, e = g[:3], g[3:]
    Jr = [-0.5j * r[(a + 1) % 3] @ r[(a + 2) % 3] for a in range(3)]
    Je = [-0.5j * e[(a + 1) % 3] @ e[(a + 2) % 3] for a in range(3)]
    Gam = 1j * r[0] @ r[1] @ r[2] @ e[0] @ e[1] @ e[2]
    return [0.5 * (Jr[a] + Je[a]) + 0.5 * Gam @ (Jr[a] - Je[a]) for a in range(3)]


def ring_hamiltonian(S, N, J2=0.0):
    """sum_i S_i . S_{i+1} + J2 sum_i S_i . S_{i+2} (indices mod N, i = 0..N-1); S[i] = list of the three spin matrices of site i."""
    H = None
    for dist, Jd in ((1, 1.0), (2, J2)):
        if Jd == 0.0:
            continue
        for i in range(N):
            j = (i + dist) % N
            for a in range(3):
                t = Jd * (S[i][a] @ S[j][a])
                H = t if H is None else H + t
    return H.tocsr()


def mean_field_6(gam, N, mu, tR, kappa, twist):
    """H0 = sum_{alpha,i} [mu i rho_i eta_i + sum_r ((t_r+kappa_r) i rho_i eta_{i+r} + (t_r-kappa_r) i rho_{i+r} eta_i)]; every bond that crosses the
       seam (i + r >= N) is multiplied by (-1)^twist.  tR, kappa: scalars (r = 1 only) or sequences (t_1, t_2), (kappa_1, kappa_2)."""
    sgn = 1.0 if twist == 0 else -1.0
    tR, kappa = np.atleast_1d(np.asarray(tR, float)), np.atleast_1d(np.asarray(kappa, float))
    K = np.zeros((N, N))
    for i in range(N):
        K[i, i] += mu
    for r in range(1, len(tR) + 1):
        for i in range(N):
            j = (i + r) % N
            s = sgn if i + r >= N else 1.0
            K[i, j] += s * (tR[r - 1] + kappa[r - 1])
            K[j, i] += s * (tR[r - 1] - kappa[r - 1])
    H = None
    for al in range(3):
        for i in range(N):
            for j in range(N):
                if K[i, j] != 0:
                    t = K[i, j] * (1j * gam[6 * i + al] @ gam[6 * j + 3 + al])
                    H = t if H is None else H + t
    return H.tocsr()


# ------------------------------------------------------------------------------------------------ Rayleigh-Schroedinger series
def lowest_two(H):
    if H.shape[0] <= 1024:
        w, v = np.linalg.eigh(H.toarray())
        return w[:2], v[:, :2]
    w, v = sla.eigsh(H, k=2, which="SA", tol=1e-13)
    o = np.argsort(w)
    return w[o], v[:, o]


def lowest_eigenvalue(H):
    if H.shape[0] <= 1024:
        return np.linalg.eigvalsh(H.toarray())[0]
    return sla.eigsh(H, k=1, which="SA", tol=1e-13)[0][0]


def cg_solve(B, b):
    try:
        x, info = sla.cg(B, b, rtol=1e-13, atol=0.0, maxiter=5000)
    except TypeError:                                   # older scipy
        x, info = sla.cg(B, b, tol=1e-13, atol=0.0, maxiter=5000)
    if info != 0:
        raise RuntimeError(f"CG did not converge (info={info})")
    return x


def rs_series(H0, V, psi0, E0, K):
    """E_0..E_K, the Taylor coefficients in xi of the lowest eigenvalue of H0 + xi V."""
    n = H0.shape[0]
    B = sla.LinearOperator((n, n), dtype=complex, matvec=lambda v: H0 @ v - E0 * v + psi0 * np.vdot(psi0, v))   # H0 - E0 + |0><0|
    psis, Vpsi, Es = [psi0], [V @ psi0], [E0]
    for order in range(1, K + 1):
        Es.append(np.vdot(psi0, Vpsi[order - 1]).real)                       # E_n = <0|V|psi_{n-1}>
        rhs = Vpsi[order - 1].copy()
        for k in range(1, order):
            rhs -= Es[k] * psis[order - k]
        rhs -= psi0 * np.vdot(psi0, rhs)                                      # project with Q
        psis.append(-cg_solve(B, rhs))                                        # psi_n = -(H0 - E0)^{-1} Q [...]
        Vpsi.append(V @ psis[-1])
    return Es


def tz_series(N, twist, K, starts=12, seed=0, H=None, gam=None, check_xi=None, J2=0.0):
    """Full-Fock-space series E_0..E_K about the TZ reference (seam sign (-1)^twist).  Returns a dict with the series, the reference
       parameters, the mean-field energy and (optionally) the exact lowest eigenvalue of H0 + check_xi V for comparison with the series."""
    if gam is None:
        gam = majoranas(3 * N)
    if H is None:
        H = ring_hamiltonian([spin_6(gam[6 * i:6 * i + 6]) for i in range(N)], N, J2)
    out = hg.minimise(N, twist, starts=starts, seed=seed, J2=J2)
    H0 = mean_field_6(gam, N, out["mu"], out["t"], out["kap"], twist)
    w, v = lowest_two(H0)
    psi0, E0 = v[:, 0], w[0]
    V = (H - H0).tocsr()
    Es = rs_series(H0, V, psi0, E0, K)
    res = dict(E=Es, gap=w[1] - w[0], MF=np.vdot(psi0, H @ psi0).real, MF_minimiser=out["E"], params=(out["mu"], out["t"], out["kap"]))
    if check_xi is not None:
        res["check"] = (lowest_eigenvalue((H0 + check_xi * V).tocsr()), sum(Es[k] * check_xi ** k for k in range(K + 1)))
    return res


def report(res, E_exact, xi=0.05):
    Es = res["E"]
    K = len(Es) - 1
    if "check" in res:
        print(f"   check: lowest eigenvalue of H0 + {xi} V = {res['check'][0]:.12f};  series to order {K} = {res['check'][1]:.12f}")
    S = Es[0] + Es[1]
    err = lambda s: 100 * (s - E_exact) / abs(E_exact)
    print(f"   order 1 (= <H>_0, mean-field energy): {S:.10f}   {err(S):+6.1f}% from exact")
    for n in range(2, K + 1):
        S += Es[n]
        print(f"   order {n}: E_n = {Es[n]:+.8f}   partial sum = {S:.10f}   {err(S):+6.1f}% from exact")


def run_tz(Ns, K, twists, J2=0.0):
    for N in Ns:
        if not 3 <= N <= 6:
            raise SystemExit("N must be 3, 4, 5 or 6 (the Fock space has 2^(3N) dimensions)")
        t0 = time.time()
        gam = majoranas(3 * N)
        H = ring_hamiltonian([spin_6(gam[6 * i:6 * i + 6]) for i in range(N)], N, J2)
        Eex = hg.exact_energy(N, 1.0, J2)
        for tw in twists:
            res = tz_series(N, tw, K, H=H, gam=gam, check_xi=0.05, J2=J2)
            print(f"\nN={N}, J2={J2:g}, twist={tw} (dim {H.shape[0]}): gap(H0) = {res['gap']:.5f};  mean-field energy <H>_0 = {res['MF']:.10f} "
                  f"(minimiser: {res['MF_minimiser']:.10f});  exact ED = {Eex:.10f}")
            report(res, Eex)
            sys.stdout.flush()
        print(f"   ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("N", nargs="*", type=int, help="ring sizes, 3 <= N <= 6 (default 3 4 5)")
    ap.add_argument("--order", type=int, default=8, help="highest order in xi (default 8)")
    ap.add_argument("--twist", choices=["0", "1", "both"], default="both", help="seam sign of the reference, (-1)^twist (default both)")
    ap.add_argument("--J2", type=float, default=0.0, help="next-nearest-neighbour coupling J2 (J1 = 1; default 0)")
    args = ap.parse_args()
    run_tz(args.N or [3, 4, 5], args.order, (0, 1) if args.twist == "both" else (int(args.twist),), args.J2)
