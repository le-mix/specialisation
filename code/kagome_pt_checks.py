#!/usr/bin/env python3
"""
Brute-force perturbation theory around a TZ Gaussian reference state of the 6-Majorana representation, for the spin-1/2 Heisenberg model on an ARBITRARY
small graph (the kagome counterpart of heisenberg_pt_checks.py; the chain is the special case "ring").

        H = sum_{bonds (i,j)} J_ij S_i . S_j ,       reference H0 = sum_{a = x,y,z} sum_{ij} K_ij  i rho^a_i eta^a_j ,   K a real N x N matrix.

Everything is done in the FULL Fock space (8^N = 2^(3N) dimensions, no Wick / diagram shortcuts), so N <= 6 (262144 dimensions), and every number can be
compared with exact diagonalisation and with kagome_diagrams.py, which computes the same series with Wick/Pfaffian diagrams.

Series.  H(xi) = H0 + xi V,  V = H - H0,  physical point xi = 1:  E(xi) = sum_n E_n xi^n  (Rayleigh-Schroedinger, intermediate normalisation),
    E_n = <0| V |psi_{n-1}>,   (H0 - E0) |psi_n> = - Q [ V |psi_{n-1}> - sum_{k=1}^{n-1} E_k |psi_{n-k}> ],   Q = 1 - |0><0|,
with |0> the (non-degenerate) ground state of H0.  E_0 + E_1 = <0|H|0> is the (variational) mean-field energy.  The resolvent is applied by conjugate gradients.
The spin operator of a site from its six Majoranas is the one of heisenberg_pt_checks.spin_6.

Graphs (--graph):  triangle | bowtie (two triangles sharing a vertex, N = 5) | ring N [--J2 x] | torus L1xL2 (kagome tori with 2 cells or fewer: N = 3, 6)
Reference (--reference):  auto (default: the best dimer covering by E_1 + E_2, ties by E_1 + E_2 + E_3, for even N; random for odd N)  |  dimer  |  minimised (the lowest Gaussian state of kagome_gaussian_S1; it
   has zero modes for odd N, then the program says so and falls back to random)  |  random (a random K with the support of the graph)

Requires numpy, scipy, heisenberg_pt_checks.py, heisenberg_gaussian_S1.py, kagome_gaussian_S1.py and kagome_lattice.py in the same directory.
Usage:  python3 kagome_pt_checks.py --graph triangle
        python3 kagome_pt_checks.py --graph ring 6 --order 4 --compare      # N = 6, three dimers, Fock dimension 262144, about 20 s; --compare adds the diagram engine
                                                                           # (E_2, E_3, E_4 agree with the brute force to 1e-12)
        python3 kagome_pt_checks.py --graph torus 2x1 --order 3 --compare   # the 6-site kagome torus (its dimer product is the exact ground state: E_n = 0 for n >= 2)
        python3 kagome_pt_checks.py --graph bowtie --reference random --compare
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heisenberg_pt_checks as pc
import kagome_gaussian_S1 as kg
import kagome_lattice as kl


def make_graph(spec, J2=0.0):
    """(N, bonds, weights, label, perms) for a graph specification: 'triangle', 'bowtie', 'ring:N', or 'torus:L1xL2' / a kagome cluster name.
       perms = lattice symmetries as site permutations (the identity only for the test graphs)."""
    s = spec.lower()
    if s == "triangle":
        return 3, np.array([[0, 1], [0, 2], [1, 2]]), np.ones(3), "triangle (N = 3)", [np.arange(3)]
    if s == "bowtie":
        return 5, np.array([[0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4]]), np.ones(6), "bowtie (two triangles sharing a site, N = 5)", [np.arange(5)]
    if s.startswith("ring:"):
        n = int(s.split(":")[1])
        b, w = kg.ring_graph(n, J2)
        perms = [(np.arange(n) + k) % n for k in range(n)] + [(-np.arange(n) + k) % n for k in range(n)]
        return n, b, w, f"ring N = {n}, J2 = {J2:g}", perms
    name = s.split(":")[1] if s.startswith("torus:") else s
    T = kl.torus_from_name(name)
    return T.N, T.bonds, T.couplings(), f"kagome torus {name} (N = {T.N})", T.all_symmetries()


def graph_hamiltonian(S, bonds, weights):
    """sum_b w_b S_i . S_j for the list S of per-site spin matrices S[i][a]."""
    H = None
    for (i, j), w in zip(np.asarray(bonds, int), np.asarray(weights, float)):
        for a in range(3):
            t = w * (S[i][a] @ S[j][a])
            H = t if H is None else H + t
    return H.tocsr()


def mean_field_K(gam, K):
    """H0 = sum_{a,i,j} K_ij  i rho^a_i eta^a_j   (rho^a_i = gam[6 i + a], eta^a_j = gam[6 j + 3 + a])."""
    N = K.shape[0]
    H = None
    for a in range(3):
        for i in range(N):
            for j in range(N):
                if K[i, j] != 0:
                    t = K[i, j] * (1j * gam[6 * i + a] @ gam[6 * j + 3 + a])
                    H = t if H is None else H + t
    return H.tocsr()


def random_reference(N, bonds, rng, scale=1.0):
    """A random gapped K with the support of the graph (on-site terms and both directions of every bond)."""
    K = 0.01 * rng.normal(size=(N, N))
    K[np.arange(N), np.arange(N)] += scale * rng.normal(size=N)
    for i, j in np.asarray(bonds, int):
        K[i, j] += scale * rng.normal()
        K[j, i] += scale * rng.normal()
    return K


def fock_series(N, bonds, weights, K, order, gam=None, H=None, check_xi=None):
    """E_0..E_order of the Rayleigh-Schroedinger series about the ground state of H0(K) in the full Fock space."""
    if gam is None:
        gam = pc.majoranas(3 * N)
    if H is None:
        H = graph_hamiltonian([pc.spin_6(gam[6 * i:6 * i + 6]) for i in range(N)], bonds, weights)
    H0 = mean_field_K(gam, K)
    w, v = pc.lowest_two(H0)
    psi0, E0 = v[:, 0], w[0]
    V = (H - H0).tocsr()
    Es = pc.rs_series(H0, V, psi0, E0, order)
    res = dict(E=Es, gap=w[1] - w[0], MF=np.vdot(psi0, H @ psi0).real)
    if check_xi is not None:
        res["check"] = (pc.lowest_eigenvalue((H0 + check_xi * V).tocsr()), sum(Es[k] * check_xi ** k for k in range(order + 1)))
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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--graph", nargs="+", default=["triangle"], help="triangle | bowtie | ring N | torus L1xL2 | a kagome cluster name (default triangle)")
    ap.add_argument("--J2", type=float, default=0.0, help="ring only")
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--reference", choices=["auto", "dimer", "minimised", "random"], default="auto",
                    help="auto: the best dimer covering (lowest E_1 + E_2, ties: E_1 + E_2 + E_3) for even N, a random K for odd N")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compare", action="store_true", help="also run the diagram engine (kagome_diagrams.py) on the same reference and print the difference")
    a = ap.parse_args()
    spec = a.graph[0] if a.graph[0] in ("triangle", "bowtie") or len(a.graph) == 1 else (a.graph[0] + ":" + a.graph[1])
    N, bonds, w, label, perms = make_graph(spec, a.J2)
    if N > 6:
        raise SystemExit("the Fock space has 8^N dimensions: N <= 6")
    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    kind = a.reference
    if kind == "auto":
        kind = "dimer" if N % 2 == 0 else "random"
    if kind == "minimised":
        r = kg.minimise(N, bonds, w, starts=24, seed=a.seed)
        if r["gap"] < 1e-6:
            print(f"the lowest Gaussian state of this graph has zero modes (E = {r['E']:.10f}): perturbation theory around it is undefined; using a random reference instead")
            kind = "random"
        else:
            K = r["K"]
            how = f"lowest-energy Gaussian state of the graph (minimiser E = {r['E']:.10f}, gap {r['gap']:.4f}, aufbau {r['aufbau']})"
    if kind == "dimer":
        import kagome_diagrams as kd
        rows, exhaustive = kd.covering_table(N, bonds, w, perms, ncover=1, seed=a.seed)
        K = kd.covering_reference(N, bonds, w, [tuple(p) for p in rows[0]["pairs"]]).K
        how = f"dimer covering {[tuple(p) for p in rows[0]['pairs']]} (lowest E_1 + E_2, ties: E_1 + E_2 + E_3, of {len(rows)} inequivalent coverings)"
    if kind == "random":
        K = random_reference(N, bonds, rng)
        how = "random K with the support of the graph"
    Eex = kg.exact_energy(N, bonds, w)[0]
    res = fock_series(N, bonds, w, K, a.order, check_xi=0.05)
    print(f"{label}: reference = {how}")
    print(f"gap(H0) = {res['gap']:.5f};  mean-field energy <H>_0 = {res['MF']:.10f};  exact ED = {Eex:.10f}   (Fock dimension {8 ** N})")
    report(res, Eex)
    if a.compare:
        import kagome_diagrams as kd
        gref = kd.GraphReference(K)
        top = min(a.order, 4 if gref.factorised else 3)       # the dense 16-leg tables make order 4 impractical for a reference with a single class
        d = kd.series(gref, bonds, w, top, perms, 1e-6)
        print(f"   diagram engine (Wick / Pfaffian) on the same reference, orders 2-{top}" + ("" if top == a.order else " (order 4 skipped: single-class reference, dense path)") + ":")
        worst = 0.0
        for n in range(2, top + 1):
            worst = max(worst, abs(d["E"][n] - res["E"][n]))
            print(f"   order {n}: E_n = {d['E'][n]:+.10f}   (Fock {res['E'][n]:+.10f})   difference {d['E'][n] - res['E'][n]:+.1e}")
        print(f"   largest difference {worst:.1e}")
    print(f"({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
