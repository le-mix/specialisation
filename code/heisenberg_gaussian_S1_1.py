#!/usr/bin/env python3
"""
Gaussian (6-Majorana) variational minimiser for the spin-1/2 J1-J2 Heisenberg ring

        H = J1 * sum_{i=1..N} S_i . S_{i+1}  +  J2 * sum_{i=1..N} S_i . S_{i+2}        (periodic;  N = "inf" for the thermodynamic limit)

(J1 = --J, default 1;  J2 = --J2, default 0, which is the nearest-neighbour ring of the previous version.  For N = 4 the next-nearest-neighbour
sum counts every diagonal twice, for N = 3 it is a second copy of the nearest-neighbour sum: H is always the sum over i written above.)

First slice S_1: time-reversal class TZ (x = y = 0) with a one-site unit cell (translation invariant).

State (per colour alpha = x,y,z; the three colours are identical copies, so the state is rotation invariant):
    <rho_i eta_j> = -i p_ij ,   <rho_i rho_j> = <eta_i eta_j> = delta_ij ,   p_ij = p(i-j) real (circulant).
Fourier data   phat(k) = sum_r p(r) e^{-ikr},   p(r) = (1/N) sum_k phat(k) e^{ikr},   k = 2 pi (n + twist/2) / N,
where twist = 0 / 1 means periodic / antiperiodic correlations (flux 0 / pi through the ring).
Constraint (necessary and sufficient for the state to exist):  |phat(k)| <= 1  and  phat(-k) = conj(phat(k)).

The energy of a bond between two sites at distance r depends only on the 2x2 block of p on these two sites,
    a = p(0) = <i rho_i eta_i> ,   b_r = p(-r) = <i rho_i eta_{i+r}> ,   c_r = p(+r) = <i rho_{i+r} eta_i> ,
    e(a,b,c,d) = 3/16 [ -(b^2+c^2) - (a-d)(b^2-c^2) + (b^2+c^2)(a d - b c) ]      (<S_i.S_j>, x=y=0, same formula for every distance),
so that, with d = a (translation invariance),
    E / N = J1 e(a, b_1, c_1, a) + J2 e(a, b_2, c_2, a).
At a minimum phat(k) = -K(k)/|K(k)| with the mean-field couplings (per colour), (E_a, E_b1, ... = partial derivatives of E/N)
    K(k) = mu + sum_{r=1,2} [ 2 t_r cos(r k) + 2 i kappa_r sin(r k) ] ,
    mu = E_a / 3 ,   t_r = (E_br + E_cr) / 6 ,   kappa_r = (E_br - E_cr) / 6 ,
i.e. phat is the ground state of
    H_MF = sum_{alpha,i} [ mu i rho_i eta_i + sum_r ( (t_r + kappa_r) i rho_i eta_{i+r} + (t_r - kappa_r) i rho_{i+r} eta_i ) ].
The script prints the residual max_k | phat + K/|K| | as a self-consistency (stationarity) check.

Usage:  python heisenberg_gaussian_S1.py 40                    # 40-site ring, both fluxes
        python heisenberg_gaussian_S1.py 12 --J2 0.5 --ed      # J2/J1 = 1/2 (Majumdar-Ghosh point), compared with exact diagonalisation
        python heisenberg_gaussian_S1.py inf --J2 0.3          # thermodynamic limit (grid of --grid points)
"""
import argparse

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------------------------------------------
# Energy functional
# ----------------------------------------------------------------------------------------------------------
def bond(a, b, c, d):
    """Bond energy e = <S_1.S_2> (TZ class; any distance) and its gradient (e_a, e_b, e_c, e_d).
    a = <i rho_1 eta_1>, b = <i rho_1 eta_2>, c = <i rho_2 eta_1>, d = <i rho_2 eta_2> (one fixed colour)."""
    q = b * b + c * c
    e = 3 / 16 * (-q - (a - d) * (b * b - c * c) + q * (a * d - b * c))
    ea = 3 / 16 * (-(b * b - c * c) + d * q)
    ed = 3 / 16 * ((b * b - c * c) + a * q)
    eb = 3 / 16 * (-2 * b * (1 + (a - d) - (a * d - b * c)) - c * q)
    ec = 3 / 16 * (-2 * c * (1 - (a - d) - (a * d - b * c)) - b * q)
    return e, ea, eb, ec, ed


def etilde(a, b, c):
    """Nearest-neighbour energy per site of a translation-invariant state (all bonds equal, d = a) and its gradient (a, b, c)."""
    e, ea, eb, ec, ed = bond(a, b, c, a)
    return e, ea + ed, eb, ec


def site_energy(m, g):
    """E/(N J1) of a translation-invariant state with moments m = (a, b1, c1, b2, c2), g = J2/J1, and the gradient with respect to m."""
    a, b1, c1, b2, c2 = m
    e, ea, eb, ec, ed = bond(a, b1, c1, a)
    grad = np.array([ea + ed, eb, ec, 0.0, 0.0])
    if g != 0.0:
        e2, ea2, eb2, ec2, ed2 = bond(a, b2, c2, a)
        e = e + g * e2
        grad = grad + g * np.array([ea2 + ed2, 0.0, 0.0, eb2, ec2])
    return e, grad


# ----------------------------------------------------------------------------------------------------------
# Ring of N sites: momentum grid and parametrisation phat_k = s_k e^{i theta_k}
# ----------------------------------------------------------------------------------------------------------
class Ring:
    def __init__(self, N, twist=0, g=0.0):
        n = np.arange(N)
        self.N, self.twist, self.g = N, twist, g
        self.k = 2 * np.pi * (n + twist / 2) / N
        partner = (-n - twist) % N                  # index of the momentum -k
        self.ip = n[n < partner]                    # one representative of each pair (k, -k)
        self.jp = partner[self.ip]                  # ... and its partner
        self.isc = n[n == partner]                  # self-conjugate momenta (k = 0 or pi): phat real
        self.P = len(self.ip)
        self.em = np.exp(-1j * self.k)              # e^{-ik}
        self.em2 = self.em * self.em                # e^{-2ik}

    def phat(self, x):
        """Parameters x = [theta (P), s (P), u (#self-conjugate)] -> phat on the full grid."""
        th, s, u = x[:self.P], x[self.P:2 * self.P], x[2 * self.P:]
        z = np.empty(self.N, complex)
        z[self.ip] = s * np.exp(1j * th)
        z[self.jp] = np.conj(z[self.ip])
        z[self.isc] = u
        return z

    def moments(self, z):
        """a = (1/N) sum phat,  b_r = (1/N) sum phat e^{-irk},  c_r = (1/N) sum phat e^{+irk}   (r = 1, 2)."""
        N = self.N
        return np.array([(z.sum() / N).real, ((z * self.em).sum() / N).real, ((z * self.em.conj()).sum() / N).real,
                         ((z * self.em2).sum() / N).real, ((z * self.em2.conj()).sum() / N).real])

    def weights(self, gm):
        """L_k = dE/dphat_k (complex weights): L_k = E_a + E_b1 e^{-ik} + E_c1 e^{ik} + E_b2 e^{-2ik} + E_c2 e^{2ik}."""
        return gm[0] + gm[1] * self.em + gm[2] * self.em.conj() + gm[3] * self.em2 + gm[4] * self.em2.conj()

    def objective(self, x):
        """F = E/J1 = N e(m) and dF/dx. Since dF = Re sum_k dphat_k L_k  (the pair k, -k contributes twice)."""
        e, gm = site_energy(self.moments(self.phat(x)), self.g)
        L = self.weights(gm)
        th, s = x[:self.P], x[self.P:2 * self.P]
        w = np.exp(1j * th) * L[self.ip]
        grad = np.concatenate([-2 * s * w.imag, 2 * w.real, L[self.isc].real])
        return self.N * e, grad

    def start(self, rng):
        """Start from the ground state of a random mean-field Hamiltonian (random direction of mu : t_r : kappa_r)."""
        if self.g == 0.0:                           # nearest neighbours only: same random sequence as the previous version
            mu, t, ka = rng.normal(size=3)
            K = mu + 2 * t * np.cos(self.k) + 2j * ka * np.sin(self.k)
        else:
            mu, t, ka, t2, ka2 = rng.normal(size=5)
            K = (mu + 2 * t * np.cos(self.k) + 2j * ka * np.sin(self.k)
                 + 2 * t2 * np.cos(2 * self.k) + 2j * ka2 * np.sin(2 * self.k))
        K = K + 1e-6 * (rng.normal(size=self.N) + 1j * rng.normal(size=self.N))   # lift accidental zeros
        ph = -K / np.abs(K)
        u = np.where(ph[self.isc].real >= 0, 1.0, -1.0)
        return np.concatenate([np.angle(ph[self.ip]), np.ones(self.P), u])


# ----------------------------------------------------------------------------------------------------------
# Minimiser and analysis
# ----------------------------------------------------------------------------------------------------------
def analyse(R, x, J=1.0):
    """Energies and mean-field couplings (units of J = J1) of the parameter vector x."""
    z = R.phat(x)
    m = R.moments(z)
    e, gm = site_energy(m, R.g)
    mu = J * gm[0] / 3
    t = (J * (gm[1] + gm[2]) / 6, J * (gm[3] + gm[4]) / 6)
    ka = (J * (gm[1] - gm[2]) / 6, J * (gm[3] - gm[4]) / 6)
    K = mu + 2 * t[0] * np.cos(R.k) + 2j * ka[0] * np.sin(R.k) + 2 * t[1] * np.cos(2 * R.k) + 2j * ka[1] * np.sin(2 * R.k)
    ok = np.abs(K) > 1e-9
    resid = np.max(np.abs(z[ok] + K[ok] / np.abs(K[ok]))) if ok.any() else 0.0
    a, b, c, b2, c2 = m
    return dict(N=R.N, twist=R.twist, J=J, J2=R.g * J, E=J * R.N * e, E_per_site=J * e, a=a, b=b, c=c, b2=b2, c2=c2,
                m1=(b + c) / 2, m2=(b - c) / 2, mu=mu, tR=t[0], kappa=ka[0], tR2=t[1], kappa2=ka[1], t=t, kap=ka,
                gap=2 * np.abs(K).min(), residual=resid, min_abs_phat=np.abs(z).min(), phat=z, k=R.k)


def minimise(N, twist=0, starts=24, seed=0, J=1.0, J2=0.0):
    """Minimise E over all |phat| <= 1 (translation-invariant TZ states) on an N-site ring; best of `starts` runs."""
    R = Ring(N, twist, J2 / J)
    rng = np.random.default_rng(seed)
    bounds = [(None, None)] * R.P + [(0.0, 1.0)] * R.P + [(-1.0, 1.0)] * len(R.isc)
    best = None
    for _ in range(starts):
        r = minimize(R.objective, R.start(rng), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=50000, maxfun=100000, maxcor=30, ftol=1e-15, gtol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    out = analyse(R, best.x, J)
    out["iterations"] = best.nit
    return out


def exact_energy(N, J=1.0, J2=0.0):
    """Ground-state energy of the periodic spin-1/2 J1-J2 ring by Lanczos (full 2^N basis, N <= ~20)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    s = np.arange(2 ** N, dtype=np.int64)
    diag = np.zeros(2 ** N)
    rows, cols, vals = [], [], []
    for dist, Jd in ((1, J), (2, J2)):
        if Jd == 0.0:
            continue
        for i in range(N):
            j = (i + dist) % N
            if j == i:
                continue
            bi, bj = (s >> i) & 1, (s >> j) & 1
            diag += 0.25 * Jd * np.where(bi == bj, 1.0, -1.0)                      # S^z_i S^z_j
            f = s[bi != bj]
            rows.append(f); cols.append(f ^ ((1 << i) | (1 << j))); vals.append(np.full(f.size, 0.5 * Jd))   # (S+S- + S-S+)/2
    rows.append(s); cols.append(s); vals.append(diag)
    H = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 ** N,) * 2).tocsr()
    return eigsh(H, k=1, which="SA")[0][0]


def show(r, inf=False):
    if inf:
        print(f"N -> infinity (k grid of {r['N']} points),  J1 = {r['J']:g}, J2 = {r['J2']:g}")
    else:
        print(f"N = {r['N']}, J1 = {r['J']:g}, J2 = {r['J2']:g}, twist = {r['twist']} ({'periodic' if r['twist'] == 0 else 'antiperiodic'} correlations)")
    print(f"  E/N               = {r['E_per_site']:.10f}      (E = {r['E']:.10f})")
    print(f"  a, b1, c1         = {r['a']:.8f}, {r['b']:.8f}, {r['c']:.8f}")
    if r["J2"] != 0.0:
        print(f"  b2, c2            = {r['b2']:.8f}, {r['c2']:.8f}")
    print(f"  m1=(b+c)/2, m2=(b-c)/2 = {r['m1']:.8f}, {r['m2']:.8f}   (twice the hopping / pairing correlator per colour)")
    print(f"  mu, tR, kappa     = {r['mu']:.8f}, {r['tR']:.8f}, {r['kappa']:.8f}   (units of J1)")
    if r["J2"] != 0.0:
        print(f"  tR2, kappa2       = {r['tR2']:.8f}, {r['kappa2']:.8f}   (distance-2 couplings of the mean-field Hamiltonian)")
    print(f"  gap 2 min|K|      = {r['gap']:.6f},   min|phat| = {r['min_abs_phat']:.6f},   "
          f"residual |phat + K/|K|| = {r['residual']:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", help='number of sites (integer >= 3) or "inf"')
    ap.add_argument("--twist", choices=["0", "1", "both"], default="both", help="periodic / antiperiodic correlations")
    ap.add_argument("--starts", type=int, default=24, help="number of random starting states")
    ap.add_argument("--grid", type=int, default=4096, help='number of k points used for N = "inf"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--J", type=float, default=1.0, help="nearest-neighbour coupling J1 (> 0)")
    ap.add_argument("--J2", type=float, default=0.0, help="next-nearest-neighbour coupling J2 (default 0)")
    ap.add_argument("--ed", action="store_true", help="compare with exact diagonalisation (N <= ~20)")
    args = ap.parse_args()

    inf = args.N.lower() in ("inf", "infinity")
    try:
        N = args.grid if inf else int(args.N)
    except ValueError:
        raise SystemExit('N must be an integer >= 3 or "inf"')
    if args.J <= 0:
        raise SystemExit("J1 must be > 0 (antiferromagnetic nearest-neighbour coupling): the minimiser works with E/J1")
    if N < 3:
        raise SystemExit("N must be >= 3 (a 2-site ring double counts its bond; use bond(0,1,1,0) = -3/4 for N = 2)")
    twists = [0] if inf else ([0, 1] if args.twist == "both" else [int(args.twist)])
    res = [minimise(N, t, args.starts, args.seed, args.J, args.J2) for t in twists]
    for r in res:
        show(r, inf)
    best = min(res, key=lambda r: r["E"])
    if inf or N % 2 == 0:
        print(f"reference: dimer product {-3 / 8 * args.J:.6f} per site (independent of J2: next-nearest neighbours lie in different dimers)"
              + (f",  exact (Bethe ansatz, J2 = 0) {(0.25 - np.log(2)) * args.J:.6f} per site" if (inf and args.J2 == 0) else ""))
    if abs(args.J2 - 0.5 * args.J) < 1e-12 and (inf or N % 2 == 0):
        print("   J2 = J1/2 is the Majumdar-Ghosh point: for even N the dimer product states are exact ground states, E = -3 N J1 / 8")
    if args.ed:
        E0 = exact_energy(N, args.J, args.J2)
        print(f"exact E0 = {E0:.10f}   (E0/N = {E0 / N:.10f});  best Gaussian - exact = {best['E'] - E0:.3e}  (must be >= 0)")
