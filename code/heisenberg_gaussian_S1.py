#!/usr/bin/env python3
"""
Gaussian (6-Majorana) variational minimiser for the nearest-neighbour spin-1/2 Heisenberg ring

        H = J * sum_{i=1..N} S_i . S_{i+1}       (N >= 3 sites, periodic;  N = "inf" for the thermodynamic limit)

in the first slice S_1: time-reversal class TZ (x = y = 0) with a one-site unit cell (translation invariant).

State (per colour alpha = x,y,z; the three colours are identical copies, so the state is rotation invariant):
    <rho_i eta_j> = -i p_ij ,   <rho_i rho_j> = <eta_i eta_j> = delta_ij ,   p_ij = p(i-j) real (circulant).
Fourier data   phat(k) = sum_r p(r) e^{-ikr},   p(r) = (1/N) sum_k phat(k) e^{ikr},   k = 2 pi (n + twist/2) / N,
where twist = 0 / 1 means periodic / antiperiodic correlations (flux 0 / pi through the ring).
Constraint (necessary and sufficient for the state to exist):  |phat(k)| <= 1  and  phat(-k) = conj(phat(k)).

The energy depends only on three numbers
    a = p(0) = <i rho_i eta_i> ,   b = p(-1) = <i rho_i eta_{i+1}> ,   c = p(+1) = <i rho_{i+1} eta_i> :
    E / (N J) = etilde(a,b,c) = e(a,b,c,d=a),
    e(a,b,c,d) = 3/16 [ -(b^2+c^2) - (a-d)(b^2-c^2) + (b^2+c^2)(a d - b c) ]      (bond energy <S_1.S_2>/J, x=y=0).
At a minimum phat(k) = -K(k)/|K(k)| with the mean-field couplings (per colour)
    K(k) = mu + 2 tR cos k + 2 i kappa sin k ,  mu = (J/3) etilde_a ,  tR = (J/6)(etilde_b + etilde_c) ,  kappa = (J/6)(etilde_b - etilde_c),
i.e. phat is the ground state of
    H_MF = sum_{alpha,i} [ mu i rho_i eta_i + (tR + kappa) i rho_i eta_{i+1} + (tR - kappa) i rho_{i+1} eta_i ].
The script prints the residual max_k | phat + K/|K| | as a self-consistency (stationarity) check.

Usage:  python heisenberg_gaussian_S1.py 40            # 40-site ring, both fluxes
        python heisenberg_gaussian_S1.py inf           # thermodynamic limit (grid of --grid points)
        python heisenberg_gaussian_S1.py 12 --ed       # also compare with exact diagonalisation (N <= ~20)
"""
import argparse

import numpy as np
from scipy.optimize import minimize


# ----------------------------------------------------------------------------------------------------------
# Energy functional
# ----------------------------------------------------------------------------------------------------------
def bond(a, b, c, d):
    """Bond energy e = <S_1.S_2>/J (TZ class) and its gradient (e_a, e_b, e_c, e_d).
    a = <i rho_1 eta_1>, b = <i rho_1 eta_2>, c = <i rho_2 eta_1>, d = <i rho_2 eta_2> (one fixed colour)."""
    q = b * b + c * c
    e = 3 / 16 * (-q - (a - d) * (b * b - c * c) + q * (a * d - b * c))
    ea = 3 / 16 * (-(b * b - c * c) + d * q)
    ed = 3 / 16 * ((b * b - c * c) + a * q)
    eb = 3 / 16 * (-2 * b * (1 + (a - d) - (a * d - b * c)) - c * q)
    ec = 3 / 16 * (-2 * c * (1 - (a - d) - (a * d - b * c)) - b * q)
    return e, ea, eb, ec, ed


def etilde(a, b, c):
    """Energy per site of a translation-invariant state (all bonds equal, d = a) and its gradient (a, b, c)."""
    e, ea, eb, ec, ed = bond(a, b, c, a)
    return e, ea + ed, eb, ec


# ----------------------------------------------------------------------------------------------------------
# Ring of N sites: momentum grid and parametrisation phat_k = s_k e^{i theta_k}
# ----------------------------------------------------------------------------------------------------------
class Ring:
    def __init__(self, N, twist=0):
        n = np.arange(N)
        self.N, self.twist = N, twist
        self.k = 2 * np.pi * (n + twist / 2) / N
        partner = (-n - twist) % N                  # index of the momentum -k
        self.ip = n[n < partner]                    # one representative of each pair (k, -k)
        self.jp = partner[self.ip]                  # ... and its partner
        self.isc = n[n == partner]                  # self-conjugate momenta (k = 0 or pi): phat real
        self.P = len(self.ip)
        self.em = np.exp(-1j * self.k)              # e^{-ik}

    def phat(self, x):
        """Parameters x = [theta (P), s (P), u (#self-conjugate)] -> phat on the full grid."""
        th, s, u = x[:self.P], x[self.P:2 * self.P], x[2 * self.P:]
        z = np.empty(self.N, complex)
        z[self.ip] = s * np.exp(1j * th)
        z[self.jp] = np.conj(z[self.ip])
        z[self.isc] = u
        return z

    def moments(self, z):
        """a = (1/N) sum phat,  b = (1/N) sum phat e^{-ik},  c = (1/N) sum phat e^{+ik}."""
        return ((z.sum() / self.N).real, ((z * self.em).sum() / self.N).real,
                ((z * self.em.conj()).sum() / self.N).real)

    def objective(self, x):
        """F = E/J = N etilde(a,b,c) and dF/dx. Since dF = Re sum_k dphat_k L_k with L_k = e_a + e_b e^{-ik} + e_c e^{ik}."""
        a, b, c = self.moments(self.phat(x))
        e, ea, eb, ec = etilde(a, b, c)
        L = ea + eb * self.em + ec * self.em.conj()
        th, s = x[:self.P], x[self.P:2 * self.P]
        w = np.exp(1j * th) * L[self.ip]
        grad = np.concatenate([-2 * s * w.imag, 2 * w.real, L[self.isc].real])
        return self.N * e, grad

    def start(self, rng):
        """Start from the ground state of a random mean-field Hamiltonian (random direction of mu : tR : kappa)."""
        mu, t, ka = rng.normal(size=3)
        K = mu + 2 * t * np.cos(self.k) + 2j * ka * np.sin(self.k)
        K = K + 1e-6 * (rng.normal(size=self.N) + 1j * rng.normal(size=self.N))   # lift accidental zeros
        ph = -K / np.abs(K)
        u = np.where(ph[self.isc].real >= 0, 1.0, -1.0)
        return np.concatenate([np.angle(ph[self.ip]), np.ones(self.P), u])


# ----------------------------------------------------------------------------------------------------------
# Minimiser and analysis
# ----------------------------------------------------------------------------------------------------------
def analyse(R, x, J=1.0):
    z = R.phat(x)
    a, b, c = R.moments(z)
    e, ea, eb, ec = etilde(a, b, c)
    mu, tR, ka = J * ea / 3, J * (eb + ec) / 6, J * (eb - ec) / 6
    K = mu + 2 * tR * np.cos(R.k) + 2j * ka * np.sin(R.k)
    ok = np.abs(K) > 1e-9
    resid = np.max(np.abs(z[ok] + K[ok] / np.abs(K[ok]))) if ok.any() else 0.0
    return dict(N=R.N, twist=R.twist, E=J * R.N * e, E_per_site=J * e, a=a, b=b, c=c,
                m1=(b + c) / 2, m2=(b - c) / 2, mu=mu, tR=tR, kappa=ka, gap=2 * np.abs(K).min(),
                residual=resid, min_abs_phat=np.abs(z).min(), phat=z, k=R.k)


def minimise(N, twist=0, starts=24, seed=0, J=1.0):
    """Minimise E over all |phat| <= 1 (translation-invariant TZ states) on an N-site ring; best of `starts` runs."""
    R = Ring(N, twist)
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


def exact_energy(N, J=1.0):
    """Ground-state energy of the periodic spin-1/2 Heisenberg ring by Lanczos (full 2^N basis, N <= ~20)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import eigsh
    s = np.arange(2 ** N, dtype=np.int64)
    diag = np.zeros(2 ** N)
    rows, cols, vals = [], [], []
    for i in range(N):
        j = (i + 1) % N
        bi, bj = (s >> i) & 1, (s >> j) & 1
        diag += 0.25 * J * np.where(bi == bj, 1.0, -1.0)                      # S^z_i S^z_j
        f = s[bi != bj]
        rows.append(f); cols.append(f ^ ((1 << i) | (1 << j))); vals.append(np.full(f.size, 0.5 * J))   # (S+S- + S-S+)/2
    rows.append(s); cols.append(s); vals.append(diag)
    H = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(2 ** N,) * 2).tocsr()
    return eigsh(H, k=1, which="SA")[0][0]


def show(r, inf=False):
    if inf:
        print(f"N -> infinity (k grid of {r['N']} points)")
    else:
        print(f"N = {r['N']}, twist = {r['twist']} ({'periodic' if r['twist'] == 0 else 'antiperiodic'} correlations)")
    print(f"  E/N               = {r['E_per_site']:.10f} J      (E = {r['E']:.10f} J)")
    print(f"  a, b, c           = {r['a']:.8f}, {r['b']:.8f}, {r['c']:.8f}")
    print(f"  m1=(b+c)/2, m2=(b-c)/2 = {r['m1']:.8f}, {r['m2']:.8f}   (twice the hopping / pairing correlator per colour)")
    print(f"  mu, tR, kappa     = {r['mu']:.8f}, {r['tR']:.8f}, {r['kappa']:.8f}   (units of J)")
    print(f"  gap 2 min|K|      = {r['gap']:.6f} J,   min|phat| = {r['min_abs_phat']:.6f},   "
          f"residual |phat + K/|K|| = {r['residual']:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("N", help='number of sites (integer >= 3) or "inf"')
    ap.add_argument("--twist", choices=["0", "1", "both"], default="both", help="periodic / antiperiodic correlations")
    ap.add_argument("--starts", type=int, default=24, help="number of random starting states")
    ap.add_argument("--grid", type=int, default=4096, help='number of k points used for N = "inf"')
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--J", type=float, default=1.0)
    ap.add_argument("--ed", action="store_true", help="compare with exact diagonalisation (N <= ~20)")
    args = ap.parse_args()

    inf = args.N.lower() in ("inf", "infinity")
    try:
        N = args.grid if inf else int(args.N)
    except ValueError:
        raise SystemExit('N must be an integer >= 3 or "inf"')
    if args.J <= 0:
        raise SystemExit("J must be > 0 (antiferromagnet): the minimiser works with E/J")
    if N < 3:
        raise SystemExit("N must be >= 3 (a 2-site ring double counts its bond; use bond(0,1,1,0) = -3/4 for N = 2)")
    twists = [0] if inf else ([0, 1] if args.twist == "both" else [int(args.twist)])
    res = [minimise(N, t, args.starts, args.seed, args.J) for t in twists]
    for r in res:
        show(r, inf)
    best = min(res, key=lambda r: r["E"])
    if inf:
        print(f"reference: dimer product {-3 / 8 * args.J:.6f} J/site,  exact (Bethe ansatz) {(0.25 - np.log(2)) * args.J:.6f} J/site")
    if args.ed:
        E0 = exact_energy(N, args.J)
        print(f"exact E0 = {E0:.10f} J   (E0/N = {E0 / N:.10f});  best Gaussian - exact = {best['E'] - E0:.3e} J  (must be >= 0)")