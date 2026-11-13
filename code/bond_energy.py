#!/usr/bin/env python3
"""
Bond energy <S_1 . S_2> for the six-Majorana spin representation

    S^a = P+ J_rho^a + P- J_eta^a ,     J^a = -(i/2) chi^b chi^c   (a,b,c cyclic),
    P+- = (1 +- Gamma)/2 ,              Gamma = i rho^x rho^y rho^z eta^x eta^y eta^z ,

evaluated in a rotation-invariant Gaussian state by Wick's theorem.

Conventions
-----------
* {chi_i, chi_j} = 2 delta_ij  (chi^2 = 1).
* Gaussian state:  <chi_A chi_B> = i G_AB  for A != B,  with G real antisymmetric.
* One flavour carries four Majoranas (rho_1, eta_1, rho_2, eta_2), and the state is
  flavour-diagonal (rotation invariant):
      <rho_i eta_j> = -i p_ij ,  p = [[a, b], [c, d]]
      <rho_1 rho_2> =  i x ,     <eta_1 eta_2> = i y.
* Majorana labels 1..12:  site s = 0,1 ; flavour a = 1,2,3
      rho(s,a) = 6 s + 2(a-1) + 1 ,   eta(s,a) = 6 s + 2(a-1) + 2.

An operator is a dict  { sorted tuple of Majorana labels : coefficient }.
"""
import collections
import sympy as sp


# ---------------------------------------------------------------- Majorana algebra
def mono_mul(m1, m2):
    """Product of two sorted monomials.  Each element j of m2 moves left past every
    element of m1 larger than j (one sign each); equal elements then cancel (chi^2 = 1)."""
    sign = (-1) ** sum(sum(1 for i in m1 if i > j) for j in m2)
    return sign, tuple(sorted(set(m1) ^ set(m2)))


def op_mul(A, B):
    res = {}
    for ka, va in A.items():
        for kb, vb in B.items():
            s, m = mono_mul(ka, kb)
            res[m] = res.get(m, 0) + s * va * vb
    res = {k: sp.expand(v) for k, v in res.items()}
    return {k: v for k, v in res.items() if v != 0}


def op_add(A, B):
    res = dict(A)
    for k, v in B.items():
        res[k] = res.get(k, 0) + v
    res = {k: sp.expand(v) for k, v in res.items()}
    return {k: v for k, v in res.items() if v != 0}


def op_scale(c, A):
    return {k: sp.expand(c * v) for k, v in A.items()}


ONE = {(): 1}


def chi(i):
    return {(i,): 1}


def prod(*ops):
    r = ONE
    for o in ops:
        r = op_mul(r, o)
    return r


# ---------------------------------------------------------------- the spin operator
def rho(s, a): return 6 * s + 2 * (a - 1) + 1
def eta(s, a): return 6 * s + 2 * (a - 1) + 2
def cyc(a): return (a % 3) + 1, ((a + 1) % 3) + 1          # (a, b, c) cyclic


def J_rho(s, a):
    b, c = cyc(a)
    return op_scale(-sp.I / 2, prod(chi(rho(s, b)), chi(rho(s, c))))


def J_eta(s, a):
    b, c = cyc(a)
    return op_scale(-sp.I / 2, prod(chi(eta(s, b)), chi(eta(s, c))))


def Gamma_op(s):
    return op_scale(sp.I, prod(*[chi(rho(s, a)) for a in (1, 2, 3)],
                               *[chi(eta(s, a)) for a in (1, 2, 3)]))


def spin(s, a):
    """S^a at site s:  (1+Gamma)/2 J_rho + (1-Gamma)/2 J_eta."""
    half = sp.Rational(1, 2)
    Pp = op_scale(half, op_add(ONE, Gamma_op(s)))
    Pm = op_scale(half, op_add(ONE, op_scale(-1, Gamma_op(s))))
    return op_add(op_mul(Pp, J_rho(s, a)), op_mul(Pm, J_eta(s, a)))


def commutator(A, B):
    return op_add(op_mul(A, B), op_scale(-1, op_mul(B, A)))


def two_site_hamiltonian():
    """H = S_1 . S_2  as a Majorana polynomial."""
    H = {}
    for a in (1, 2, 3):
        H = op_add(H, op_mul(spin(0, a), spin(1, a)))
    return H


# ---------------------------------------------------------------- Wick's theorem
def pfaffian(m):
    """Pfaffian of an antisymmetric sympy matrix, by expansion along the first row."""
    n = m.shape[0]
    if n == 0:
        return sp.Integer(1)
    if n % 2:
        return sp.Integer(0)
    total = 0
    for j in range(2, n + 1):                      # 1-based column index as in the usual formula
        if m[0, j - 1] == 0:
            continue
        rest = [i for i in range(n) if i not in (0, j - 1)]
        total += (-1) ** j * m[0, j - 1] * pfaffian(m.extract(rest, rest))
    return total


def covariance(p, R, Rp):
    """12x12 real antisymmetric G for the flavour-diagonal state.
    p = <rho_i eta_j> = -i p_ij (2x2), R = <rho rho>/i (2x2 antisymmetric), Rp = <eta eta>/i."""
    G = sp.zeros(12, 12)
    Q = -p
    for fl in (1, 2, 3):
        for s in (0, 1):
            for t in (0, 1):
                G[rho(s, fl) - 1, rho(t, fl) - 1] += R[s, t]
                G[eta(s, fl) - 1, eta(t, fl) - 1] += Rp[s, t]
                G[rho(s, fl) - 1, eta(t, fl) - 1] += Q[s, t]
                G[eta(t, fl) - 1, rho(s, fl) - 1] += -Q[s, t]
    return G

def bond_energy(H, G):
    """<H> = sum over monomials of coefficient * Pf( i G restricted to the monomial's indices )."""
    total = 0
    for mono, coeff in H.items():
        idx = [i - 1 for i in mono]
        M = sp.I * G.extract(idx, idx) if idx else sp.zeros(0, 0)
        total += coeff * pfaffian(M)
    return sp.expand(total)

# ---------------------------------------------------------------- the two mean-field classes
def covariance_TZ(p):
    """Time reversal  rho -> rho, eta -> -eta.
    Only <rho_i eta_j> = -i p_ij is nonzero;  <rho rho> = <eta eta> = 0."""
    zero = sp.zeros(2, 2)
    return covariance(p, zero, zero)


def covariance_TiY(r, q):
    """Time reversal  rho -> eta, eta -> -rho.
        <rho_1 rho_2> = i r ,   <eta_1 eta_2> = -i r      (so <eta eta> = -<rho rho>)
        <rho_1 eta_2> = -i q ,   <rho_2 eta_1> = i q      (antisymmetric rho-eta block)
    and no on-site correlators:  <rho_i eta_i> = 0."""
    R = sp.Matrix([[0, r], [-r, 0]])
    Q = sp.Matrix([[0, -q], [q, 0]])
    return covariance(Q, R, -R)            # covariance() expects p = -Q


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    a, b, c, d, x, y, r, q = sp.symbols("a b c d x y r q")

    # 1. the spin operator itself, on one site
    sq = {}
    for aa in (1, 2, 3):
        sq = op_add(sq, op_mul(spin(0, aa), spin(0, aa)))
    print("S^2 on one site (expect {(): 3/4}):", sq)
    print("[Sx,Sy] - i Sz  (expect {}):", op_add(commutator(spin(0, 1), spin(0, 2)),
                                                 op_scale(-sp.I, spin(0, 3))))

    # 2. the two-site Hamiltonian S_1 . S_2
    H = two_site_hamiltonian()
    deg = collections.Counter(len(k) for k in H)
    print("Majorana monomials in S1.S2:", len(H), "  by degree:", dict(sorted(deg.items())))

    # 3. T-Z, computed directly from its own covariance matrix
    e_T = bond_energy(H, covariance_TZ(sp.Matrix([[a, b], [c, d]])))
    print("\nT-Z, computed directly:")
    print("   e(a,b,c,d) =", sp.factor(e_T))
    e_old = sp.Rational(3, 16) * (a*b**2*d - a*b**2 + a*c**2*d + a*c**2 - b**3*c - b*c**3
                                  + b**2*d - c**2*d - b**2 - c**2)
    print("   agrees with the closed form derived earlier:", sp.expand(e_T - e_old) == 0)

    # 4. T-iY, computed directly from its own covariance matrix
    e_Tp = bond_energy(H, covariance_TiY(r, q))
    print("\nT-iY, computed directly:")
    print("   e(r,q) =", sp.factor(e_Tp))
    print("   agrees with (3/8)(q^2+r^2)(q^2-r^2-1):",
          sp.expand(e_Tp - sp.Rational(3, 8) * (q**2 + r**2) * (q**2 - r**2 - 1)) == 0)

    # 5. the general six-number result (both classes are special cases of it)
    p = sp.Matrix([[a, b], [c, d]])
    e = bond_energy(H, covariance(p, sp.Matrix([[0, x], [-x, 0]]), sp.Matrix([[0, y], [-y, 0]])))
    print("\nGeneral state, e(a,b,c,d; x,y) =", sp.factor(e))
    W = b**2 + c**2 - x**2 - y**2
    e_compact = sp.Rational(3, 16) * (-(b**2 + c**2 + x**2 + y**2) - (a - d) * (b**2 - c**2)
                                      - (a + d) * (x**2 - y**2) + W * (a * d - b * c - x * y))
    print("   agrees with the compact formula:", sp.expand(e - e_compact) == 0)

    # 6. the three direct computations must be consistent with each other
    print("\nConsistency:")
    print("   general result at x=y=0 equals the direct T-Z result:",
          sp.expand(e.subs({x: 0, y: 0}) - e_T) == 0)
    print("   general result at (a=d=0, b=-q, c=q, x=r, y=-r) equals the direct T' result:",
          sp.expand(e.subs({a: 0, d: 0, b: -q, c: q, x: r, y: -r}) - e_Tp) == 0)

    # 7. two-site checks with known answers
    print("\nTwo-site checks (the exact eigenvalues of S1.S2 are -3/4 and +1/4):")
    print("   T-Z, b=c=1 (rest 0):              ", e_T.subs({a: 0, b: 1, c: 1, d: 0}))
    print("   T-iY class, r=1, q=0:      ", e_Tp.subs({r: 1, q: 0}))
    print("   invariance under rho<->eta on site 2 only:",
          sp.expand(e - e.subs({b: -x, x: -b, c: y, y: c, d: -d}, simultaneous=True)) == 0)