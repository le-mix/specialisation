#!/usr/bin/env python3
"""
Energy plots for the spin-1/2 Heisenberg model on the kagome lattice (strictly nearest neighbour, J = 1 on every bond, periodic clusters)
        H = sum_<ij> S_i . S_j
computed with the methods of this folder:

    exact diagonalisation                      kagome_gaussian_S1.exact_energy     Lanczos on the 2^N spin states (here N <= 24)
    Gaussian mean field (6 Majoranas, TZ)      kagome_gaussian_S1.minimise         variational; for every cluster tested the optimum is a perfect dimer covering, E_1 = -3N/8
    Rayleigh-Schroedinger series, Fock space   kagome_pt_checks.fock_series        brute force in 2^(3N) dimensions: N <= 6 only
    Rayleigh-Schroedinger series, diagrams     kagome_diagrams (Wick / Pfaffian)   orders 2-4 around a dimer-covering reference, any even N

THE REFERENCE.  The Gaussian state that minimises the energy is not unique: every perfect covering of the lattice by dimers (a singlet on each dimer, E_1 = -3N/8)
is a global minimum of the mean-field functional.  The coverings differ only beyond first order.  For every cluster the program enumerates ALL perfect dimer coverings up to the
lattice symmetries (kagome_diagrams.all_coverings: 2, 6, 12, 19, 104 inequivalent coverings for N = 12, 18a, 18b, 24, 30 and 350 or 67 for N = 36) and scores each by E_1 + E_2.
For N >= 24 the second-order energy is the same bulk value -17/768 per site for every covering, so the choice is made at third order: E_3 is computed for every covering that
ties for the lowest E_1 + E_2 (and for the `ncover` best in any case), and the covering with the lowest E_1 + E_2 (ties: lowest E_1 + E_2 + E_3) is THE reference; its orders
1, 2, 3 (and 4 if requested) are the plotted curves.  This is a rule, not a derivation: a lowest truncated energy is not a bound, and it is the optimistic end of the spread.
The shaded band is the range of the partial sums over the coverings (order 2: all coverings; order 3: all coverings for which E_3 was computed, i.e. all of them for N >= 24),
an indication of how much the truncated series still depends on the choice of reference.  The selection rule uses nothing beyond third order.

plot_kagome_all_methods(sizes=("12", "18a", "18b", "24"), ...)
    One column per cluster; top: the partial sums E_0+E_1 (= mean field), +E_2, +E_3, (+E_4) per site against the order, with the exact energy as a line and the
    range over the coverings as a band; bottom: the same as a percentage of the exact energy.  Orange circles: the brute-force Fock-space series (only for N <= 6;
    the 6-site torus 2x1 is a degenerate test case because the dimer product is its exact ground state).
plot_kagome_diagrammatic(sizes=("12", "18a", "18b", "24", "30", "36a", "36b", "36c", "36d"), ...)
    The diagrammatic partial sums per site against N for the even clusters, with exact diagonalisation (N <= 24) and the quoted literature values
    (N = 30, 36: unverified) and the DMRG energies per site of the infinite lattice, -0.4379(3) (Yan, Huse, White 2011) and -0.4386(5) (Depenbrock, McCulloch, Schollwoeck 2012).
Both return (figure, results), print every number they plot, and take show=False and savepath="figure.png".

Cost.  Measured on the machine on which these scripts were developed (2 cores, one process per core, OMP_NUM_THREADS=1):
       exact diagonalisation: N = 24 about 2 minutes and 2.3 GB of memory, N = 12-18 a few seconds (N = 27 and above: not done, the literature values are used);
       coverings: all inequivalent coverings scored at order 2 in under a minute per cluster (N = 12 ... 36); E_3 about 1-2 s per covering at N = 36 (all 350 coverings: 6-15 minutes,
       longer when other jobs share the cores);
       order 4 for ONE covering: 8-16 minutes of one core for every cluster from N = 12 to N = 36 (the program's estimates: 11, 16, 15, 13, 14, 14, 8, 8, 12 minutes for
       12, 18a, 18b, 24, 30, 36a, 36b, 36c, 36d; measured wall time 8-15 minutes with a core to itself, up to 30 minutes when four or five jobs shared the two cores), because the
       number of distinct integrands (10-20 thousand) saturates with N (kagome_diagrams.py, "identical integrands"); the program prints an estimate before every (cluster, order 4)
       (estimate_only=True prints it and stops) and skips the orders whose estimate exceeds budget_hours.
       Finished results are written to a cache file (default kagome_cache.json next to this file; shipped with the scripts); delete it to recompute.
Needs numpy, scipy, matplotlib >= 3.4 and kagome_diagrams.py, kagome_gaussian_S1.py, kagome_lattice.py, kagome_pt_checks.py, heisenberg_diagrams.py,
heisenberg_gaussian_S1.py, heisenberg_pt_checks.py in the same folder.
Usage:  python3 kagome_plots.py all --sizes 2x1 12 18a --order 3 --save fig.png
        python3 kagome_plots.py large --order 3
        python3 kagome_plots.py large --order 4 --estimate
In Python:  from kagome_plots import *;  fig, res = plot_kagome_all_methods(sizes=["12", "18a"], max_order=3)
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kagome_diagrams as kd
import kagome_gaussian_S1 as kg
import kagome_lattice as kl
import kagome_pt_checks as kp

__all__ = ["plot_kagome_all_methods", "plot_kagome_diagrammatic", "compute"]

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kagome_cache.json")
VERSION = 1
SEC_PER_ROW = {2: 1.0e-4, 3: 1.5e-4, 4: 6.0e-4}      # seconds per (evaluation, node tuple), one core (flat band; measured, see the estimate printed by the program)
SEC_PER_TUPLE = 1.0e-6                                # seconds per candidate configuration when the diagrams are enumerated (order 4)
MAX_FOCK_N = 6
DMRG = (("Yan-Huse-White 2011", -0.4379, 0.0003), ("Depenbrock et al. 2012", -0.4386, 0.0005))     # E / N of the infinite lattice

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


# ================================================================================================ cache and small helpers
def _load(path):
    if path is None or not os.path.exists(path):
        return {"version": VERSION}
    try:
        with open(path) as f:
            c = json.load(f)
        return c if c.get("version") == VERSION else {"version": VERSION}
    except (OSError, ValueError):
        return {"version": VERSION}


def _save(path, cache):
    """Merge `cache` into the file (several processes may share one cache) and write it atomically."""
    if path is None:
        return
    import fcntl
    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        disk = _load(path)
        disk.update(cache)
        tmp = path + f".{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(disk, f)
        os.replace(tmp, path)
        cache.update(disk)


def _fmt_time(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def _pairs_key(pairs):
    return ";".join(f"{i}-{j}" for i, j in pairs)


def _cluster(name):
    T = kl.torus_from_name(name)
    return T, T.N, T.bonds, T.couplings(), T.all_symmetries()


# ================================================================================================ computation
def _exact(name, N, bonds, J, ed, ed_max_N, cache, path, verbose):
    key = f"ed|{name}"
    if key in cache:
        return cache[key]["E"], cache[key]["source"]
    if ed and N <= ed_max_N:
        t0 = time.time()
        E = float(kg.exact_energy(N, bonds, J)[0])
        cache[key] = {"E": E, "source": "ED"}
        _save(path, cache)
        if verbose:
            print(f"   {name}: exact diagonalisation {E:.9f} ({time.time() - t0:.0f} s)", flush=True)
        return E, "ED"
    if name in kg.LITERATURE_ED:
        E, ver = kg.LITERATURE_ED[name]
        return float(E), ("literature (reproduced here by ED)" if ver else "literature (unverified)")
    return None, None


def _coverings(name, N, bonds, J, perms, ncover, seed, tol, cache, path, verbose):
    """All inequivalent dimer coverings with P_1, P_2 (and P_3 for the ties and the `ncover` best), sorted by E_1 + E_2, ties by E_1 + E_2 + E_3 (best first); see kagome_diagrams.covering_table."""
    key = f"cov|{name}|{ncover}|{seed}|r2"
    if key in cache:
        return cache[key]["rows"], cache[key]["exhaustive"]
    rows, exhaustive = kd.covering_table(N, bonds, J, perms, ncover, seed, tol, verbose=verbose)
    cache[key] = {"rows": rows, "exhaustive": exhaustive}
    _save(path, cache)
    return rows, exhaustive


def _order4(name, N, bonds, J, perms, row, tol, budget_hours, cache, path, verbose, estimate_only):
    """P_4 for one covering (cached); None if skipped because of the budget."""
    key = f"p4|{name}|{_pairs_key(row['pairs'])}"
    if key in cache:
        return cache[key]
    ref = kd.covering_reference(N, bonds, J, [tuple(p) for p in row["pairs"]])
    t0 = time.time()
    d = kd.GraphDiagrams(ref, bonds, J, 4, perms, tol)
    c = d.counts()
    est = c["evaluations"] * c["nodes"] * SEC_PER_ROW[4]
    print(f"   {name}: order 4: {c['configurations']} connected configurations -> {c['evaluations']} distinct evaluations x {c['nodes']} time nodes ({c['quadrature']}); "
          f"enumeration {time.time() - t0:.0f} s, estimated evaluation {_fmt_time(est)}", flush=True)
    if estimate_only or (budget_hours is not None and est > 3600 * budget_hours):
        if not estimate_only:
            print(f"      skipped (budget {budget_hours} h)", flush=True)
        return None
    t0 = time.time()
    P4 = d.energy()
    cache[key] = P4
    _save(path, cache)
    print(f"      done: P_4 = {P4:+.8f} ({_fmt_time(time.time() - t0)})", flush=True)
    return P4


def _fock(name, N, bonds, J, row, order, cache, path, verbose):
    key = f"fock|{name}|{_pairs_key(row['pairs'])}|{order}"
    if key in cache:
        return cache[key]
    ref = kd.covering_reference(N, bonds, J, [tuple(p) for p in row["pairs"]])
    t0 = time.time()
    E = [float(x) for x in kp.fock_series(N, bonds, J, ref.K, order)["E"]]
    cache[key] = E
    _save(path, cache)
    if verbose:
        print(f"   {name}: Fock-space series to order {order} ({time.time() - t0:.0f} s)", flush=True)
    return E


def compute(sizes, max_order=3, ncover=24, seed=0, tol=1e-6, ed=True, ed_max_N=24, budget_hours=1.0, fock=True, cache=DEFAULT_CACHE, estimate_only=False, verbose=True):
    """results[name] = dict(N, exact, exact_source, mf, partial (best covering, orders 1..), terms, band (min/max partial sums over the coverings, orders 1..3),
       pairs (the reference covering), fock (partial sums or None))."""
    c = _load(cache)
    results = {}
    for name in sizes:
        T, N, bonds, J, perms = _cluster(name)
        Eex, src = _exact(name, N, bonds, J, ed and not estimate_only, ed_max_N, c, cache, verbose)
        rows, exhaustive = _coverings(name, N, bonds, J, perms, ncover, seed, tol, c, cache, verbose)
        best = rows[0]
        P = {1: best["P"][0], 2: best["P"][1], 3: best["P"][2]}
        if max_order >= 4:
            P4 = _order4(name, N, bonds, J, perms, best, tol, budget_hours, c, cache, verbose, estimate_only)
            if P4 is not None:
                P[4] = P4
        top = max(P)
        E = kd.energies(P, top)
        partial = list(np.cumsum([E[n] for n in range(1, top + 1)]))
        S2 = [r["P"][0] + r["P"][1] for r in rows]
        S3 = [r["P"][0] + r["P"][1] + r["P"][2] + r["P"][1] for r in rows if r["P"][2] is not None]
        S1 = [r["P"][0] for r in rows]
        band = {1: (min(S1), max(S1)), 2: (min(S2), max(S2)), 3: (min(S3), max(S3))}
        fk = None
        if fock and N <= MAX_FOCK_N:
            fk = list(np.cumsum(_fock(name, N, bonds, J, best, min(max_order, 4), c, cache, verbose))[1:])
        results[name] = dict(N=N, exact=Eex, exact_source=src, mf=P[1], partial=partial, terms=[E[n] for n in range(1, top + 1)], band=band,
                             pairs=best["pairs"], ncover=len(rows), n3=len(S3), exhaustive=exhaustive, fock=fk, median3=float(np.median(S3)))
    return results


# ================================================================================================ printing
def _print_all(results, per_site):
    for name, r in results.items():
        N, ex = r["N"], r["exact"]
        sc = 1.0 / N if per_site else 1.0
        print(f"\nkagome cluster {name} (N = {N}), reference = the lowest E_1 + E_2 (ties: E_1 + E_2 + E_3) of {r['ncover']} inequivalent dimer coverings ({'all of them' if r['exhaustive'] else 'random sample'});  exact = " +
              (f"{ex * sc:.6f} ({r['exact_source']})" if ex is not None else "n/a"))
        for k, S in enumerate(r["partial"], start=1):
            lo, hi = r["band"].get(k, (None, None)) if k <= 3 else (None, None)
            print(f"   order {k}{' (mean field)' if k == 1 else ''}: E_{k} = {r['terms'][k - 1] * sc:+.6f}   partial sum = {S * sc:.6f}" +
                  (f"   {100 * (S - ex) / abs(ex):+6.2f}% from exact" if ex is not None else "") +
                  (f"   [over the coverings: {lo * sc:.6f} .. {hi * sc:.6f}]" if lo is not None else ""))
        if r["fock"] is not None:
            print("   Fock-space series partial sums: " + "  ".join(f"{s * sc:.6f}" for s in r["fock"]))


# ================================================================================================ plotting
def _rc():
    return {"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
            "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
            "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK, "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2,
            "ytick.labelcolor": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10, "legend.frameon": False, "legend.fontsize": 9,
            "font.size": 10, "lines.solid_capstyle": "round", "axes.ymargin": 0.14, "axes.xmargin": 0.05}


_HAM = "H = Σ S_i·S_j over the nearest-neighbour bonds of the kagome lattice (J = 1), periodic clusters"


def _titles(fig, title, second):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5).set_in_layout(False)
    fig.text(0.01, 0.93, _HAM + "\n" + second, ha="left", va="top", fontsize=9, color=INK2, linespacing=1.5)


def _pct(v, _):
    return "0%" if abs(v) < 1e-9 else f"{v:+.0f}%".replace("-", "−")


def plot_kagome_all_methods(sizes=("12", "18a", "18b", "24"), max_order=3, ncover=24, seed=0, tol=1e-6, per_site=True, budget_hours=1.0, fock=True,
                            cache=DEFAULT_CACHE, estimate_only=False, show=False, savepath=None, verbose=True):
    """All methods on the small kagome clusters (see the module docstring).  Returns (figure, results)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    sizes = [str(s) for s in sizes]
    res = compute(sizes, max_order, ncover, seed, tol, True, 24, budget_hours, fock, cache, estimate_only, verbose)
    if estimate_only:
        return None, res
    _print_all(res, per_site)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(sizes), figsize=(max(6.4, 3.5 * len(sizes) + 0.8), 6.8), squeeze=False, sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
        anyfock = False
        for j, name in enumerate(sizes):
            r, top, bot = res[name], axes[0, j], axes[1, j]
            N, ex = r["N"], r["exact"]
            sc = 1.0 / N if per_site else 1.0
            x = np.arange(1, len(r["partial"]) + 1)
            for ax, f in ((top, lambda S: np.asarray(S) * sc), (bot, lambda S: 100.0 * (np.asarray(S) - ex) / abs(ex))):
                if ex is None and ax is bot:
                    continue
                if ex is not None:
                    ax.axhline(float(f(ex)), color=INK, lw=1.2, zorder=2)
                for k in (2, 3):
                    if k <= len(x) and r["band"].get(k):
                        lo, hi = r["band"][k]
                        ax.fill_between([k - 0.18, k + 0.18], float(f(lo)), float(f(hi)), color=BLUE, alpha=0.25, lw=0, zorder=1)
                ax.plot(x, f(r["partial"]), color=BLUE, lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=3)
                if r["fock"] is not None:
                    ax.plot(np.arange(1, len(r["fock"]) + 1), f(r["fock"]), ls="none", marker="o", ms=11, mfc="none", mec=ORANGE, mew=1.5, zorder=4)
                    anyfock = True
                ax.plot([1], [float(f(r["mf"]))], ls="none", marker="D", ms=7, color=AQUA, mec=SURFACE, mew=1.2, zorder=5)
            top.set_title(f"N = {N}" if name == str(N) else f"{name}  (N = {N})", loc="left", fontsize=11)
            if ex is not None:
                top.text(0.03, ex * sc, f"exact {ex * sc:.4f}", transform=top.get_yaxis_transform(), ha="left", va="top", fontsize=8, color=INK2)
            top.tick_params(labelbottom=False)
            bot.set_xticks(range(1, max_order + 1))
            bot.set_xticklabels(["1\n(mean field)"] + [str(k) for k in range(2, max_order + 1)])
            bot.set_xlim(0.6, max_order + 0.4)
            bot.set_xlabel("order of the series")
            bot.yaxis.set_major_formatter(FuncFormatter(_pct))
            if j == 0:
                top.set_ylabel("E / N" if per_site else "E")
                bot.set_ylabel("relative to exact")
        h = [Line2D([], [], color=INK, lw=1.2, label="exact diagonalisation"),
             Line2D([], [], color=AQUA, ls="none", marker="D", ms=7, mec=SURFACE, mew=1.2, label="Gaussian mean field = dimer covering"),
             Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2, label="diagrams, reference covering (lowest E_1 + E_2, ties: E_1 + E_2 + E_3)"),
             Patch(color=BLUE, alpha=0.25, lw=0, label="range over the inequivalent coverings (order 2: all, order 3: all with E_3)")]
        if anyfock:
            h.append(Line2D([], [], color=ORANGE, ls="none", marker="o", ms=11, mfc="none", mew=1.5, label="Fock-space series (brute force)"))
        _titles(fig, "Ground-state energy of the kagome Heisenberg antiferromagnet", "partial sums of the series around a dimer-covering Gaussian state; the reference is the covering with the lowest E_1 + E_2 (ties: E_1 + E_2 + E_3)")
        fig.tight_layout(rect=(0, 0.1, 1, 0.865))
        fig.legend(handles=h, loc="lower center", ncol=2 if len(sizes) < 3 else 3, bbox_to_anchor=(0.5, 0.0))
    return _finish(fig, savepath, show), res


def plot_kagome_diagrammatic(sizes=("12", "18a", "18b", "24", "30", "36a", "36b", "36c", "36d"), max_order=3, ncover=24, seed=0, tol=1e-6, budget_hours=1.0,
                             cache=DEFAULT_CACHE, estimate_only=False, show=False, savepath=None, verbose=True):
    """The diagrammatic partial sums per site against N for the even kagome clusters.  Returns (figure, results)."""
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    sizes = [str(s) for s in sizes]
    res = compute(sizes, max_order, ncover, seed, tol, True, 24, budget_hours, False, cache, estimate_only, verbose)
    if estimate_only:
        return None, res
    _print_all(res, True)
    Ns = sorted({r["N"] for r in res.values()})
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(7.4, 5.4))
        ax.axhspan(min(v - e for _, v, e in DMRG), max(v + e for _, v, e in DMRG), color=INK2, alpha=0.18, lw=0, zorder=1)
        nord = max(len(r["partial"]) for r in res.values())
        for r in res.values():                 # range of the partial sums over the enumerated coverings (orders 2 and 3), behind the points
            for k in (2, 3):
                if k <= len(r["partial"]) and r["band"].get(k):
                    lo, hi = r["band"][k]
                    ax.plot([r["N"], r["N"]], [lo / r["N"], hi / r["N"]], color=RAMP[k - 1], lw=4.5, alpha=0.35, solid_capstyle="butt", zorder=2)
        for k in range(nord):
            xs = [r["N"] for r in res.values() if len(r["partial"]) > k]
            ys = [r["partial"][k] / r["N"] for r in res.values() if len(r["partial"]) > k]
            ax.plot(xs, ys, ls="none", marker="o", ms=7, color=RAMP[k], mec=SURFACE, mew=1.0, zorder=3 + k)
        for name, r in res.items():
            if r["exact"] is not None:
                ax.plot([r["N"]], [r["exact"] / r["N"]], ls="none", marker="s" if "unverified" not in r["exact_source"] else "s", ms=7,
                        mfc=INK if "unverified" not in r["exact_source"] else "none", mec=INK, mew=1.2, zorder=10)
        ax.set_xlabel("number of sites N")
        ax.set_ylabel("E / N")
        ax.set_xticks(Ns)
        h = [Line2D([], [], color=RAMP[k], ls="none", marker="o", ms=7, label=("mean field (order 1)" if k == 0 else f"through order {k + 1}")) for k in range(nord)]
        h += [Line2D([], [], color=INK, ls="none", marker="s", ms=7, label="exact diagonalisation (N ≤ 24)"),
              Line2D([], [], color=INK, ls="none", marker="s", ms=7, mfc="none", label="literature ED, unverified (N ≥ 27)")]
        from matplotlib.patches import Patch
        h.append(Line2D([], [], color=RAMP[2], lw=4.5, alpha=0.35, solid_capstyle="butt", label="range over the enumerated coverings (orders 2, 3)"))
        h.append(Patch(color=INK2, alpha=0.18, lw=0, label="DMRG, infinite lattice: −0.4379(3), −0.4386(5)"))
        _titles(fig, "Kagome Heisenberg antiferromagnet: diagrammatic series per site",
                "reference: the dimer covering with the lowest E_1 + E_2 (ties: E_1 + E_2 + E_3); same N: different tori")
        fig.tight_layout(rect=(0, 0.17, 1, 0.86))
        fig.legend(handles=h, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=8.5)
    return _finish(fig, savepath, show), res


def _finish(fig, savepath, show):
    if savepath:
        fig.savefig(savepath, dpi=160)
        print(f"figure written to {savepath}")
    if show:
        import matplotlib.pyplot as plt
        plt.show()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, helptext in (("all", "all methods on the small clusters"), ("large", "diagrammatic series on the larger even clusters")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--sizes", nargs="+", default=None)
        p.add_argument("--order", type=int, default=3, choices=[2, 3, 4])
        p.add_argument("--coverings", type=int, default=24)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--tol", type=float, default=1e-6)
        p.add_argument("--budget-hours", type=float, default=1.0)
        p.add_argument("--no-cache", action="store_true")
        p.add_argument("--estimate", action="store_true")
        p.add_argument("--save", default=None)
        p.add_argument("--show", action="store_true")
    a = ap.parse_args()
    cache = None if a.no_cache else DEFAULT_CACHE
    if a.cmd == "all":
        plot_kagome_all_methods(a.sizes or ("12", "18a", "18b", "24"), a.order, a.coverings, a.seed, a.tol, True, a.budget_hours, True, cache, a.estimate, a.show, a.save)
    else:
        plot_kagome_diagrammatic(a.sizes or ("12", "18a", "18b", "24", "30", "36a", "36b", "36c", "36d"), a.order, a.coverings, a.seed, a.tol, a.budget_hours, cache, a.estimate, a.show, a.save)


if __name__ == "__main__":
    main()
