#!/usr/bin/env python3
"""
Energy plots for the spin-1/2 J1-J2 Heisenberg ring
        H = J1 sum_i S_i.S_{i+1} + J2 sum_i S_i.S_{i+2}          (periodic, N sites, J1 > 0)
computed with the four methods of this folder:

    exact diagonalisation                      heisenberg_gaussian_S1.exact_energy      Lanczos on the 2^N spin states
    Gaussian mean field (6 Majoranas, TZ)      heisenberg_gaussian_S1.minimise          variational; it is order 1 of both series below
    Rayleigh-Schroedinger series, Fock space   heisenberg_pt_checks.tz_series           brute force in 2^(3N) dimensions: N <= 6 only
    Rayleigh-Schroedinger series, diagrams     heisenberg_diagrams (Wick / Pfaffian)    orders 2-4, any even N (odd N: dense path, about 30x slower)

plot_energies_all_methods(N=8, J1=1.0, J2=0.0, ...)
    All four methods for the ring size N: an integer or a list such as [4, 6, 8] (the Fock-space series exists for N <= 6 only).
    view="order" (default): one column per N; top: the partial sums E_0+E_1 (= mean field), +E_2, +E_3, +E_4 against the order, with the exact
                            energy as a line; bottom: the same as a percentage of the exact energy.
    view="N":               energy against N, one curve per method and order (needs several N).
plot_energies_diagrammatic(J1=1.0, J2=0.0, Nmax=50, ...)
    The diagrammatic partial sums (mean field, +E_2, +E_3, +E_4) against N for the even ring sizes 4 ... Nmax.  reference=True adds exact
    diagonalisation (N <= 16) and, where known, the thermodynamic-limit energy (J2 = 0: Bethe ansatz; J2 = J1/2: Majumdar-Ghosh).

Both return (figure, results), print every number they plot, and take show=False and savepath="figure.png".  results[N] is a dict with
'exact', 'mf', 'diag' (partial sums of orders 1..), 'terms' (E_1, E_2, ...), 'fock' (or None), 'twist' ...; energies include the factor J1.

Units.   The codes work with H/J1, so every energy is J1 times the J1 = 1 energy at J2/J1.  J1 must be > 0.  Validated: 0 <= J2/J1 <= 1.5.

Cost.    The diagrams are the expensive part.  A (configuration, node tuple) costs about 2.6e-4 s on one core at order 4 (8 times less for each
         order below, because the tables have 8^n entries); the number of them grows like N^(n-1) at order n, times 2^n vertex-type choices
         when J2 != 0 (only 1 type when J2 = 0), times the number of quadrature nodes (9-12, set by tol).  Measured on the machine on which
         these scripts were developed, one core, J2 = 0, tol = 1e-4, order 4:   N = 4: 2 s,  N = 6: 6 s,  N = 8: 44 s,  N = 10: about 2 min.
         Orders 2 and 3 are cheap (order 3 at N = 50: about 3 s).  The earlier production runs for J2 != 0 (tol 1e-5 to 1e-6, order 4 included)
         took about 11, 33 and 51 min for N = 6, 8, 10.
         The functions print an estimate for every (N, order) before they start (estimate_only=True prints it and stops), skip the orders
         whose estimate exceeds budget_hours, and write every finished (N, order) to a cache file (default heisenberg_cache.json, next to
         this file), so an interrupted run resumes and a second plot at the same or a neighbouring J2 reuses the diagrams:
         below the switching point (J2 < J1 for N = infinity, about 0.8 at N = 8) all J2 share one reference and therefore one set of
         diagrams, which is exactly what the cache stores (the polynomials T_{n,m} of heisenberg_diagrams).  workers=k runs k jobs in parallel
         (default: 1 for plot_energies_all_methods, one per core for plot_energies_diagrammatic).
Cache.   The heisenberg_cache.json supplied with this file already holds: orders 2-4 for N = 6, 8, 10 at every J2 below the switching point (the
         production tables of heisenberg_j1j2_results.txt, 8 printed digits); orders 2 and 3 for every even N <= 50 (tolerance 1e-4; J2 = 0 and
         J2 != 0 below the switching point); order 4 for N = 4 (J2 = 0, 0.3, 0.5) and, at J2 = 0, for N = 4 ... 10.  Anything else is computed on
         first use and added to the file; delete the file (or pass cache=None) to recompute everything.

Needs numpy, scipy, matplotlib >= 3.4 and heisenberg_gaussian_S1.py, heisenberg_pt_checks.py, heisenberg_diagrams.py in the same folder.
Usage:  python3 heisenberg_plots.py all 8 --J2 0.3                      # all methods, N = 8
        python3 heisenberg_plots.py all 4 6 8 --view N --save fig.png    # energy against N
        python3 heisenberg_plots.py large --J2 0 --Nmax 50 --workers 8   # diagrammatic series up to N = 50 (default: all cores)
        python3 heisenberg_plots.py large --J2 0.3 --estimate            # print the cost estimate only
In Python:  from heisenberg_plots import *;  fig, res = plot_energies_all_methods(N=[4, 6, 8], J1=1.0, J2=0.3)
"""
import argparse
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import heisenberg_gaussian_S1 as hg
import heisenberg_diagrams as hd

for _mod, _names in ((hg, ("minimise", "exact_energy")), (hd, ("Reference", "Diagrams", "vertex_means", "energies"))):
    for _name in _names:
        if not hasattr(_mod, _name):
            raise ImportError(f"{_mod.__name__}.{_name} is missing: use the J1-J2 versions of heisenberg_gaussian_S1.py, heisenberg_pt_checks.py "
                              f"and heisenberg_diagrams.py (all three in the same folder as this file)")
if not hasattr(hd.Reference, "equivalent"):
    raise ImportError("heisenberg_diagrams.Reference.equivalent is missing: use the J1-J2 version of heisenberg_diagrams.py")

__all__ = ["plot_energies_all_methods", "plot_energies_diagrammatic"]

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "heisenberg_cache.json")     # next to this file
SEC_PER_TRIPLE = 2.6e-4     # seconds per (configuration, node tuple) at order 4 on one core; 8 times less for each order below (tables of 8^n entries)
DENSE_FACTOR = 30.0         # the dense 16-leg path (odd N) is about 30 times slower per configuration
MAX_FOCK_N = 6              # the brute-force series lives in 2^(3N) dimensions

# colours: validated with the dataviz palette checker (categorical slots 1-3; one-hue ramp for the orders)
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"       # diagrams, Fock space, mean field
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]        # orders 1-4 of one series, light to dark


# ================================================================================================ input checks, formatting
def _couplings(J1, J2):
    if not J1 > 0:
        raise ValueError("J1 must be > 0: the mean-field and perturbative codes work with H/J1, and every energy is J1 times the J1 = 1 energy at J2/J1")
    if J2 < 0:
        warnings.warn("J2 < 0 (ferromagnetic next-nearest neighbours) has not been validated for this reference family", stacklevel=3)
    return float(J2) / float(J1)


def _sizes(Ns):
    out = set()
    for n in Ns:
        try:
            ok = float(n) == int(n)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            raise ValueError(f"ring sizes must be integers >= 3, got {n!r}")
        out.add(int(n))
    out = sorted(out)
    if not out or out[0] < 3:
        raise ValueError("ring sizes must be integers >= 3")
    return out


def _fmt_time(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 2 * 86400:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


# ================================================================================================ cache
def _load_cache(path):
    cache = {"version": 1, "diagrams": [], "fock": []}
    if path and os.path.exists(path):
        try:
            with open(path) as f:
                loaded = json.load(f)
            if loaded.get("version") == 1:
                cache["diagrams"], cache["fock"] = loaded.get("diagrams", []), loaded.get("fock", [])
            else:
                warnings.warn(f"cache {path} has an unknown version; ignored", stacklevel=2)
        except (OSError, ValueError) as err:
            warnings.warn(f"cache {path} could not be read ({err}); ignored", stacklevel=2)
    return cache


def _save_cache(path, cache):
    """Write the cache atomically; a failure (read-only folder, full disk) only warns, so that a long run is never lost at its end."""
    if not path:
        return
    tmp = f"{path}.tmp{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(cache, f, indent=1)
        os.replace(tmp, path)
    except OSError as err:
        warnings.warn(f"cache {path} could not be written ({err}); the results of this run are not saved", stacklevel=2)


def _find_T(cache, ref, n, tol, types):
    """Diagram sums T_{n,m} (J1 = 1, m = number of distance-2 vertices) of an equivalent reference computed with accuracy <= tol, or None.
       An entry computed with the types (1, 2) serves every J2; one computed with type 1 only serves J2 = 0."""
    for e in cache["diagrams"]:
        if (e["N"] == ref.N and e["n"] == n and e["tol"] <= tol * (1 + 1e-9) and (len(types) == 1 or len(e["types"]) == 2)
                and ref.equivalent(SimpleNamespace(N=e["N"], twist=e["twist"], mu=e["mu"], tR=e["tR"], kappa=e["kappa"], tR2=e["tR2"], kappa2=e["kappa2"]))):
            return np.array(e["T"], float)
    return None


def _store_T(cache, ref, n, tol, types, T, seconds):
    cache["diagrams"].append(dict(N=int(ref.N), n=int(n), tol=float(tol), types=[int(t) for t in types], twist=int(ref.twist), mu=float(ref.mu),
                                  tR=float(ref.tR), kappa=float(ref.kappa), tR2=float(ref.tR2), kappa2=float(ref.kappa2),
                                  T=[float(x) for x in T], seconds=float(seconds)))


def _find_fock(cache, N, twist, ratio, K):
    for e in cache["fock"]:
        if e["N"] == N and e["twist"] == twist and abs(e["ratio"] - ratio) < 1e-12 and e["K"] >= K:
            return list(e["E"][:K + 1])
    return None


def _store_fock(cache, N, twist, ratio, K, E, seconds):
    cache["fock"].append(dict(N=int(N), twist=int(twist), ratio=float(ratio), K=int(K), E=[float(x) for x in E], seconds=float(seconds)))


# ================================================================================================ planning and running the jobs
def _plan_diagrams(Ns, ratio, max_order, tol, budget, sec_per_triple, cache, twist):
    """For each N: the reference, and for each order 2..max_order either the cached diagram sums or a ready-to-run Diagrams object with its cost estimate."""
    types = (1,) if ratio == 0.0 else (1, 2)
    plan = {}
    for N in Ns:
        try:
            ref = hd.Reference(N, twist=twist, J2=ratio)
        except ValueError as err:
            warnings.warn(f"N = {N}: {err}; skipped", stacklevel=3)
            continue
        orders, blocked = {}, False
        for n in range(2, max_order + 1):
            T = _find_T(cache, ref, n, tol, types)
            if T is not None:
                orders[n] = dict(T=T, cached=True, est=0.0)
            elif blocked:
                orders[n] = dict(T=None, skipped="a lower order is missing", est=None)
            else:
                d = hd.Diagrams(ref, n, tol, dense=None, symmetric=True, verbose=False, types=types)
                est = len(d.nodes) * len(types) ** n * N ** (n - 1) * sec_per_triple * 8.0 ** (n - 4) * (DENSE_FACTOR if d.dense else 1.0)
                if budget is not None and est > budget:
                    blocked = True
                    orders[n] = dict(T=None, skipped="estimate above the budget", est=est, K=d.K)
                else:
                    orders[n] = dict(T=None, d=d, est=est, K=d.K)
        plan[N] = dict(ref=ref, types=types, orders=orders)
    return plan


def _print_plan(plan, ratio, tol, fock_todo, workers, budget):
    print(f"\ndiagrammatic series: J2/J1 = {ratio:g}, quadrature tolerance {tol:g}, vertex types {'1' if ratio == 0 else '1 and 2'}"
          f"{'' if budget is None else f', budget {_fmt_time(budget)} per (N, order)'}")
    print("estimated CPU time per (N, order) on one core:")
    jobs = []
    for N, pl in plan.items():
        cells = []
        for n, o in sorted(pl["orders"].items()):
            if o.get("cached"):
                cells.append(f"{n}: cached")
            elif o.get("skipped"):
                cells.append(f"{n}: SKIPPED ({o['skipped']}" + (f", {_fmt_time(o['est'])}" if o.get("est") else "") + ")")
            else:
                cells.append(f"{n}: {_fmt_time(o['est'])}")
                jobs.append(o["est"])
        print(f"   N = {N:>2}  twist {pl['ref'].twist}  {'dense (odd N)' if not pl['ref'].factorised else '             '}   " + "   ".join(cells))
    if fock_todo:
        print(f"   Fock-space series still to compute for N = {', '.join(str(n) for n in fock_todo)} (minutes at N = 6)")
    if jobs:
        print(f"   total {_fmt_time(sum(jobs))} of CPU time; with {workers} worker(s) about {_fmt_time(max(max(jobs), sum(jobs) / max(workers, 1)))} of wall time", flush=True)
    else:
        print("   no diagram job to run", flush=True)


_TPL = None


def _init_worker():
    global _TPL
    try:
        from threadpoolctl import threadpool_limits
        _TPL = threadpool_limits(1)
    except Exception:
        pass


def _energy_job(d):
    t0 = time.time()
    return d.energy(), time.time() - t0


def _fock_job(N, twist, ratio, K):
    import heisenberg_pt_checks as pc
    t0 = time.time()
    out = pc.tz_series(N, twist, K, J2=ratio)
    return [float(x) for x in out["E"]], time.time() - t0


def _run_jobs(plan, fock_todo, ratio, tol, max_order, cache, cache_path, workers, verbose):
    jobs = [dict(kind="diag", N=N, n=n, est=o["est"]) for N, pl in plan.items() for n, o in pl["orders"].items() if o.get("d") is not None]
    jobs.sort(key=lambda j: -j["est"])
    jobs += [dict(kind="fock", N=N, n=None, est=0.0) for N in fock_todo]
    fock_out, done = {}, [0]

    def finish(j, value, sec):
        done[0] += 1
        pl = plan[j["N"]]
        if j["kind"] == "diag":
            T = np.asarray(value, float)
            pl["orders"][j["n"]]["T"] = T
            _store_T(cache, pl["ref"], j["n"], tol, pl["types"], T, sec)
            what = f"diagrams, order {j['n']}"
        else:
            fock_out[j["N"]] = value
            _store_fock(cache, j["N"], pl["ref"].twist, ratio, max_order, value, sec)
            what = f"Fock-space series to order {max_order}"
        _save_cache(cache_path, cache)
        if verbose:
            print(f"   [{done[0]}/{len(jobs)}] N = {j['N']}, {what}: {_fmt_time(sec)}" + (f" (estimate {_fmt_time(j['est'])})" if j["est"] else ""), flush=True)

    def args(j):
        pl = plan[j["N"]]
        if j["kind"] == "diag":
            return _energy_job, (pl["orders"][j["n"]]["d"],)
        return _fock_job, (j["N"], pl["ref"].twist, ratio, max_order)

    if workers <= 1:
        for j in jobs:
            fn, a = args(j)
            if j["kind"] == "diag":
                a[0].verbose = verbose >= 2                 # chunk-by-chunk progress of a long job
            finish(j, *fn(*a))
    elif jobs:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
            futs = {}
            for j in jobs:
                fn, a = args(j)
                futs[ex.submit(fn, *a)] = j
            try:
                for f in as_completed(futs):
                    finish(futs[f], *f.result())
            except BaseException:
                ex.shutdown(wait=False, cancel_futures=True)
                raise
    return fock_out


def _partial_sums(ref, N, ratio, T_by_order, max_order):
    """Partial sums S_1..S_n and terms E_1..E_n of the diagrammatic series (J1 = 1), n = highest order with all of P_2..P_n known."""
    c = hd.vertex_means(ref)
    P = {1: np.array([3 * N * c[1], 3 * N * c[2]] + [0.0] * (max_order - 1))}
    n_av = 1
    for n in range(2, max_order + 1):
        T = T_by_order.get(n)
        if T is None:
            break
        Pn = np.zeros(max_order + 1)
        Pn[:len(T)] = T
        P[n] = Pn
        n_av = n
    _, E = hd.energies(P, ratio, n_av)
    terms = np.array([E[k] for k in range(1, n_av + 1)])
    return np.cumsum(terms), terms


def _compute(Ns, J1, J2, max_order, tol, budget_hours, fock_max_N, ed_max_N, cache, workers, sec_per_triple, estimate_only, verbose, twist):
    """Everything the two plot functions need: results[N] with the exact energy (N <= ed_max_N), the mean field, the diagrammatic and the Fock-space series."""
    ratio = _couplings(J1, J2)
    if max_order not in (2, 3, 4):
        raise ValueError("max_order must be 2, 3 or 4")
    if twist not in (None, 0, 1):
        raise ValueError("twist must be None (seam sign with the lower mean-field energy), 0 or 1")
    Ns = _sizes(Ns)
    workers = (os.cpu_count() or 1) if workers is None else int(workers)
    if workers < 1:
        raise ValueError("workers must be a positive integer (or None: one per core)")
    budget = None if budget_hours is None else 3600.0 * budget_hours
    cdata = _load_cache(cache)
    plan = _plan_diagrams(Ns, ratio, max_order, tol, budget, sec_per_triple, cdata, twist)
    fock_have, fock_todo = {}, []
    for N in plan:
        if N <= min(fock_max_N, MAX_FOCK_N):
            E = _find_fock(cdata, N, plan[N]["ref"].twist, ratio, max_order)
            if E is None:
                fock_todo.append(N)
            else:
                fock_have[N] = E
    if fock_todo:
        import inspect
        import heisenberg_pt_checks as pc
        if "J2" not in inspect.signature(pc.tz_series).parameters:
            raise ImportError("heisenberg_pt_checks.tz_series has no J2 argument: use the J1-J2 version of heisenberg_pt_checks.py")
    if verbose or estimate_only:
        _print_plan(plan, ratio, tol, fock_todo, workers, budget)
    if estimate_only:
        return {N: {n: ("cached" if o.get("cached") else o.get("est")) for n, o in pl["orders"].items()} for N, pl in plan.items()}
    if not plan:
        raise RuntimeError("no ring size could be treated")
    fock_have.update(_run_jobs(plan, fock_todo, ratio, tol, max_order, cdata, cache, workers, verbose))
    results = {}
    for N, pl in plan.items():
        S, terms = _partial_sums(pl["ref"], N, ratio, {n: o["T"] for n, o in pl["orders"].items() if o.get("T") is not None}, max_order)
        fock = None
        if N in fock_have:
            E = fock_have[N]
            fock = J1 * np.cumsum([E[0] + E[1]] + list(E[2:max_order + 1]))
        results[N] = dict(N=N, J1=J1, J2=J2, twist=pl["ref"].twist, mf=J1 * pl["ref"].E_min, diag=J1 * S, terms=J1 * terms, fock=fock,
                          exact=hg.exact_energy(N, J1, J2) if N <= ed_max_N else None, orders=len(S), max_order=max_order,
                          skipped=[n for n, o in pl["orders"].items() if o.get("skipped")], K={n: o.get("K") for n, o in pl["orders"].items()})
    return results


# ================================================================================================ drawing
def _rc():
    return {"figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
            "axes.grid": True, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
            "grid.linestyle": "-", "axes.labelcolor": INK2, "axes.titlecolor": INK, "text.color": INK, "xtick.color": AXIS, "ytick.color": AXIS,
            "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10, "legend.frameon": False,
            "legend.fontsize": 9, "font.size": 10, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "axes.ymargin": 0.14, "axes.xmargin": 0.05}


_HAM = "H = J1 Σ S_i·S_(i+1) + J2 Σ S_i·S_(i+2),  periodic ring of N sites"


def _titles(fig, title, second_line):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5).set_in_layout(False)     # the rect of tight_layout below reserves the room for the title block
    fig.text(0.01, 0.93, _HAM + "\n" + second_line, ha="left", va="top", fontsize=9, color=INK2, linespacing=1.5)


def _second_line(results):
    have = [N for N, r in results.items() if r["fock"] is not None]
    if len(have) < len(results):
        return f"partial sums around the TZ Gaussian state;  Fock-space series for N <= {max(have) if have else MAX_FOCK_N} only"
    return "partial sums of the series around the TZ Gaussian state"


def _pct(v, _):
    return "0%" if abs(v) < 1e-9 else f"{v:+.0f}%".replace("-", "−")


def _handles(kinds, exact_marker=False):
    from matplotlib.lines import Line2D
    h = {"exact": Line2D([], [], color=INK, lw=1.2, marker="o" if exact_marker else None, ms=5, label="exact diagonalisation"),
         "mf": Line2D([], [], color=AQUA, ls="none", marker="D", ms=7, mec=SURFACE, mew=1.2, label="Gaussian mean field (minimiser)"),
         "diag": Line2D([], [], color=BLUE, lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2, label="diagrams (Wick / Pfaffian)"),
         "fock": Line2D([], [], color=ORANGE, ls="none", marker="o", ms=11, mfc="none", mew=1.5, label="Fock-space series (brute force)")}
    return [h[k] for k in kinds]


def _plot_by_order(results, J1, J2, per_site, max_order):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    Ns = sorted(results)
    with plt.rc_context(_rc()):
        fig, axes = plt.subplots(2, len(Ns), figsize=(max(6.4, 3.7 * len(Ns) + 0.8), 6.8), squeeze=False, sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
        any_fock = False
        for j, N in enumerate(Ns):
            r, top, bot = results[N], axes[0, j], axes[1, j]
            sc, ex = (1.0 / N if per_site else 1.0), r["exact"]
            x = np.arange(1, len(r["diag"]) + 1)
            xf = None if r["fock"] is None else np.arange(1, len(r["fock"]) + 1)
            any_fock |= xf is not None
            for ax, f in ((top, lambda S: np.asarray(S) * sc), (bot, lambda S: 100.0 * (np.asarray(S) - ex) / abs(ex))):
                ax.axhline(float(f(ex)), color=INK, lw=1.2, zorder=2)
                ax.plot(x, f(r["diag"]), color=BLUE, lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=3)
                if xf is not None:
                    ax.plot(xf, f(r["fock"]), ls="none", marker="o", ms=11, mfc="none", mec=ORANGE, mew=1.5, zorder=4)
                ax.plot([1], [float(f(r["mf"]))], ls="none", marker="D", ms=7, color=AQUA, mec=SURFACE, mew=1.2, zorder=5)
            top.set_title(f"N = {N}", loc="left", fontsize=11)
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
        handles = _handles(["exact", "mf", "diag"] + (["fock"] if any_fock else []))
        _titles(fig, f"Ground-state energy of the spin-1/2 ring:  J1 = {J1:g},  J2 = {J2:g}", _second_line(results))
        fig.tight_layout(rect=(0, 0.1, 1, 0.865))
        fig.legend(handles=handles, loc="lower center", ncol=2 if len(Ns) < 3 else 4, bbox_to_anchor=(0.5, 0.0))
    return fig


def _plot_by_N(results, J1, J2, per_site, max_order):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter
    Ns = sorted(results)
    sc = {N: (1.0 / N if per_site else 1.0) for N in Ns}
    with plt.rc_context(_rc()):
        fig, (top, bot) = plt.subplots(2, 1, figsize=(7.6, 7.4), sharex=True, gridspec_kw=dict(height_ratios=[3, 2]))
        ex = {N: results[N]["exact"] for N in Ns}
        dev = {N: (lambda S, e=ex[N]: 100.0 * (np.asarray(S) - e) / abs(e)) for N in Ns}
        for ax, f in ((top, lambda N, S: np.asarray(S) * sc[N]), (bot, lambda N, S: dev[N](S))):
            ax.plot(Ns, [f(N, ex[N]) for N in Ns], color=INK, lw=1.2, marker="o", ms=5, zorder=2)
            ax.plot(Ns, [f(N, results[N]["mf"]) for N in Ns], color=AQUA, lw=1.5, marker="D", ms=7, mec=SURFACE, mew=1.2, zorder=3)
            for n in range(2, max_order + 1):
                ns = [N for N in Ns if len(results[N]["diag"]) >= n]
                if ns:
                    ax.plot(ns, [f(N, results[N]["diag"][n - 1]) for N in ns], color=RAMP[n - 1], lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=3 + n)
                nf = [N for N in Ns if results[N]["fock"] is not None and len(results[N]["fock"]) >= n]
                if nf:
                    ax.plot(nf, [f(N, results[N]["fock"][n - 1]) for N in nf], ls="none", marker="o", ms=11, mfc="none", mec=ORANGE, mew=1.5, zorder=9)
        bot.set_xticks(Ns)
        bot.set_xlabel("ring size N")
        top.set_ylabel("E / N" if per_site else "E")
        bot.set_ylabel("relative to exact")
        bot.yaxis.set_major_formatter(FuncFormatter(_pct))
        h = _handles(["exact", "mf"], exact_marker=True) + [Line2D([], [], color=RAMP[n - 1], lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2,
                                                                    label=f"diagrams, order {n}") for n in range(2, max_order + 1)]
        if any(results[N]["fock"] is not None for N in Ns):
            h += _handles(["fock"])
        _titles(fig, f"Ground-state energy of the spin-1/2 ring:  J1 = {J1:g},  J2 = {J2:g}", _second_line(results))
        fig.tight_layout(rect=(0, 0.11, 1, 0.865))
        fig.legend(handles=h, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    return fig


def _known_limit(J1, J2):
    r = J2 / J1
    if abs(r) < 1e-12:
        return J1 * (0.25 - math.log(2)), "N = infinity, Bethe ansatz"
    if abs(r - 0.5) < 1e-12:
        return -0.375 * J1, "Majumdar-Ghosh dimer product (exact)"
    return None


def _plot_large(results, J1, J2, per_site, max_order, reference, ed_max_N, tol, budget_hours):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    Ns = sorted(results)
    sc = {N: (1.0 / N if per_site else 1.0) for N in Ns}
    names = ["mean field (order 1)"] + [f"+ $E_{n}$ (order {n})" for n in range(2, max_order + 1)]
    with plt.rc_context(_rc()):
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        handles = []
        for k in range(1, max_order + 1):
            ns = [N for N in Ns if len(results[N]["diag"]) >= k]
            if not ns:
                continue
            y = [results[N]["diag"][k - 1] * sc[N] for N in ns]
            ax.plot(ns, y, color=RAMP[k - 1], lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2, zorder=3 + k)
            ax.annotate(names[k - 1].split(" (")[0].replace("mean field", "MF"), xy=(ns[-1], y[-1]), xytext=(7, 0), textcoords="offset points", va="center", ha="left",
                        fontsize=9, color=INK2)
            handles.append(Line2D([], [], color=RAMP[k - 1], lw=1.5, marker="o", ms=6, mec=SURFACE, mew=1.2, label=names[k - 1]))
        if reference:
            Ne = [N for N in Ns if results[N]["exact"] is not None]
            if Ne:
                ax.plot(Ne, [results[N]["exact"] * sc[N] for N in Ne], ls="none", marker="o", ms=6, color=INK, mec=SURFACE, mew=1.2, zorder=10)
                handles.append(Line2D([], [], color=INK, ls="none", marker="o", ms=6, mec=SURFACE, mew=1.2, label=f"exact diagonalisation (N <= {ed_max_N})"))
            lim = _known_limit(J1, J2)
            if lim is not None:
                ax.plot([Ns[0], Ns[-1]], [lim[0] * (1 if per_site else Ns[0]), lim[0] * (1 if per_site else Ns[-1])], color=INK2, lw=1.0, zorder=2)
                handles.append(Line2D([], [], color=INK2, lw=1.0, label=f"{lim[1]}: {lim[0] / J1:.4f} J1 per site"))
        ax.set_xlim(Ns[0] - 1, Ns[-1] + 0.1 * (Ns[-1] - Ns[0]) + 1)
        ax.set_xlabel("ring size N")
        ax.set_ylabel("E / N" if per_site else "E")
        _titles(fig, f"Diagrammatic series around the TZ Gaussian state:  J1 = {J1:g},  J2 = {J2:g}",
                f"partial sums of the series, orders 1-{max_order};  time-quadrature tolerance {tol:g}")
        short = sorted({N for N in Ns if len(results[N]["diag"]) < max_order})
        note = ""
        if short:
            have = {k: max([N for N in Ns if len(results[N]["diag"]) >= k] or [0]) for k in range(2, max_order + 1)}
            note = ("; ".join(f"order {k} computed for N <= {m}" for k, m in have.items() if m < Ns[-1])
                    + (f"  (time budget {budget_hours:g} h per point)" if budget_hours is not None else ""))
        # the legend sits below the axes, where it cannot hide a curve or the reference line
        ncol = 3 if len(handles) > 3 else len(handles)
        nrow = -(-len(handles) // ncol)
        y_leg = 0.045 if note else 0.01
        fig.tight_layout(rect=(0, y_leg + 0.045 * nrow + 0.01, 1, 0.865))
        fig.legend(handles=handles, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, y_leg))
        if note:
            fig.text(0.01, 0.012, note, ha="left", va="bottom", fontsize=8.5, color=INK2)
    return fig


# ================================================================================================ tables
def _print_all_methods(results, per_site):
    for N in sorted(results):
        r = results[N]
        ex = r["exact"]
        print(f"\nN = {N},  J1 = {r['J1']:g},  J2 = {r['J2']:g}   (reference: seam sign twist = {r['twist']})")
        print(f"   {'method':<36}{'E':>14}{'E / N':>12}{'relative to exact':>20}")
        rows = [("exact diagonalisation", ex), ("Gaussian mean field (minimiser)", r["mf"])]
        rows += [(f"diagrams, order {k}", S) for k, S in enumerate(r["diag"][1:], start=2)]
        if r["fock"] is not None:
            rows += [(f"Fock-space series, order {k}", S) for k, S in enumerate(r["fock"][1:], start=2)]
        for i, (name, E) in enumerate(rows):
            print(f"   {name:<36}{E:>14.8f}{E / N:>12.6f}" + ("" if i == 0 else f"{100 * (E - ex) / abs(ex):>+18.2f} %"))
        if len(r["terms"]) > 1 and abs(r["mf"] - ex) > 1e-8 and np.all(np.abs(r["terms"][1:]) < 1e-8 * max(1.0, abs(ex))):
            print(f"   all corrections vanish: this reference (twist {r['twist']}) is an eigenstate of H that is not the ground state; "
                  f"twist={1 - r['twist']} gives the other seam sign")
        if r["skipped"]:
            print(f"   (diagram orders {r['skipped']} skipped: over the time budget)")
        if r["fock"] is None:
            print(f"   (no Fock-space series: it needs 2^(3N) = {2 ** (3 * N):.2e} dimensions, implemented for N <= {MAX_FOCK_N})")


def _print_large(results, max_order):
    print("\n   N   twist" + "".join(f"{'E/N, order ' + str(k):>17}" for k in range(1, max_order + 1)))
    for N in sorted(results):
        r = results[N]
        print(f"  {N:>3}   {r['twist']:>3}  " + "".join(f"{r['diag'][k] / N:>17.6f}" if k < len(r["diag"]) else f"{'-':>17}" for k in range(max_order))
              + (f"     exact {r['exact'] / N:.6f}" if r["exact"] is not None else ""))


# ================================================================================================ the two public functions
def plot_energies_all_methods(N=8, J1=1.0, J2=0.0, max_order=4, tol=1e-5, view="order", per_site=False, twist=None, fock_max_N=MAX_FOCK_N, budget_hours=None,
                              cache=DEFAULT_CACHE, workers=1, sec_per_triple=SEC_PER_TRIPLE, estimate_only=False, savepath=None, show=True, verbose=True):
    """Ground-state energy of the J1-J2 ring for the ring size N (an integer or a list, N <= 8) from all four methods: exact diagonalisation,
       Gaussian mean field, Fock-space series (N <= fock_max_N <= 6) and diagrammatic series, partial sums up to max_order (2, 3 or 4).
       view = "order" (columns of N, energy against the order) or "N" (energy against N).  per_site: plot E/N.  tol: accuracy of the time quadrature.
       twist: seam sign of the reference, None = the one with the lower mean-field energy (as in heisenberg_diagrams), or 0 / 1.
       budget_hours: skip the diagram orders whose estimated CPU time exceeds this (None: run everything).  cache: file with finished results (None: off).
       workers: parallel processes.  estimate_only: print the cost estimate and return (None, estimate).  Returns (figure, results)."""
    Ns = _sizes([N]) if np.ndim(N) == 0 else _sizes(N)
    if view not in ("order", "N"):
        raise ValueError('view must be "order" or "N"')
    if view == "N" and len(Ns) < 2:
        raise ValueError('view="N" needs at least two ring sizes')
    if max(Ns) > 12:
        warnings.warn("this function is meant for N <= 8: the diagrams cost grows like N^3 at order 4, and the exact energy needs 2^N states", stacklevel=2)
    results = _compute(Ns, J1, J2, max_order, tol, budget_hours, fock_max_N, 20, cache, workers, sec_per_triple, estimate_only, verbose, twist)
    if estimate_only:
        return None, results
    _print_all_methods(results, per_site)
    fig = (_plot_by_order if view == "order" else _plot_by_N)(results, J1, J2, per_site, max_order)
    _finish(fig, savepath, show)
    return fig, results


def plot_energies_diagrammatic(J1=1.0, J2=0.0, Nmax=50, Ns=None, max_order=4, tol=1e-4, budget_hours=1.0, per_site=True, twist=None, reference=False,
                               ed_max_N=16, cache=DEFAULT_CACHE, workers=None, sec_per_triple=SEC_PER_TRIPLE, estimate_only=False, savepath=None, show=True,
                               verbose=True):
    """Diagrammatic partial sums (order 1 = mean field, + E_2, + E_3, + E_4) against the ring size, for the even N = 4 ... Nmax (or the sizes Ns;
       odd N use the dense path: slow).  budget_hours: an order is skipped for every N whose estimated CPU time exceeds it (default 1 h; None: run
       everything).  At J2 = 0 and tol 1e-4, order 4 costs about 40 min at N = 28 and 5 h at N = 50 on one core, and 16 times more for J2 != 0
       (the 2^4 vertex-type choices); orders 2 and 3 cost seconds (J2 = 0) to a few minutes (J2 != 0) for every N.  With the default budget the
       J2 = 0 run does order 4 up to N = 32 (4.3 h of CPU time in all) and a J2 != 0 run stops order 4 at N = 12; the plan printed at the start
       shows the estimate for every (N, order), and estimate_only=True prints it and stops.
       twist: seam sign of the reference (None: lower mean-field energy).  reference=True adds exact diagonalisation for N <= ed_max_N and the
       exactly known energy per site where there is one (J2 = 0: Bethe ansatz, J2 = J1/2: Majumdar-Ghosh).  per_site: plot E/N (default).
       workers: parallel processes, one (N, order) job each (default None: one per core).  Returns (figure, results)."""
    if Ns is None:
        if _sizes([Nmax])[0] < 4:
            raise ValueError("Nmax must be at least 4")
        Ns = list(range(4, _sizes([Nmax])[0] + 1, 2))
    else:
        Ns = _sizes(Ns)
    results = _compute(Ns, J1, J2, max_order, tol, budget_hours, 0, ed_max_N if reference else 0, cache, workers, sec_per_triple, estimate_only, verbose, twist)
    if estimate_only:
        return None, results
    _print_large(results, max_order)
    fig = _plot_large(results, J1, J2, per_site, max_order, reference, ed_max_N, tol, budget_hours)
    _finish(fig, savepath, show)
    return fig, results


def _finish(fig, savepath, show):
    import matplotlib.pyplot as plt
    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"\nfigure written to {savepath}")
    if show:
        plt.show()


# ================================================================================================ command line
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--J1", type=float, default=1.0)
        p.add_argument("--J2", type=float, default=0.0)
        p.add_argument("--order", type=int, default=4, choices=[2, 3, 4], help="highest order of the series (default 4)")
        p.add_argument("--cache", default=DEFAULT_CACHE, help="file for finished results (default: heisenberg_cache.json next to this file; 'none' switches it off)")
        p.add_argument("--workers", type=int, default=None, help="parallel processes (default: 1 for 'all', one per core for 'large')")
        p.add_argument("--estimate", action="store_true", help="print the cost estimate and stop")
        p.add_argument("--save", default=None, help="write the figure to this file")
        p.add_argument("--no-show", action="store_true")
        p.add_argument("--quiet", action="store_true")
        p.add_argument("--per-site", action="store_true", dest="per_site", default=None)
        p.add_argument("--twist", choices=["best", "0", "1"], default="best", help="seam sign of the reference (default: lower mean-field energy)")
        p.add_argument("--budget-hours", type=float, default=None, help="skip (N, order) pairs whose estimated CPU time is larger")

    a = sub.add_parser("all", help="all methods for the ring size(s) N (<= 8)")
    a.add_argument("N", nargs="+", type=int)
    a.add_argument("--tol", type=float, default=1e-5)
    a.add_argument("--view", choices=["order", "N"], default="order")
    a.add_argument("--fock-max-N", type=int, default=MAX_FOCK_N)
    common(a)
    b = sub.add_parser("large", help="diagrammatic series for N up to Nmax")
    b.add_argument("--Nmax", type=int, default=50)
    b.add_argument("--tol", type=float, default=1e-4)
    b.add_argument("--reference", action="store_true", help="add exact diagonalisation (N <= 16) and the known thermodynamic limit")
    common(b)
    args = ap.parse_args()
    cache = None if str(args.cache).lower() == "none" else args.cache
    common_kw = dict(J1=args.J1, J2=args.J2, max_order=args.order, tol=args.tol, cache=cache, estimate_only=args.estimate,
                     savepath=args.save, show=not args.no_show, verbose=not args.quiet, twist={"best": None, "0": 0, "1": 1}[args.twist])
    try:
        if args.cmd == "all":
            plot_energies_all_methods(N=args.N, view=args.view, per_site=bool(args.per_site), fock_max_N=args.fock_max_N, budget_hours=args.budget_hours,
                                      workers=1 if args.workers is None else args.workers, **common_kw)
        else:
            plot_energies_diagrammatic(Nmax=args.Nmax, per_site=True if args.per_site is None else args.per_site, reference=args.reference,
                                       budget_hours=1.0 if args.budget_hours is None else args.budget_hours, workers=args.workers, **common_kw)
    except ValueError as err:
        ap.error(str(err))


if __name__ == "__main__":
    main()
