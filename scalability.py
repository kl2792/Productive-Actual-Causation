"""Scalability experiment: random DAG generation + timing comparison.

Generates random binary SCMs and measures runtime of all_causes(Y, y)
for Bounded (Prop 5.4), Ours (Def 4.1), and HP Modified (2015).

Parameters:
  n ∈ {1, 2, ..., 20}
  density ∈ {0.2, 0.4, 0.6}
  indegree ∈ {2, 3, 5, unbounded}
  seeds-ours seeds for bounded+ours; seeds-baselines seeds for hp/bv/cness

Run:  python scalability.py [--output results.csv] [--timeout 10] [--workers 100]
      [--seeds-ours 500] [--seeds-baselines 50]
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import csv
import multiprocessing as mp
import os
import random
import signal
import sys
import time
from itertools import product as iproduct

from causation import SCM, Checker
from bounded_checker import BoundedChecker
from hp_modified import HPChecker
from bv_checker import BVChecker
from cness_checker import CNESSChecker


# ---------------------------------------------------------------------------
# Random DAG generation
# ---------------------------------------------------------------------------

def random_dag(n, density, max_indegree=None, rng=None):
    """Generate a random DAG on n nodes via Erdos-Renyi (upper triangle)."""
    if rng is None:
        rng = random.Random()

    vars_ = [f"V{i}" for i in range(n)]
    parents = {v: [] for v in vars_}

    for j in range(n):
        potential = [vars_[i] for i in range(j)]
        pa = [v for v in potential if rng.random() < density]
        if max_indegree is not None and len(pa) > max_indegree:
            pa = rng.sample(pa, max_indegree)
        parents[vars_[j]] = pa

    return vars_, parents


def random_binary_equations(vars_, parents, rng=None):
    """Generate random binary truth tables for each variable."""
    if rng is None:
        rng = random.Random()

    equations = {}
    for v in vars_:
        pa = parents[v]
        k = len(pa)
        n_inputs = 1 << k
        tt = tuple(rng.randint(0, 1) for _ in range(n_inputs))

        def make_func(pa_list, truth_table):
            def func(p):
                idx = 0
                for bit, parent in enumerate(pa_list):
                    idx |= (p[parent] << bit)
                return truth_table[idx]
            return func

        equations[v] = make_func(pa, tt)

    return equations


def random_scm(n, density, max_indegree=None, seed=None):
    """Generate a random binary SCM."""
    rng = random.Random(seed)
    vars_, parents = random_dag(n, density, max_indegree, rng)
    equations = random_binary_equations(vars_, parents, rng)

    scm = SCM(equations, parents)
    actual = scm.eval()

    Y = vars_[-1]
    y = actual[Y]

    return scm, Y, y


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


# ---------------------------------------------------------------------------
# Single-experiment worker (runs in a subprocess)
# ---------------------------------------------------------------------------

def _run_timed(checker_cls, scm, Y, y, timeout):
    """Run all_causes for a single checker with timeout."""
    ck = checker_cls(scm)
    result = {}
    try:
        old = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, timeout)
        t0 = time.perf_counter()
        causes = ck.all_causes(Y, y)
        t1 = time.perf_counter()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        result["time"] = t1 - t0
        result["ncauses"] = len(causes)
        result["cache"] = len(ck.cache)
        result["timeout"] = False
    except _Timeout:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
        result["time"] = timeout
        result["ncauses"] = -1
        result["cache"] = len(ck.cache)
        result["timeout"] = True

    if hasattr(ck, "frontier_max"):
        result["frontier_max"] = ck.frontier_max

    return result


def _worker(args):
    """Worker function for multiprocessing. Returns one CSV row as a dict."""
    n, density, max_indeg, seed, timeout, run_baselines = args
    scm, Y, y = random_scm(n, density, max_indeg, seed)
    n_edges = sum(len(pa) for pa in scm.pa.values())

    row = {"n": n, "density": density,
           "max_indegree": str(max_indeg) if max_indeg else "inf",
           "seed": seed, "n_vars": len(scm.vars), "n_edges": n_edges}

    checkers = [("bounded", BoundedChecker), ("ours", Checker)]
    if run_baselines:
        checkers += [("hp", HPChecker), ("bv", BVChecker), ("cness", CNESSChecker)]

    for prefix, cls in checkers:
        r = _run_timed(cls, scm, Y, y, timeout)
        row[f"{prefix}_time"] = r["time"]
        row[f"{prefix}_ncauses"] = r["ncauses"]
        row[f"{prefix}_cache"] = r["cache"]
        row[f"{prefix}_timeout"] = r["timeout"]
        if "frontier_max" in r:
            row["bounded_frontier_max"] = r["frontier_max"]

    # Fill empty strings for skipped baseline columns
    for prefix in ([] if run_baselines else ["hp", "bv", "cness"]):
        for suffix in ["_time", "_ncauses", "_cache", "_timeout"]:
            row[f"{prefix}{suffix}"] = ""

    row.setdefault("bounded_frontier_max", -1)
    return row


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

COLUMNS = [
    "n", "density", "max_indegree", "seed",
    "bounded_time", "ours_time", "hp_time", "bv_time", "cness_time",
    "bounded_ncauses", "ours_ncauses", "hp_ncauses", "bv_ncauses", "cness_ncauses",
    "bounded_cache", "ours_cache", "hp_cache", "bv_cache", "cness_cache",
    "bounded_timeout", "ours_timeout", "hp_timeout", "bv_timeout", "cness_timeout",
    "bounded_frontier_max",
    "n_vars", "n_edges",
]


def _truncate_partial_row(output_path):
    """If the file was killed mid-write, the last line may be incomplete.
    Truncate back to the last complete newline so DictReader parses cleanly.
    """
    if not os.path.exists(output_path):
        return
    with open(output_path, "rb") as f:
        content = f.read()
    if not content.endswith(b"\n"):
        last_newline = content.rfind(b"\n")
        if last_newline == -1:
            return  # only a header with no newline — leave it
        with open(output_path, "r+b") as f:
            f.seek(last_newline + 1)
            f.truncate()
        print(f"[resume] Truncated partial last row in {output_path}",
              flush=True)


def _load_completed(output_path):
    """Return set of (n, density, max_indegree_str, seed) already in CSV.

    If the file's schema doesn't match COLUMNS (e.g. BV/CNESS added later),
    returns empty set so the experiment runs fresh.
    """
    _truncate_partial_row(output_path)
    completed = set()
    if not os.path.exists(output_path):
        return completed
    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        existing_cols = set(reader.fieldnames or [])
        required_cols = set(COLUMNS)
        if not required_cols.issubset(existing_cols):
            missing = required_cols - existing_cols
            print(f"[resume] Schema mismatch — missing columns: {missing}. "
                  f"Starting fresh.", flush=True)
            return completed
        for row in reader:
            try:
                key = (int(row["n"]), float(row["density"]),
                       row["max_indegree"], int(row["seed"]))
                completed.add(key)
            except (KeyError, ValueError):
                pass
    return completed


def run_experiment(output_path="results/scalability.csv", timeout=10.0,
                   n_seeds_ours=500, n_seeds_baselines=50, n_workers=None):
    """Run full scalability experiment with multiprocessing.

    Idempotent: if output_path already exists, completed rows are skipped
    and new results are appended. Safe to kill and restart.

    seeds 0..(n_seeds_baselines-1): run all checkers (bounded, ours, hp, bv, cness)
    seeds n_seeds_baselines..(n_seeds_ours-1): run only bounded + ours
    """

    n_values = list(range(1, 21))  # 1 through 20
    densities = [0.2, 0.4, 0.6]
    indegrees = [2, 3, 5, None]  # None = unbounded

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Resume: skip already-completed experiments
    completed = _load_completed(output_path)
    file_exists = os.path.exists(output_path)

    # Build work list, excluding completed keys
    work = []
    for n in n_values:
        for density in densities:
            for max_indeg in indegrees:
                indeg_str = str(max_indeg) if max_indeg else "inf"
                for seed in range(n_seeds_ours):
                    key = (n, density, indeg_str, seed)
                    if key not in completed:
                        run_baselines = seed < n_seeds_baselines
                        work.append((n, density, max_indeg, seed, timeout, run_baselines))

    total_all = len(n_values) * len(densities) * len(indegrees) * n_seeds_ours
    n_skip = len(completed)
    total = len(work)

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 64)
    print(f"Total experiments: {total_all} | Already done: {n_skip} | "
          f"Remaining: {total}", flush=True)
    if total == 0:
        print("Nothing to do.", flush=True)
        return
    print(f"Running {total} experiments with {n_workers} workers, "
          f"timeout={timeout}s", flush=True)

    t_start = time.perf_counter()
    open_mode = "a" if file_exists else "w"
    with mp.Pool(n_workers) as pool, \
         open(output_path, open_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not file_exists:
            writer.writeheader()

        for i, row in enumerate(pool.imap_unordered(_worker, work), 1):
            writer.writerow({k: (f"{row[k]:.6f}" if k.endswith("_time") and row[k] != ""
                                 else row[k])
                             for k in COLUMNS})
            f.flush()

            if i % 50 == 0 or i == total:
                elapsed = time.perf_counter() - t_start
                print(f"[{i}/{total}] {elapsed:.1f}s elapsed", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\nResults written to {output_path}")
    print(f"Completed: {total} experiments in {elapsed:.1f}s "
          f"({elapsed/60:.1f} min)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AC scalability experiment")
    parser.add_argument("--output", "-o", default="results/scalability.csv",
                        help="Output CSV path")
    parser.add_argument("--timeout", "-t", type=float, default=10.0,
                        help="Timeout per checker per experiment (seconds)")
    parser.add_argument("--seeds-ours", type=int, default=500,
                        help="Number of seeds for bounded+ours checkers")
    parser.add_argument("--seeds-baselines", type=int, default=50,
                        help="Number of seeds for hp/bv/cness checkers")
    parser.add_argument("--workers", "-w", type=int, default=None,
                        help="Number of parallel workers (default: cpu_count)")
    args = parser.parse_args()

    run_experiment(args.output, args.timeout, args.seeds_ours,
                   args.seeds_baselines, args.workers)
