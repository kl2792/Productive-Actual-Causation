"""Bounded-degree causation checker exploiting the witness frontier bound.

Prop 5.4 (Witness Frontier): For path pi = [X, P2, ..., Pm, Y], define
  F(pi) = union_{P in pi} (Pa(P) \\ pi)
Any witness W satisfying C1-C2 for path pi can be replaced by W' in F(pi).

Non-descendant restriction (Def 4.1): W must be in nd_G(X) \\ {Y}.
When F(pi) ⊆ nd_G(X), the frontier is complete and the search space is F(pi).
When F(pi) contains descendants of X, the frontier replacement may violate
nd, so the checker falls back to searching nd_G(X) \\ {Y} for that path.

With bounded max in-degree k and longest path L, both |F(pi)| and
|nd_G(X) ∩ Anc(Y)| are O(k^L), so the checker runs in polynomial time
(Cor 5.5).

Run: python bounded_checker.py
"""
from itertools import combinations, product

from causation import SCM, Checker


class BoundedChecker(Checker):
    """Causation checker using path-first witness frontier enumeration.

    Identical results to Checker. Poly-time when max in-degree k and
    longest path L are bounded.

    Key structural change from Checker:
      Checker:        for witness (over nd_G(X)\\{Y})  -> for path
      BoundedChecker: for path -> for witness in F(pi) ∩ nd_G(X)
                      (falls back to nd_G(X)\\{Y} when F(pi) ⊄ nd_G(X))
    """

    def __init__(self, scm):
        super().__init__(scm)
        self.frontier_max = 0  # diagnostic: max |F(pi)| seen

    def _evaluate(self, X, x, Y, y, do, full):
        """Core evaluation: path-first, frontier-restricted witnesses."""
        scm = self.scm
        do_d = dict(do)
        actual = scm.eval(do_d)

        # C0: factual check
        if actual.get(X) != x or actual.get(Y) != y:
            return False

        # Non-descendant restriction (Def 4.1): W ⊆ nd_G(X) \ {Y}
        desc_X = scm.descendants(X)
        nd_set = frozenset(
            v for v in scm.vars if v not in (X, Y) and v not in desc_X
        )
        nd_others = sorted(nd_set)  # fallback search space

        x_p = 1 - x  # binary flip

        for path in scm.paths(X, Y):
            # Compute witness frontier F(pi):
            # non-path parents of ALL path nodes (including X).
            # Bound: |F(pi)| <= k + (k-1)L for in-degree k, path length L.
            path_set = set(path)
            frontier = set()
            for node in path:
                for pa in scm.pa.get(node, []):
                    if pa not in path_set:
                        frontier.add(pa)

            # Witness search space selection (Prop 5.4 + nd restriction):
            # - If F(pi) ⊆ nd_G(X): frontier is complete, search F(pi)
            # - Otherwise: frontier contains descendants, so the frontier
            #   replacement may violate nd. Fall back to nd_G(X) \ {Y}.
            #   Example: preemption path ST→SH→BS has F={BH} ⊆ desc(ST),
            #   but valid witness BT ∈ nd_G(ST) is outside F.
            if frontier <= nd_set:
                search_vars = sorted(frontier)
            else:
                search_vars = nd_others

            self.frontier_max = max(self.frontier_max, len(search_vars))

            # Enumerate witnesses W subset of search_vars with binary values
            for size in range(len(search_vars) + 1):
                for w_vars in combinations(search_vars, size):
                    for w_vals in product([0, 1], repeat=len(w_vars)):
                        W = dict(zip(w_vars, w_vals))

                        # -- C1: counterfactual dependence --
                        self.stats["c1"] += 1
                        cf_do = {**do_d, X: x_p, **W}
                        cf = scm.eval(cf_do)
                        if cf[Y] == y:
                            continue  # Y didn't flip

                        # C1 minimality: no strict subset of {X} union W
                        # also flips Y
                        full_intv = {X: x_p, **W}
                        if not self._is_minimal(full_intv, do_d, Y, y):
                            continue

                        # -- C2: intermediates change and are causes --
                        # Disjointness automatic: F(pi) ∩ path = empty
                        w_vars_set = set(W.keys())
                        if not self._c2_check(path, actual, cf, Y, y,
                                              do, w_vars_set):
                            continue

                        # -- C3: asymmetric causation --
                        if full:
                            self.stats["c3_deps"] += 1
                            cf_model = frozenset({**do_d, X: x_p}.items())
                            if self._check(X, x_p, Y, y, cf_model,
                                           full=False):
                                continue  # x' satisfies C012 -> C3 fails

                        return True  # All checks passed

        return False


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------

def verify():
    """Verify BoundedChecker matches Checker on all built-in examples."""
    from causation import (
        example_switch, example_overdetermination, example_chain,
        example_chain_with_side_inputs, example_preemption,
        example_fire_warehouse, example_bv2_controlled_backup,
        example_bv2_full, EXPECTED,
    )

    print("=" * 60)
    print("  BoundedChecker vs Checker — Verification")
    print("=" * 60)

    examples = [
        ("switch", {
            "S": [], "C1": ["S"], "C2": ["S"], "Y": ["C1", "C2"],
        }, {
            "S": lambda p: 1, "C1": lambda p: p["S"],
            "C2": lambda p: 1 - p["S"], "Y": lambda p: p["C1"] | p["C2"],
        }),
        ("overdetermination", {
            "X1": [], "X2": [], "Y": ["X1", "X2"],
        }, {
            "X1": lambda p: 1, "X2": lambda p: 1,
            "Y": lambda p: p["X1"] | p["X2"],
        }),
        ("chain", {
            "A": [], "B": ["A"], "C": ["B"], "Y": ["C"],
        }, {
            "A": lambda p: 1, "B": lambda p: p["A"],
            "C": lambda p: p["B"], "Y": lambda p: p["C"],
        }),
        ("chain_side", {
            "A": [], "Z1": [], "Z2": [],
            "B": ["A", "Z1"], "C": ["B", "Z2"], "Y": ["C"],
        }, {
            "A": lambda p: 1, "Z1": lambda p: 1, "Z2": lambda p: 1,
            "B": lambda p: p["A"] & p["Z1"],
            "C": lambda p: p["B"] & p["Z2"], "Y": lambda p: p["C"],
        }),
        ("preemption", {
            "ST": [], "BT": [],
            "SH": ["ST"], "BH": ["BT", "SH"], "BS": ["SH", "BH"],
        }, {
            "ST": lambda p: 1, "BT": lambda p: 1,
            "SH": lambda p: p["ST"],
            "BH": lambda p: p["BT"] & (1 - p["SH"]),
            "BS": lambda p: p["SH"] | p["BH"],
        }),
        ("fire_warehouse", {
            "X1": [], "S": ["X1"], "X2": ["X1"],
            "X3": ["S"], "X4": ["S"], "Y": ["X2", "X3", "X4"],
        }, {
            "X1": lambda p: 1, "S": lambda p: p["X1"],
            "X2": lambda p: p["X1"], "X3": lambda p: p["S"],
            "X4": lambda p: 1 - p["S"],
            "Y": lambda p: p["X2"] | p["X3"] | p["X4"],
        }),
        ("bv2_controlled", {
            "X1": [], "L": ["X1"], "R": ["X1"], "Y": ["L", "R"],
        }, {
            "X1": lambda p: 1, "L": lambda p: p["X1"],
            "R": lambda p: 1 - p["X1"], "Y": lambda p: p["L"] | p["R"],
        }),
        ("bv2_full", {
            "X1": [], "S": ["X1"], "L": ["S"],
            "R": ["S"], "Y": ["X1", "L", "R"],
        }, {
            "X1": lambda p: 1, "S": lambda p: p["X1"],
            "L": lambda p: p["S"], "R": lambda p: 1 - p["S"],
            "Y": lambda p: p["X1"] | p["L"] | p["R"],
        }),
    ]

    total, match, mismatch = 0, 0, 0
    for name, parents, equations in examples:
        scm = SCM(equations, parents)
        ck = Checker(scm)
        bk = BoundedChecker(scm)

        actual = scm.eval()
        Y = "Y" if "Y" in scm.vars else "BS" if "BS" in scm.vars else "FF"

        for V in scm.vars:
            if V == Y:
                continue
            total += 1
            r_ck = ck.is_cause(V, actual[V], Y, actual[Y])
            r_bk = bk.is_cause(V, actual[V], Y, actual[Y])
            if r_ck == r_bk:
                match += 1
            else:
                mismatch += 1
                print(f"  MISMATCH [{name}] {V}={actual[V]} -> {Y}={actual[Y]}: "
                      f"Checker={r_ck}, BoundedChecker={r_bk}")

        print(f"  [{name}] frontier_max={bk.frontier_max}")

    print(f"\n  {total} queries: {match} match, {mismatch} mismatch")
    if mismatch == 0:
        print("  All queries match.")
    else:
        print("  ** MISMATCHES FOUND **")


if __name__ == "__main__":
    verify()
