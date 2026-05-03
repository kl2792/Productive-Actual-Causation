"""Beckers & Vennekens (2018) actual causation checker.

Implements the BV definition based on NESS (Necessary Element of a
Sufficient Set) causation:

  Sufficiency: X=x is sufficient for E=e if for ALL assignments to
    V\\(X union {E}), E=e holds under do(X=x, others=anything).

  Direct NESS: C=c directly NESS-causes E=e if there exists W=w (actual
    values, W subset of V\\{C,E}) such that {C=c, W=w} is sufficient for
    E=e but {W=w} alone is not.

  NESS: C=c NESS-causes E=e if there is a chain of direct NESS links
    from C to E through intermediate variables.

  BV-cause: C=c BV-causes E=e if (1) C=c NESS-causes E=e, and (2) there
    exists c' != c such that C=c' does NOT NESS-cause E=e in M_{C=c'}.

Run: python bv_checker.py
"""
from itertools import combinations, product
from causation import SCM


class BVChecker:
    """Beckers & Vennekens (2018) actual causation checker."""

    def __init__(self, scm):
        self.scm = scm
        self.cache = {}

    def is_cause(self, X, x, Y, y, do=frozenset()):
        """Check if X=x is a BV-cause of Y=y."""
        key = (X, x, Y, y, do)
        if key in self.cache:
            return self.cache[key]
        result = self._check(X, x, Y, y, do)
        self.cache[key] = result
        return result

    def all_causes(self, Y, y, do=frozenset()):
        """Find every BV-cause of Y=y."""
        actual = self.scm.eval(dict(do))
        return [(V, actual[V]) for V in self.scm.vars
                if V != Y and self.is_cause(V, actual[V], Y, y, do)]

    def _check(self, X, x, Y, y, do):
        do_d = dict(do)
        actual = self.scm.eval(do_d)

        # Factuality
        if actual.get(X) != x or actual.get(Y) != y:
            return False

        # Condition 1: X=x NESS-causes Y=y
        if not self._ness_causes(X, x, Y, y, actual, do_d):
            return False

        # Condition 2: exists c' != c such that C=c' does NOT NESS-cause
        # Y=y in the submodel M_{C=c'}
        x_prime = 1 - x
        sub_do = {**do_d, X: x_prime}
        sub_actual = self.scm.eval(sub_do)
        if sub_actual[Y] != y:
            # Y doesn't even hold, so trivially c' doesn't NESS-cause
            return True
        if not self._ness_causes(X, x_prime, Y, y, sub_actual, sub_do):
            return True

        return False

    def _is_sufficient(self, fixed, E, e, do_d):
        """Check if fixed assignments are sufficient for E=e.

        fixed: dict {var: val} of variables held fixed.
        Returns True if for ALL assignments to V\\(fixed union {E}),
        E evaluates to e.
        """
        scm = self.scm
        free_vars = [v for v in scm.vars if v not in fixed and v != E]
        for vals in product([0, 1], repeat=len(free_vars)):
            intervention = {**do_d, **fixed, **dict(zip(free_vars, vals))}
            result = scm.eval(intervention)
            if result[E] != e:
                return False
        return True

    def _direct_ness(self, C, c, E, e, actual, do_d):
        """Check if C=c directly NESS-causes E=e.

        Searches for witness W among actual-value assignments to
        V\\{C, E}. For efficiency, W is drawn from parents of E since
        sufficiency of {C=c, W=w} for E=e depends only on the structural
        equation for E (which depends only on parents of E).
        """
        key = ("dness", C, c, E, e, frozenset(do_d.items()))
        if key in self.cache:
            return self.cache[key]

        # W candidates: parents of E minus C (only parents matter for
        # direct influence on E's equation)
        w_candidates = [v for v in self.scm.pa.get(E, [])
                        if v != C and v != E]

        result = False
        for size in range(len(w_candidates) + 1):
            for w_vars in combinations(w_candidates, size):
                W = {v: actual[v] for v in w_vars}
                fixed_with_c = {C: c, **W}
                fixed_without_c = dict(W)

                # {C=c, W=w} sufficient for E=e
                if not self._is_sufficient(fixed_with_c, E, e, do_d):
                    continue
                # {W=w} alone NOT sufficient for E=e
                if self._is_sufficient(fixed_without_c, E, e, do_d):
                    continue
                result = True
                break
            if result:
                break

        self.cache[key] = result
        return result

    def _ness_causes(self, C, c, E, e, actual, do_d):
        """Check if C=c NESS-causes E=e via a chain of direct NESS links.

        Uses BFS through the causal graph. At each step, checks if there
        is a direct NESS link from the current variable to the next.
        """
        key = ("ness", C, c, E, e, frozenset(do_d.items()))
        if key in self.cache:
            return self.cache[key]

        # Direct case
        if self._direct_ness(C, c, E, e, actual, do_d):
            self.cache[key] = True
            return True

        # Chain: C=c directly NESS-causes some intermediate M=m,
        # and M=m NESS-causes E=e
        visited = {C}
        frontier = [C]
        while frontier:
            next_frontier = []
            for node in frontier:
                node_val = actual[node]
                for child in self.scm.ch[node]:
                    if child in visited or child == E:
                        # Check direct NESS to E separately
                        if child == E:
                            if self._direct_ness(node, node_val, E, e,
                                                 actual, do_d):
                                self.cache[key] = True
                                return True
                        continue
                    child_val = actual[child]
                    if self._direct_ness(node, node_val, child, child_val,
                                         actual, do_d):
                        visited.add(child)
                        next_frontier.append(child)
            frontier = next_frontier

        # Check if any reachable intermediate NESS-causes E
        for mid in visited:
            if mid == C:
                continue
            mid_val = actual[mid]
            if self._direct_ness(mid, mid_val, E, e, actual, do_d):
                self.cache[key] = True
                return True

        self.cache[key] = False
        return False


def verify():
    """Verify BV checker on standard examples."""
    print("=" * 60)
    print("  Beckers & Vennekens (2018) Checker — Verification")
    print("=" * 60)

    # 1. Switch: S=1, C1=S, C2=not S, Y=C1|C2
    # BV: S=1 NOT a cause (S=0 also NESS-causes Y=1 via C2)
    scm = SCM(
        equations={
            "S":  lambda p: 1,
            "C1": lambda p: p["S"],
            "C2": lambda p: 1 - p["S"],
            "Y":  lambda p: p["C1"] | p["C2"],
        },
        parents={"S": [], "C1": ["S"], "C2": ["S"], "Y": ["C1", "C2"]},
    )
    bv = BVChecker(scm)
    r = bv.is_cause("S", 1, "Y", 1)
    print(f"\n--- Switch ---")
    print(f"  S=1 causes Y=1?  {r}  (expected: False)")
    assert not r, "Switch: S=1 should NOT be a BV-cause"
    print(f"  All causes: {bv.all_causes('Y', 1)}")

    # 2. Preemption: ST=1, BT=1, SH=ST, BH=BT&~SH, BS=SH|BH
    # BV: ST=1 NOT a cause (known failure — ST=0 NESS-causes BS=1 via Billy)
    scm = SCM(
        equations={
            "ST": lambda p: 1,
            "BT": lambda p: 1,
            "SH": lambda p: p["ST"],
            "BH": lambda p: p["BT"] & (1 - p["SH"]),
            "BS": lambda p: p["SH"] | p["BH"],
        },
        parents={
            "ST": [], "BT": [],
            "SH": ["ST"], "BH": ["BT", "SH"], "BS": ["SH", "BH"],
        },
    )
    bv = BVChecker(scm)
    r = bv.is_cause("ST", 1, "BS", 1)
    print(f"\n--- Preemption ---")
    print(f"  ST=1 causes BS=1?  {r}  (expected: False, known BV failure)")
    assert not r, "Preemption: ST=1 should NOT be a BV-cause (known failure)"
    print(f"  All causes: {bv.all_causes('BS', 1)}")

    # 3. Overdetermination: X1=X2=1, Y=X1|X2
    # BV: X1=1 IS a cause (X1=0 does not NESS-cause Y=1 in M_{X1=0})
    scm = SCM(
        equations={
            "X1": lambda p: 1,
            "X2": lambda p: 1,
            "Y":  lambda p: p["X1"] | p["X2"],
        },
        parents={"X1": [], "X2": [], "Y": ["X1", "X2"]},
    )
    bv = BVChecker(scm)
    r1 = bv.is_cause("X1", 1, "Y", 1)
    r2 = bv.is_cause("X2", 1, "Y", 1)
    print(f"\n--- Overdetermination ---")
    print(f"  X1=1 causes Y=1?  {r1}  (expected: True)")
    print(f"  X2=1 causes Y=1?  {r2}  (expected: True)")
    assert r1, "Overdetermination: X1=1 should be a BV-cause"
    assert r2, "Overdetermination: X2=1 should be a BV-cause"
    print(f"  All causes: {bv.all_causes('Y', 1)}")

    print("\nAll BV verification checks passed.")


if __name__ == "__main__":
    verify()
