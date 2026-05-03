"""Beckers (2021) CNESS actual causation checker.

Implements the CNESS definition, which refines BV (2018) by requiring
NESS causation along specific causal paths:

  Sufficiency, Direct NESS, NESS: same as BV (2018).

  NESS along path: C=c NESS-causes E=e along path p=(C, P1, ..., Pn, E)
    if C=c directly NESS-causes P1=p1, P1=p1 directly NESS-causes P2=p2,
    ..., Pn=pn directly NESS-causes E=e (all at actual values).

  CNESS-cause: C=c CNESS-causes E=e if (1) C=c NESS-causes E=e along
    some path p, and (2) there exists c' != c such that C=c' does NOT
    NESS-cause E=e along any subpath of p in M_{C=c'}.

  A subpath of p is a path whose variables are all members of p.

Run: python cness_checker.py
"""
from itertools import combinations, product
from causation import SCM


class CNESSChecker:
    """Beckers (2021) CNESS actual causation checker."""

    def __init__(self, scm):
        self.scm = scm
        self.cache = {}

    def is_cause(self, X, x, Y, y, do=frozenset()):
        """Check if X=x is a CNESS-cause of Y=y."""
        key = (X, x, Y, y, do)
        if key in self.cache:
            return self.cache[key]
        result = self._check(X, x, Y, y, do)
        self.cache[key] = result
        return result

    def all_causes(self, Y, y, do=frozenset()):
        """Find every CNESS-cause of Y=y."""
        actual = self.scm.eval(dict(do))
        return [(V, actual[V]) for V in self.scm.vars
                if V != Y and self.is_cause(V, actual[V], Y, y, do)]

    def _check(self, X, x, Y, y, do):
        do_d = dict(do)
        actual = self.scm.eval(do_d)

        # Factuality
        if actual.get(X) != x or actual.get(Y) != y:
            return False

        # Find all directed paths from X to Y
        paths = self.scm.paths(X, Y)
        if not paths:
            return False

        x_prime = 1 - x

        for path in paths:
            # Condition 1: X=x NESS-causes Y=y along this path
            if not self._ness_along_path(path, actual, do_d):
                continue

            # Condition 2: exists c' != c such that C=c' does NOT
            # NESS-cause E=e along any subpath of path in M_{C=c'}
            sub_do = {**do_d, X: x_prime}
            sub_actual = self.scm.eval(sub_do)

            if sub_actual[Y] != y:
                # Y doesn't hold at all under c', so trivially no
                # NESS causation
                return True

            # Get all subpaths of path that start at X and end at Y
            subpaths = self._subpaths(path, X, Y)

            # Check: c' must NOT NESS-cause E=e along ANY subpath
            c_prime_ness_any = False
            for sp in subpaths:
                if self._ness_along_path(sp, sub_actual, sub_do):
                    c_prime_ness_any = True
                    break

            if not c_prime_ness_any:
                return True

        return False

    def _is_sufficient(self, fixed, E, e, do_d):
        """Check if fixed assignments are sufficient for E=e."""
        scm = self.scm
        free_vars = [v for v in scm.vars if v not in fixed and v != E]
        for vals in product([0, 1], repeat=len(free_vars)):
            intervention = {**do_d, **fixed, **dict(zip(free_vars, vals))}
            result = scm.eval(intervention)
            if result[E] != e:
                return False
        return True

    def _direct_ness(self, C, c, E, e, actual, do_d):
        """Check if C=c directly NESS-causes E=e."""
        key = ("dness", C, c, E, e, frozenset(do_d.items()))
        if key in self.cache:
            return self.cache[key]

        w_candidates = [v for v in self.scm.pa.get(E, [])
                        if v != C and v != E]

        result = False
        for size in range(len(w_candidates) + 1):
            for w_vars in combinations(w_candidates, size):
                W = {v: actual[v] for v in w_vars}
                fixed_with_c = {C: c, **W}
                fixed_without_c = dict(W)

                if not self._is_sufficient(fixed_with_c, E, e, do_d):
                    continue
                if self._is_sufficient(fixed_without_c, E, e, do_d):
                    continue
                result = True
                break
            if result:
                break

        self.cache[key] = result
        return result

    def _ness_along_path(self, path, actual, do_d):
        """Check if there is a chain of direct NESS links along path.

        path = [C, P1, P2, ..., Pn, E]
        Requires: C=c dNESS P1=p1, P1=p1 dNESS P2=p2, ..., Pn=pn dNESS E=e.
        """
        for i in range(len(path) - 1):
            src = path[i]
            dst = path[i + 1]
            src_val = actual[src]
            dst_val = actual[dst]
            if not self._direct_ness(src, src_val, dst, dst_val,
                                     actual, do_d):
                return False
        return True

    def _subpaths(self, path, src, dst):
        """Find all subpaths of path from src to dst.

        A subpath uses only variables in path and forms a valid directed
        path in the causal graph.
        """
        path_set = set(path)
        # Build subgraph restricted to path variables
        adj = {v: [] for v in path}
        for v in path:
            for ch in self.scm.ch[v]:
                if ch in path_set:
                    adj[v].append(ch)

        # Enumerate all simple paths in this subgraph from src to dst
        result = []
        self._enum_paths(adj, src, dst, [src], set([src]), result)
        return result

    def _enum_paths(self, adj, cur, dst, path, visited, result):
        """DFS enumeration of simple paths."""
        if cur == dst:
            result.append(list(path))
            return
        for nxt in adj[cur]:
            if nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                self._enum_paths(adj, nxt, dst, path, visited, result)
                path.pop()
                visited.discard(nxt)


def verify():
    """Verify CNESS checker on standard examples."""
    print("=" * 60)
    print("  Beckers (2021) CNESS Checker — Verification")
    print("=" * 60)

    # 1. Switch: S=1, C1=S, C2=not S, Y=C1|C2
    # CNESS: S=1 IS a cause (known regression).
    # S=1 NESS-causes Y=1 along (S,C1,Y). In M_{S=0}, the only subpath
    # using variables from (S,C1,Y) is (S,C1,Y) itself — S=0 does NOT
    # NESS-cause Y=1 along (S,C1,Y) because C1=0 does not directly
    # NESS-cause Y=1.
    scm = SCM(
        equations={
            "S":  lambda p: 1,
            "C1": lambda p: p["S"],
            "C2": lambda p: 1 - p["S"],
            "Y":  lambda p: p["C1"] | p["C2"],
        },
        parents={"S": [], "C1": ["S"], "C2": ["S"], "Y": ["C1", "C2"]},
    )
    ck = CNESSChecker(scm)
    r = ck.is_cause("S", 1, "Y", 1)
    print(f"\n--- Switch ---")
    print(f"  S=1 causes Y=1?  {r}  (expected: True, known CNESS regression)")
    assert r, "Switch: S=1 should be a CNESS-cause (known regression)"
    print(f"  All causes: {ck.all_causes('Y', 1)}")

    # 2. Preemption: ST=1, BT=1, SH=ST, BH=BT&~SH, BS=SH|BH
    # CNESS: ST=1 IS a cause (correct).
    # ST=1 NESS along (ST,SH,BS). In M_{ST=0}, ST=0 NESS along
    # (ST,SH,BS)? SH=0, BH=1, BS=1. SH=0 dNESS BS=1? No (BH=1
    # makes BS=1 regardless). So ST=0 does NOT NESS along (ST,SH,BS).
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
    ck = CNESSChecker(scm)
    r = ck.is_cause("ST", 1, "BS", 1)
    print(f"\n--- Preemption ---")
    print(f"  ST=1 causes BS=1?  {r}  (expected: True)")
    assert r, "Preemption: ST=1 should be a CNESS-cause"
    r_bt = ck.is_cause("BT", 1, "BS", 1)
    print(f"  BT=1 causes BS=1?  {r_bt}")
    print(f"  All causes: {ck.all_causes('BS', 1)}")

    # 3. Overdetermination: X1=X2=1, Y=X1|X2
    # CNESS: X1=1 IS a cause.
    scm = SCM(
        equations={
            "X1": lambda p: 1,
            "X2": lambda p: 1,
            "Y":  lambda p: p["X1"] | p["X2"],
        },
        parents={"X1": [], "X2": [], "Y": ["X1", "X2"]},
    )
    ck = CNESSChecker(scm)
    r1 = ck.is_cause("X1", 1, "Y", 1)
    r2 = ck.is_cause("X2", 1, "Y", 1)
    print(f"\n--- Overdetermination ---")
    print(f"  X1=1 causes Y=1?  {r1}  (expected: True)")
    print(f"  X2=1 causes Y=1?  {r2}  (expected: True)")
    assert r1, "Overdetermination: X1=1 should be a CNESS-cause"
    assert r2, "Overdetermination: X2=1 should be a CNESS-cause"
    print(f"  All causes: {ck.all_causes('Y', 1)}")

    print("\nAll CNESS verification checks passed.")


if __name__ == "__main__":
    verify()
