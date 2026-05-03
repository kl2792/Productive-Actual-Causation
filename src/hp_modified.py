"""HP Modified 2015 Checker for actual causation.

Implements the Modified HP definition (Halpern 2015):
  AC1 (Factuality): X=x and Y=y in (M,u)
  AC2 (Counterfactual): exists W ⊆ V\\{X,Y} and x'≠x such that Y_{x',w} ≠ y
       where w = actual values of W in (M,u)
  AC3 (Minimality): no strict subset of X satisfies AC1+AC2

X can be a SET of variables (multivariate causes). A singleton variable X=x
is a cause of Y=y iff it appears in some minimal cause set satisfying AC1-AC3.

Key differences from our definition (Def 4.1):
  - No path requirement (our C2)
  - No asymmetry check (our C3)
  - Witness W is always fixed at actual values
  - Allows multivariate cause sets (ours is singleton only)

Run: python hp_modified.py
"""
from itertools import combinations, product
from causation import SCM, Checker


class HPChecker:
    """HP Modified (2015) actual causation checker with multivariate support.

    Checks whether X=x participates in some actual cause of Y=y.
    The cause set S ⊇ {X} must satisfy:
      AC1: S=actual(S) and Y=y hold
      AC2: exists W ⊆ V\\(S∪{Y}) and s'≠s with Y_{s',w} ≠ y (w=actual)
      AC3: no strict subset of S satisfies AC1+AC2
    """

    def __init__(self, scm):
        self.scm = scm
        self.cache = {}

    def is_cause(self, X, x, Y, y, do=frozenset()):
        """Check if X=x is an actual cause of Y=y under HP Modified.

        Returns True iff X appears in some minimal cause set S such that
        S=actual(S) satisfies AC1-AC3 for Y=y.
        """
        key = (X, x, Y, y, do)
        if key in self.cache:
            return self.cache[key]
        result = self._check(X, x, Y, y, do)
        self.cache[key] = result
        return result

    def _check(self, X, x, Y, y, do):
        scm = self.scm
        do_d = dict(do)
        actual = scm.eval(do_d)

        # AC1 (basic): X=x and Y=y must hold
        if actual.get(X) != x or actual.get(Y) != y:
            return False

        # Candidates for cause set: all variables except Y
        candidates = [v for v in scm.vars if v != Y]

        # Try all subsets S containing X, smallest first (for efficiency)
        for size in range(1, len(candidates) + 1):
            for s_vars in combinations(candidates, size):
                if X not in s_vars:
                    continue

                s_vals = tuple(actual[v] for v in s_vars)

                # AC2: exists W ⊆ V \ (S ∪ {Y}) and s' ≠ s with Y_{s',w} ≠ y
                remaining = [v for v in scm.vars
                             if v not in s_vars and v != Y]

                if not self._ac2(scm, s_vars, s_vals, Y, y,
                                 remaining, actual, do_d):
                    continue

                # AC3: no strict subset of S satisfies AC1+AC2
                if not self._ac3(scm, s_vars, s_vals, Y, y, actual, do_d):
                    continue

                return True

        return False

    def _ac2(self, scm, s_vars, s_vals, Y, y, remaining, actual, do_d):
        """Check AC2: exists alternative s' and witness W with Y_{s',w} ≠ y."""
        s_dict = dict(zip(s_vars, s_vals))

        # Enumerate alternative assignments to S
        for s_prime_vals in product([0, 1], repeat=len(s_vars)):
            s_prime = dict(zip(s_vars, s_prime_vals))
            if s_prime == s_dict:
                continue  # s' must differ from s

            # Enumerate witnesses W ⊆ remaining (actual values only)
            for w_size in range(len(remaining) + 1):
                for w_vars in combinations(remaining, w_size):
                    W = {v: actual[v] for v in w_vars}
                    cf_do = {**do_d, **s_prime, **W}
                    cf = scm.eval(cf_do)
                    if cf[Y] != y:
                        return True
        return False

    def _ac3(self, scm, s_vars, s_vals, Y, y, actual, do_d):
        """Check AC3: no strict subset of S satisfies AC1+AC2."""
        for sub_size in range(1, len(s_vars)):
            for sub_vars in combinations(s_vars, sub_size):
                sub_vals = tuple(actual[v] for v in sub_vars)
                sub_remaining = [v for v in scm.vars
                                 if v not in sub_vars and v != Y]

                if self._ac2(scm, sub_vars, sub_vals, Y, y,
                             sub_remaining, actual, do_d):
                    return False  # Subset also works → S not minimal
        return True

    def all_causes(self, Y, y, do=frozenset()):
        """Find every variable that is part of an actual cause of Y=y."""
        actual = self.scm.eval(dict(do))
        return [(V, actual[V]) for V in self.scm.vars
                if V != Y and self.is_cause(V, actual[V], Y, y, do)]


# ---------------------------------------------------------------------------
# Verification on known examples
# ---------------------------------------------------------------------------

def verify():
    """Verify HP Modified on examples with expected results."""

    print("=" * 60)
    print("  HP Modified (2015) Checker — Verification")
    print("  (with multivariate cause support)")
    print("=" * 60)

    # 1. Switch: HP should accept S=1, ours rejects
    scm = SCM(
        equations={
            "S":  lambda p: 1,
            "C1": lambda p: p["S"],
            "C2": lambda p: 1 - p["S"],
            "Y":  lambda p: p["C1"] | p["C2"],
        },
        parents={"S": [], "C1": ["S"], "C2": ["S"], "Y": ["C1", "C2"]},
    )
    hp = HPChecker(scm)
    ours = Checker(scm)
    print("\n--- Switch ---")
    print(f"  HP:   S=1→Y=1? {hp.is_cause('S', 1, 'Y', 1)}")   # True
    print(f"  Ours: S=1→Y=1? {ours.is_cause('S', 1, 'Y', 1)}")  # False
    print(f"  HP all causes:   {hp.all_causes('Y', 1)}")
    print(f"  Ours all causes: {ours.all_causes('Y', 1)}")

    # 2. Overdetermination: both should agree (HP via multivariate set)
    scm = SCM(
        equations={
            "X1": lambda p: 1,
            "X2": lambda p: 1,
            "Y":  lambda p: p["X1"] | p["X2"],
        },
        parents={"X1": [], "X2": [], "Y": ["X1", "X2"]},
    )
    hp = HPChecker(scm)
    ours = Checker(scm)
    print("\n--- Overdetermination ---")
    print(f"  HP:   X1=1→Y=1? {hp.is_cause('X1', 1, 'Y', 1)}")  # True
    print(f"  HP:   X2=1→Y=1? {hp.is_cause('X2', 1, 'Y', 1)}")  # True
    print(f"  Ours: X1=1→Y=1? {ours.is_cause('X1', 1, 'Y', 1)}")  # True
    print(f"  Ours: X2=1→Y=1? {ours.is_cause('X2', 1, 'Y', 1)}")  # True
    print(f"  HP all causes:   {hp.all_causes('Y', 1)}")

    # 3. BV.2 controlled backup (switch): HP accepts, ours rejects
    scm = SCM(
        equations={
            "X1": lambda p: 1,
            "L":  lambda p: p["X1"],
            "R":  lambda p: 1 - p["X1"],
            "Y":  lambda p: p["L"] | p["R"],
        },
        parents={"X1": [], "L": ["X1"], "R": ["X1"], "Y": ["L", "R"]},
    )
    hp = HPChecker(scm)
    ours = Checker(scm)
    print("\n--- BV.2 Controlled Backup (switch) ---")
    print(f"  HP:   X1=1→Y=1? {hp.is_cause('X1', 1, 'Y', 1)}")  # True
    print(f"  Ours: X1=1→Y=1? {ours.is_cause('X1', 1, 'Y', 1)}")  # False
    print(f"  HP all causes:   {hp.all_causes('Y', 1)}")

    # 4. Preemption: both should agree
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
    hp = HPChecker(scm)
    ours = Checker(scm)
    print("\n--- Preemption ---")
    print(f"  HP:   ST=1→BS=1? {hp.is_cause('ST', 1, 'BS', 1)}")  # True
    print(f"  Ours: ST=1→BS=1? {ours.is_cause('ST', 1, 'BS', 1)}")  # True
    print(f"  HP:   BT=1→BS=1? {hp.is_cause('BT', 1, 'BS', 1)}")  # ?
    print(f"  Ours: BT=1→BS=1? {ours.is_cause('BT', 1, 'BS', 1)}")  # False

    # 5. Firing squad: C->R1, C->R2, D=R1|R2
    scm = SCM(
        equations={
            "C":  lambda p: 1,
            "R1": lambda p: p["C"],
            "R2": lambda p: p["C"],
            "D":  lambda p: p["R1"] | p["R2"],
        },
        parents={"C": [], "R1": ["C"], "R2": ["C"], "D": ["R1", "R2"]},
    )
    hp = HPChecker(scm)
    ours = Checker(scm)
    print("\n--- Firing Squad ---")
    print(f"  HP:   C=1→D=1?  {hp.is_cause('C', 1, 'D', 1)}")
    print(f"  HP:   R1=1→D=1? {hp.is_cause('R1', 1, 'D', 1)}")
    print(f"  HP:   R2=1→D=1? {hp.is_cause('R2', 1, 'D', 1)}")
    print(f"  Ours: C=1→D=1?  {ours.is_cause('C', 1, 'D', 1)}")
    print(f"  Ours: R1=1→D=1? {ours.is_cause('R1', 1, 'D', 1)}")
    print(f"  Ours: R2=1→D=1? {ours.is_cause('R2', 1, 'D', 1)}")

    print("\nDone.")


if __name__ == "__main__":
    verify()
