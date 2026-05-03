"""Comprehensive tests for all four causation checkers.

Tests our definition (Checker), HP Modified (HPChecker), BV (BVChecker),
and CNESS (CNESSChecker) against seven canonical examples from the paper.
"""
import pytest
from causation import SCM, Checker
from baselines.hp_modified import HPChecker
from baselines.bv_checker import BVChecker
from baselines.cness_checker import CNESSChecker


# ---------------------------------------------------------------------------
# SCM builders (one per example)
# ---------------------------------------------------------------------------

def make_switch():
    """Switch: S=1, C1=S, C2=not S, Y=C1|C2."""
    return SCM(
        equations={
            "S":  lambda p: 1,
            "C1": lambda p: p["S"],
            "C2": lambda p: 1 - p["S"],
            "Y":  lambda p: p["C1"] | p["C2"],
        },
        parents={"S": [], "C1": ["S"], "C2": ["S"], "Y": ["C1", "C2"]},
    )


def make_preemption():
    """Late preemption: ST=1, BT=1, SH=ST, BH=BT & ~SH, BS=SH|BH."""
    return SCM(
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


def make_overdetermination():
    """Symmetric overdetermination: X1=X2=1, Y=X1|X2."""
    return SCM(
        equations={
            "X1": lambda p: 1,
            "X2": lambda p: 1,
            "Y":  lambda p: p["X1"] | p["X2"],
        },
        parents={"X1": [], "X2": [], "Y": ["X1", "X2"]},
    )


def make_chain():
    """Chain: A=1, B=A, C=B, Y=C."""
    return SCM(
        equations={
            "A": lambda p: 1,
            "B": lambda p: p["A"],
            "C": lambda p: p["B"],
            "Y": lambda p: p["C"],
        },
        parents={"A": [], "B": ["A"], "C": ["B"], "Y": ["C"]},
    )


def make_firing_squad():
    """Firing squad: C=1, R1=C, R2=C, D=R1|R2."""
    return SCM(
        equations={
            "C":  lambda p: 1,
            "R1": lambda p: p["C"],
            "R2": lambda p: p["C"],
            "D":  lambda p: p["R1"] | p["R2"],
        },
        parents={"C": [], "R1": ["C"], "R2": ["C"], "D": ["R1", "R2"]},
    )


def make_reduced_switch():
    """Reduced switch: X=1, W=not X, Y=X|W."""
    return SCM(
        equations={
            "X": lambda p: 1,
            "W": lambda p: 1 - p["X"],
            "Y": lambda p: p["X"] | p["W"],
        },
        parents={"X": [], "W": ["X"], "Y": ["X", "W"]},
    )


def make_reviewer_counterexample():
    """Reviewer's counterexample: S=1, A=1, B=1, C1=S&A, C2=~S&B, Y=C1|C2."""
    return SCM(
        equations={
            "S":  lambda p: 1,
            "A":  lambda p: 1,
            "B":  lambda p: 1,
            "C1": lambda p: p["S"] & p["A"],
            "C2": lambda p: (1 - p["S"]) & p["B"],
            "Y":  lambda p: p["C1"] | p["C2"],
        },
        parents={
            "S": [], "A": [], "B": [],
            "C1": ["S", "A"], "C2": ["S", "B"], "Y": ["C1", "C2"],
        },
    )




# ---------------------------------------------------------------------------
# 1. Switch
# ---------------------------------------------------------------------------

class TestSwitch:
    """Switch: S=1, C1=S, C2=not S, Y=C1|C2."""

    @pytest.fixture
    def scm(self):
        return make_switch()

    # -- Ours --
    def test_ours_s_not_cause(self, scm):
        """Our definition: S=1 is NOT a cause of Y=1 (C3 blocks it)."""
        assert not Checker(scm).is_cause("S", 1, "Y", 1)

    def test_ours_c1_is_cause(self, scm):
        """Our definition: C1=1 IS a cause of Y=1."""
        assert Checker(scm).is_cause("C1", 1, "Y", 1)

    # -- HP Modified --
    def test_hp_s_is_cause(self, scm):
        """HP Modified: S=1 IS a cause of Y=1 (no asymmetry check)."""
        assert HPChecker(scm).is_cause("S", 1, "Y", 1)

    def test_hp_c1_is_cause(self, scm):
        """HP Modified: C1=1 IS a cause of Y=1."""
        assert HPChecker(scm).is_cause("C1", 1, "Y", 1)

    # -- BV --
    def test_bv_s_not_cause(self, scm):
        """BV: S=1 is NOT a cause (S=0 also NESS-causes Y=1 via C2)."""
        assert not BVChecker(scm).is_cause("S", 1, "Y", 1)

    def test_bv_c1_is_cause(self, scm):
        """BV: C1=1 IS a cause of Y=1."""
        assert BVChecker(scm).is_cause("C1", 1, "Y", 1)

    # -- CNESS --
    def test_cness_s_is_cause(self, scm):
        """CNESS: S=1 IS a cause (known regression, path-restricted check)."""
        assert CNESSChecker(scm).is_cause("S", 1, "Y", 1)

    def test_cness_c1_is_cause(self, scm):
        """CNESS: C1=1 IS a cause of Y=1."""
        assert CNESSChecker(scm).is_cause("C1", 1, "Y", 1)


# ---------------------------------------------------------------------------
# 2. Late preemption
# ---------------------------------------------------------------------------

class TestPreemption:
    """Late preemption (Billy/Suzy): ST=1, BT=1, BS=1."""

    @pytest.fixture
    def scm(self):
        return make_preemption()

    # -- Ours --
    def test_ours_st_is_cause(self, scm):
        """Our definition: ST=1 IS a cause of BS=1."""
        assert Checker(scm).is_cause("ST", 1, "BS", 1)

    def test_ours_bt_not_cause(self, scm):
        """Our definition: BT=1 is NOT a cause of BS=1."""
        assert not Checker(scm).is_cause("BT", 1, "BS", 1)

    # -- HP Modified --
    def test_hp_st_is_cause(self, scm):
        """HP Modified: ST=1 IS a cause of BS=1."""
        assert HPChecker(scm).is_cause("ST", 1, "BS", 1)

    def test_hp_bt_not_cause(self, scm):
        """HP Modified: BT=1 is NOT a cause of BS=1."""
        assert not HPChecker(scm).is_cause("BT", 1, "BS", 1)

    # -- BV --
    def test_bv_st_not_cause(self, scm):
        """BV: ST=1 is NOT a cause (known BV failure on preemption)."""
        assert not BVChecker(scm).is_cause("ST", 1, "BS", 1)

    def test_bv_bt_not_cause(self, scm):
        """BV: BT=1 is NOT a cause of BS=1."""
        assert not BVChecker(scm).is_cause("BT", 1, "BS", 1)

    # -- CNESS --
    def test_cness_st_is_cause(self, scm):
        """CNESS: ST=1 IS a cause of BS=1."""
        assert CNESSChecker(scm).is_cause("ST", 1, "BS", 1)

    def test_cness_bt_not_cause(self, scm):
        """CNESS: BT=1 is NOT a cause of BS=1."""
        assert not CNESSChecker(scm).is_cause("BT", 1, "BS", 1)


# ---------------------------------------------------------------------------
# 3. Symmetric overdetermination
# ---------------------------------------------------------------------------

class TestOverdetermination:
    """Symmetric overdetermination: X1=X2=1, Y=X1|X2."""

    @pytest.fixture
    def scm(self):
        return make_overdetermination()

    @pytest.mark.parametrize("checker_cls", [Checker, HPChecker, BVChecker, CNESSChecker])
    def test_x1_is_cause(self, scm, checker_cls):
        """All definitions: X1=1 IS a cause of Y=1."""
        assert checker_cls(scm).is_cause("X1", 1, "Y", 1)

    @pytest.mark.parametrize("checker_cls", [Checker, HPChecker, BVChecker, CNESSChecker])
    def test_x2_is_cause(self, scm, checker_cls):
        """All definitions: X2=1 IS a cause of Y=1."""
        assert checker_cls(scm).is_cause("X2", 1, "Y", 1)


# ---------------------------------------------------------------------------
# 4. Chain
# ---------------------------------------------------------------------------

class TestChain:
    """Chain: A=1, B=A, C=B, Y=C. All intermediates are causes."""

    @pytest.fixture
    def scm(self):
        return make_chain()

    @pytest.mark.parametrize("var", ["A", "B", "C"])
    @pytest.mark.parametrize("checker_cls", [Checker, HPChecker, BVChecker, CNESSChecker])
    def test_all_causes(self, scm, var, checker_cls):
        """All definitions: every variable on the chain IS a cause of Y=1."""
        assert checker_cls(scm).is_cause(var, 1, "Y", 1)


# ---------------------------------------------------------------------------
# 5. Firing squad
# ---------------------------------------------------------------------------

class TestFiringSquad:
    """Firing squad: C=1, R1=C, R2=C, D=R1|R2."""

    @pytest.fixture
    def scm(self):
        return make_firing_squad()

    @pytest.mark.parametrize("var", ["C", "R1", "R2"])
    @pytest.mark.parametrize("checker_cls", [Checker, HPChecker, BVChecker, CNESSChecker])
    def test_all_causes(self, scm, var, checker_cls):
        """All definitions: C, R1, R2 are all causes of D=1."""
        assert checker_cls(scm).is_cause(var, 1, "D", 1)


# ---------------------------------------------------------------------------
# 6. Reduced switch
# ---------------------------------------------------------------------------

class TestReducedSwitch:
    """Reduced switch: X=1, W=not X, Y=X|W."""

    @pytest.fixture
    def scm(self):
        return make_reduced_switch()

    # -- Ours --
    def test_ours_x_not_cause(self, scm):
        """Our definition: X=1 is NOT a cause (C3 blocks; X=0 also causes Y=1)."""
        assert not Checker(scm).is_cause("X", 1, "Y", 1)

    # -- HP Modified --
    def test_hp_x_is_cause(self, scm):
        """HP Modified: X=1 IS a cause (no asymmetry check)."""
        assert HPChecker(scm).is_cause("X", 1, "Y", 1)

    # -- BV --
    def test_bv_x_not_cause(self, scm):
        """BV: X=1 is NOT a cause (X=0 also NESS-causes Y=1)."""
        assert not BVChecker(scm).is_cause("X", 1, "Y", 1)

    # -- CNESS --
    def test_cness_x_is_cause(self, scm):
        """CNESS: X=1 IS a cause (known regression, same as switch)."""
        assert CNESSChecker(scm).is_cause("X", 1, "Y", 1)


# ---------------------------------------------------------------------------
# 7. Reviewer's counterexample
# ---------------------------------------------------------------------------

class TestReviewerCounterexample:
    """Reviewer's counterexample: S=1, A=1, B=1, C1=S&A, C2=~S&B, Y=C1|C2.

    Actual values: S=1, A=1, B=1, C1=1, C2=0, Y=1.
    """

    @pytest.fixture
    def scm(self):
        return make_reviewer_counterexample()

    # -- Ours --
    def test_ours_b_not_cause(self, scm):
        """Our definition: B=1 is NOT a cause (C2=0, so B has no active path)."""
        assert not Checker(scm).is_cause("B", 1, "Y", 1)

    def test_ours_s_not_cause(self, scm):
        """Our definition: S=1 is NOT a cause (C3 blocks)."""
        assert not Checker(scm).is_cause("S", 1, "Y", 1)

    def test_ours_a_is_cause(self, scm):
        """Our definition: A=1 IS a cause of Y=1."""
        assert Checker(scm).is_cause("A", 1, "Y", 1)

    def test_ours_c1_is_cause(self, scm):
        """Our definition: C1=1 IS a cause of Y=1."""
        assert Checker(scm).is_cause("C1", 1, "Y", 1)

    # -- HP Modified --
    def test_hp_b_not_cause(self, scm):
        """HP Modified: B=1 is NOT a cause.

        {S,B} fails AC3: the subset {S} alone satisfies AC2 with W={C2:0},
        so {S,B} is not minimal. No other set containing B works either.
        """
        assert not HPChecker(scm).is_cause("B", 1, "Y", 1)

    # -- BV --
    def test_bv_b_not_cause(self, scm):
        """BV: B=1 is NOT a cause."""
        assert not BVChecker(scm).is_cause("B", 1, "Y", 1)

    def test_bv_s_not_cause(self, scm):
        """BV: S=1 is NOT a cause."""
        assert not BVChecker(scm).is_cause("S", 1, "Y", 1)

    # -- CNESS --
    def test_cness_s_is_cause(self, scm):
        """CNESS: S=1 IS a cause (known regression)."""
        assert CNESSChecker(scm).is_cause("S", 1, "Y", 1)

    def test_cness_b_not_cause(self, scm):
        """CNESS: B=1 is NOT a cause.

        B's only path to Y is (B, C2, Y), but C2=0 so B=1 does not
        directly NESS-cause C2=0 in a way that propagates to Y=1.
        """
        assert not CNESSChecker(scm).is_cause("B", 1, "Y", 1)


# ---------------------------------------------------------------------------
# 8. SCM.eval() cache
# ---------------------------------------------------------------------------

class TestEvalCache:
    """Unit tests for SCM._eval_cache added in commit 20564ec."""

    @pytest.fixture
    def scm(self):
        return make_switch()

    def test_cache_hit_returns_equal_dict(self, scm):
        """Same do-intervention returns the identical cached dict on second call."""
        result1 = scm.eval({"S": 0})
        result2 = scm.eval({"S": 0})
        assert result1 == result2
        assert result1 is result2  # same object, not a copy

    def test_cache_grows_for_distinct_interventions(self, scm):
        """Each distinct frozenset key gets its own cache entry."""
        scm.eval({})
        scm.eval({"S": 0})
        scm.eval({"S": 1, "C1": 0})
        assert len(scm._eval_cache) == 3

    def test_cache_does_not_mix_interventions(self, scm):
        """do={S:0} and do={S:1} produce different cached values."""
        v0 = scm.eval({"S": 0})
        v1 = scm.eval({"S": 1})
        assert v0["C1"] == 0
        assert v1["C1"] == 1

    def test_cache_empty_do_and_none_share_key(self, scm):
        """eval(None) and eval({}) use the same frozenset() key."""
        r_none = scm.eval(None)
        r_empty = scm.eval({})
        assert r_none is r_empty

    def test_cache_isolated_across_instances(self):
        """Two SCM instances built from identical specs have separate caches."""
        scm_a = make_switch()
        scm_b = make_switch()
        scm_a.eval({"S": 0})
        assert len(scm_a._eval_cache) == 1
        assert len(scm_b._eval_cache) == 0

    def test_cache_correctness_regression(self, scm):
        """Cached values are numerically correct (regression against live eval)."""
        # Actual world (no intervention): S=1, C1=1, C2=0, Y=1
        actual = scm.eval({})
        assert actual == {"S": 1, "C1": 1, "C2": 0, "Y": 1}
        # Counterfactual do(S=0): C1=0, C2=1, Y=1
        cf = scm.eval({"S": 0})
        assert cf == {"S": 0, "C1": 0, "C2": 1, "Y": 1}
        # Second call still correct after cache population
        assert scm.eval({"S": 0}) == {"S": 0, "C1": 0, "C2": 1, "Y": 1}

    def test_cache_partial_intervention_correctness(self):
        """Partial intervention (only some vars forced) is cached and correct."""
        scm = make_chain()  # A->B->C->Y, all 1
        # Force A=0: should propagate B=0, C=0, Y=0
        result = scm.eval({"A": 0})
        assert result == {"A": 0, "B": 0, "C": 0, "Y": 0}
        # Cached second call
        assert scm.eval({"A": 0}) is result
