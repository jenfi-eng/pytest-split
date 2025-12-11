import itertools
from collections import namedtuple
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import List, Set

    from _pytest.nodes import Item

from pytest_split.algorithms import (
    AlgorithmBase,
    Algorithms,
)

item = namedtuple("item", "nodeid")  # noqa: PYI024


class TestAlgorithms:
    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_test(self, algo_name):
        durations = {"a": 1, "b": 1, "c": 1}
        items = [item(x) for x in durations]
        algo = Algorithms[algo_name].value
        first, second, third = algo(splits=3, items=items, durations=durations)

        # each split should have one test
        assert first.selected == [item("a")]
        assert first.deselected == [item("b"), item("c")]
        assert first.duration == 1

        assert second.selected == [item("b")]
        assert second.deselected == [item("a"), item("c")]
        assert second.duration == 1

        assert third.selected == [item("c")]
        assert third.deselected == [item("a"), item("b")]
        assert third.duration == 1

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_tests_handles_tests_in_durations_but_missing_from_items(
        self, algo_name
    ):
        durations = {"a": 1, "b": 1}
        items = [item(x) for x in ["a"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("a")]
        assert second.selected == []

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__split_tests_handles_tests_with_missing_durations(self, algo_name):
        durations = {"a": 1}
        items = [item(x) for x in ["a", "b"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("a")]
        assert second.selected == [item("b")]

    def test__split_test_handles_large_duration_at_end(self):
        """NOTE: only least_duration does this correctly"""
        durations = {"a": 1, "b": 1, "c": 1, "d": 3}
        items = [item(x) for x in ["a", "b", "c", "d"]]
        algo = Algorithms["least_duration"].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        assert first.selected == [item("d")]
        assert second.selected == [item(x) for x in ["a", "b", "c"]]

    @pytest.mark.parametrize(
        ("algo_name", "expected"),
        [
            ("duration_based_chunks", [[item("a"), item("b")], [item("c"), item("d")]]),
            ("least_duration", [[item("a"), item("c")], [item("b"), item("d")]]),
        ],
    )
    def test__split_tests_calculates_avg_test_duration_only_on_present_tests(
        self, algo_name, expected
    ):
        # If the algo includes test e's duration to calculate the averge then
        # a will be expected to take a long time, and so 'a' will become its
        # own group. Intended behaviour is that a gets estimated duration 1 and
        # this will create more balanced groups.
        durations = {"b": 1, "c": 1, "d": 1, "e": 10000}
        items = [item(x) for x in ["a", "b", "c", "d"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        expected_first, expected_second = expected
        assert first.selected == expected_first
        assert second.selected == expected_second

    @pytest.mark.parametrize(
        ("algo_name", "expected"),
        [
            (
                "duration_based_chunks",
                [[item("a"), item("b"), item("c"), item("d"), item("e")], []],
            ),
            (
                "least_duration",
                [[item("e")], [item("a"), item("b"), item("c"), item("d")]],
            ),
        ],
    )
    def test__split_tests_maintains_relative_order_of_tests(self, algo_name, expected):
        durations = {"a": 2, "b": 3, "c": 4, "d": 5, "e": 10000}
        items = [item(x) for x in ["a", "b", "c", "d", "e"]]
        algo = Algorithms[algo_name].value
        splits = algo(splits=2, items=items, durations=durations)

        first, second = splits
        expected_first, expected_second = expected
        assert first.selected == expected_first
        assert second.selected == expected_second

    def test__split_tests_same_set_regardless_of_order(self):
        """NOTE: only least_duration does this correctly"""
        tests = ["a", "b", "c", "d", "e", "f", "g"]
        durations = {t: 1 for t in tests}
        items = [item(t) for t in tests]
        algo = Algorithms["least_duration"].value
        for n in (2, 3, 4):
            selected_each: List[Set[Item]] = [set() for _ in range(n)]
            for order in itertools.permutations(items):
                splits = algo(splits=n, items=order, durations=durations)
                for i, group in enumerate(splits):
                    if not selected_each[i]:
                        selected_each[i] = set(group.selected)
                    assert selected_each[i] == set(group.selected)

    def test__algorithms_members_derived_correctly(self):
        for a in Algorithms.names():
            assert issubclass(Algorithms[a].value.__class__, AlgorithmBase)


class MyAlgorithm(AlgorithmBase):
    def __call__(self, a, b, c):
        """no-op"""


class MyOtherAlgorithm(AlgorithmBase):
    def __call__(self, a, b, c):
        """no-op"""


class TestAbstractAlgorithm:
    def test__hash__returns_correct_result(self):
        algo = MyAlgorithm()
        assert algo.__hash__() == hash(algo.__class__.__name__)

    def test__hash__returns_same_hash_for_same_class_instances(self):
        algo1 = MyAlgorithm()
        algo2 = MyAlgorithm()
        assert algo1.__hash__() == algo2.__hash__()

    def test__hash__returns_different_hash_for_different_classes(self):
        algo1 = MyAlgorithm()
        algo2 = MyOtherAlgorithm()
        assert algo1.__hash__() != algo2.__hash__()

    def test__eq__returns_true_for_same_instance(self):
        algo = MyAlgorithm()
        assert algo.__eq__(algo) is True

    def test__eq__returns_false_for_different_instance(self):
        algo1 = MyAlgorithm()
        algo2 = MyOtherAlgorithm()
        assert algo1.__eq__(algo2) is False

    def test__eq__returns_true_for_same_algorithm_different_instance(self):
        algo1 = MyAlgorithm()
        algo2 = MyAlgorithm()
        assert algo1.__eq__(algo2) is True

    def test__eq__returns_false_for_non_algorithm_object(self):
        algo = MyAlgorithm()
        other = "not an algorithm"
        assert algo.__eq__(other) is NotImplemented


class TestWeights:
    """Tests for group weights functionality."""

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__weights_none_equals_equal_weights(self, algo_name):
        """Passing weights=None should behave the same as equal weights."""
        durations = {"a": 1, "b": 1, "c": 1}
        items = [item(x) for x in durations]
        algo = Algorithms[algo_name].value

        groups_no_weights = algo(splits=3, items=items, durations=durations)
        groups_equal_weights = algo(
            splits=3, items=items, durations=durations, weights=[1, 1, 1]
        )

        for g1, g2 in zip(groups_no_weights, groups_equal_weights):
            assert g1.selected == g2.selected
            assert g1.duration == g2.duration

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__weighted_split_assigns_more_to_higher_weight(self, algo_name):
        """Groups with higher weights should get more test duration."""
        durations = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1}
        items = [item(x) for x in durations]
        algo = Algorithms[algo_name].value

        # Weight 2 for group 1, weight 1 for group 2
        # Group 1 should get ~2/3 of tests, Group 2 should get ~1/3
        groups = algo(splits=2, items=items, durations=durations, weights=[2.0, 1.0])

        # Group 1 should have more duration than group 2
        assert groups[0].duration >= groups[1].duration
        # With 6 equal tests and 2:1 weights, group 1 should get 4, group 2 should get 2
        assert len(groups[0].selected) == 4
        assert len(groups[1].selected) == 2

    def test__least_duration_weighted_split_with_unequal_durations(self):
        """Test least_duration algorithm with weights and varying durations."""
        # Total duration = 12, with weights [2, 1], group 1 should get ~8, group 2 ~4
        durations = {"a": 4, "b": 3, "c": 2, "d": 2, "e": 1}
        items = [item(x) for x in durations]
        algo = Algorithms["least_duration"].value

        groups = algo(splits=2, items=items, durations=durations, weights=[2.0, 1.0])

        # Check that group 1 has approximately twice the duration of group 2
        ratio = (
            groups[0].duration / groups[1].duration
            if groups[1].duration > 0
            else float("inf")
        )
        assert (
            1.5 <= ratio <= 2.5
        )  # Allow some flexibility due to discrete test assignment

    def test__duration_based_chunks_weighted_split(self):
        """Test duration_based_chunks algorithm respects weights."""
        durations = {"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 1}
        items = [item(x) for x in durations]
        algo = Algorithms["duration_based_chunks"].value

        # 3:1 weights - group 1 gets 75%, group 2 gets 25%
        groups = algo(splits=2, items=items, durations=durations, weights=[3.0, 1.0])

        # With 6 tests at 1 sec each, group 1 should get ~4.5 sec worth
        # This means tests a, b, c, d (4 tests), and group 2 gets e, f (2 tests)
        assert len(groups[0].selected) >= 4
        assert len(groups[1].selected) <= 2

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__weighted_split_is_deterministic(self, algo_name):
        """Weighted splits should be deterministic - same inputs, same outputs."""
        durations = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
        items = [item(x) for x in durations]
        algo = Algorithms[algo_name].value
        weights = [2.0, 1.0, 1.0]

        # Run multiple times
        results = [
            algo(splits=3, items=items, durations=durations, weights=weights)
            for _ in range(5)
        ]

        # All results should be identical
        for i in range(1, len(results)):
            for j in range(3):
                assert results[0][j].selected == results[i][j].selected
                assert results[0][j].duration == results[i][j].duration

    @pytest.mark.parametrize("algo_name", Algorithms.names())
    def test__three_groups_with_weights(self, algo_name):
        """Test three groups with different weights."""
        # 9 tests, weights 3:2:1 -> group 1 gets 50%, group 2 gets 33%, group 3 gets 17%
        durations = {chr(ord("a") + i): 1 for i in range(9)}
        items = [item(x) for x in sorted(durations.keys())]
        algo = Algorithms[algo_name].value

        groups = algo(
            splits=3, items=items, durations=durations, weights=[3.0, 2.0, 1.0]
        )

        total_selected = sum(len(g.selected) for g in groups)
        assert total_selected == 9

        # Check rough proportions
        assert len(groups[0].selected) >= len(groups[1].selected)
        assert len(groups[1].selected) >= len(groups[2].selected)
