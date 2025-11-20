"""Tests for utility helpers used in table formatting."""

from polars_eval_metrics.utils import parse_pivot_column


class TestParsePivotColumn:
    """Verify structured column parsing for pivot output."""

    def test_parses_brace_wrapped_values(self) -> None:
        """Legacy brace-wrapped strings should round-trip in order."""

        column = '{"mean","auc"}'
        assert parse_pivot_column(column) == ("mean", "auc")

    def test_parses_tuple_representation(self) -> None:
        """Python tuple repr from Polars should be supported."""

        column = "('median', 'rmse')"
        assert parse_pivot_column(column) == ("median", "rmse")

    def test_parses_dict_representation(self) -> None:
        """Dictionary-style labels retain estimate before metric."""

        column = "{'estimate': 'p50', 'metric': 'rmse'}"
        assert parse_pivot_column(column) == ("p50", "rmse")

    def test_returns_none_for_unstructured_columns(self) -> None:
        """Non-structured column names should return None."""

        assert parse_pivot_column("accuracy") is None
