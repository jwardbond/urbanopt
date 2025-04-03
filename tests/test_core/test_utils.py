import tempfile
from pathlib import Path

import geopandas as gpd
import pytest

from urbanopt import PathwayOptimizer


def test_print_solution_summary_after_solve(sample_gdf: gpd.GeoDataFrame, capsys):
    """Test that print_solution_summary correctly formats and prints results after solving."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.set_objective({"cost_emb": 1.0})
    optimizer.add_min_opportunity(2.0)
    optimizer.solve()

    optimizer.print_solution_summary()
    captured = capsys.readouterr()

    # Should print all required information
    assert "Optimization Summary" in captured.out
    assert "Objective Value" in captured.out
    assert "Total Opportunity" in captured.out
    assert "Solve Time" in captured.out
    assert "Selected Pathways" in captured.out


def test_print_solution_summary_before_solve(sample_gdf: gpd.GeoDataFrame):
    """Test that print_solution_summary raises error if called before solving."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    with pytest.raises(RuntimeError, match="Model has not been solved yet"):
        optimizer.print_solution_summary()


def test_debug_model_basic_info(sample_gdf: gpd.GeoDataFrame, capsys):
    """Test that debug_model prints basic model information."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.add_max_opportunity(5.0, tag="test")

    optimizer.debug_model()
    captured = capsys.readouterr()

    # Should print basic model information
    assert "Gurobi Model Debug Info" in captured.out
    assert f"Variables: {len(optimizer._variables)}" in captured.out
    assert "Constraints: 1" in captured.out  # One constraint added
    assert "Objective set: False" in captured.out
    assert f"Model status: {optimizer.model.Status}" in captured.out


def test_debug_model_verbose(sample_gdf: gpd.GeoDataFrame, capsys):
    """Test that debug_model prints detailed information in verbose mode."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.add_max_opportunity(5.0, tag="test")
    optimizer.solve()

    optimizer.debug_model(verbose=True, max_vars=2)
    captured = capsys.readouterr()

    # Should print detailed information
    assert "Variables (first 2):" in captured.out
    assert "Constraint Tags:" in captured.out
    assert "[test] 1 constraints" in captured.out

    # Should respect max_vars limit
    var_lines = [line for line in captured.out.split("\n") if "x_" in line]
    assert len(var_lines) <= 2


def test_export_model(sample_gdf: gpd.GeoDataFrame):
    """Test that export_model creates a readable .lp file."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.set_objective({"cost_emb": 1.0})
    optimizer.add_max_opportunity(5.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_model"
        optimizer.export_model(str(path))

        # Should create .lp file
        lp_file = path.with_suffix(".lp")
        assert lp_file.exists()
