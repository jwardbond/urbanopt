import geopandas as gpd
import gurobipy as gp
import pytest
from shapely.geometry import Point

from urbanopt import PathwayOptimizer


def test_solve_basic_optimization(sample_gdf: gpd.GeoDataFrame):
    """Test that solve() works correctly for a basic optimization problem."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Set objective and constraints
    optimizer.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})
    optimizer.add_opportunity_constraints(
        2.0, ">="
    )  # Should force at least one pathway

    # Solve and check results
    optimizer.solve()
    selected = optimizer.get_selected_pids()

    # Should select valid pathways
    assert len(selected) > 0
    assert all(pid in optimizer.pids for pid in selected)

    #  Should select the correct optimal pathway
    assert len(selected) == 1
    assert selected[0] == 2

    # Should have optimal solution status
    assert optimizer.model.Status == gp.GRB.OPTIMAL


def test_get_selected_pids_requires_solve(sample_gdf: gpd.GeoDataFrame):
    """Test that get_selected_pids() raises error if called before solving."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.set_objective({"cost_emb": 1.0})

    # Should raise error when accessing results before solve
    with pytest.raises(RuntimeError) as exc_info:
        optimizer.get_selected_pids()

    assert "Model has not been solved yet" in str(exc_info.value)


def test_get_solution_summary_requires_solve(sample_gdf: gpd.GeoDataFrame):
    """Test that get_solution_summary() raises error if called before solving."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.set_objective({"cost_emb": 1.0})

    # Should raise error when accessing results before solve
    with pytest.raises(RuntimeError) as exc_info:
        optimizer.get_solution_summary()

    assert "Model has not been solved yet" in str(exc_info.value)


def test_solve_infeasible_model(sample_gdf: gpd.GeoDataFrame):
    """Test that solve() raises error for infeasible model."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.set_objective({"cost_emb": 1.0})

    # Add contradictory constraints
    optimizer.add_opportunity_constraints(10.0, ">=")  # Require high opportunity
    optimizer.add_opportunity_constraints(5.0, "<=")  # But limit to low opportunity

    # Should raise error for infeasible model
    with pytest.raises(RuntimeError) as exc_info:
        optimizer.solve()

    assert "Model is infeasible" in str(exc_info.value)


def test_get_solution_summary_contents(sample_gdf: gpd.GeoDataFrame):
    """Test that get_solution_summary() returns correct information."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Set objective and constraints
    optimizer.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})
    optimizer.add_opportunity_constraints(2.0, ">=")

    # Solve and get summary
    optimizer.solve()
    summary = optimizer.get_solution_summary()

    # Should contain all required fields
    assert "objective_value" in summary
    assert "cost_column_sums" in summary
    assert "total_opportunity" in summary
    assert "solve_time" in summary
    assert "selected_count" in summary

    # Should have valid values
    assert summary["selected_count"] == len(optimizer.get_selected_pids())

    assert isinstance(summary["cost_column_sums"], dict)
    assert len(summary["cost_column_sums"]) == 2
    assert "cost_emb" in summary["cost_column_sums"]
    assert "cost_transit" in summary["cost_column_sums"]
    assert pytest.approx(summary["cost_column_sums"]["cost_emb"]) == 20
    assert pytest.approx(summary["cost_column_sums"]["cost_transit"]) == 15

    assert summary["solve_time"] >= 0
    assert summary["objective_value"] >= 0
    assert summary["total_opportunity"] >= 2.0  # Due to min_opportunity constraint


def test_solve_handles_multiple_constraints(sample_gdf: gpd.GeoDataFrame):
    """Test solving with multiple interacting constraints."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Set objective and constraints
    optimizer.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})
    optimizer.add_opportunity_constraints(2.0, ">=")
    optimizer.add_opportunity_constraints(4.0, "<=")
    point = Point(0.5, 0.5)
    optimizer.add_max_opportunity_near_point(3.0, point, 1.0, proj_crs="EPSG:3347")

    # Solve and check results
    optimizer.solve()
    selected = optimizer.get_selected_pids()
    summary = optimizer.get_solution_summary()

    # Should satisfy all constraints
    assert 2.0 <= summary["total_opportunity"] <= 4.0
    assert len(selected) > 0

    # Should have optimal solution
    assert optimizer.model.Status == gp.GRB.OPTIMAL
