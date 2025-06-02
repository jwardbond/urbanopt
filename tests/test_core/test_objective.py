import geopandas as gpd
import gurobipy as gp
import pytest

from urbanopt import PathwayOptimizer


def test_set_objective_invalid_weights(sample_gdf: gpd.GeoDataFrame):
    """Test that set_objective raises ValueError for invalid cost columns."""
    sample_gdf["cost_emb"] = [10, 20, 30]
    sample_gdf["cost_transit"] = [5, 15, 25]
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    invalid_weights = {"cost_emb": 1.0, "invalid_cost": 0.5}

    with pytest.raises(ValueError) as exc_info:
        optimizer.set_objective(invalid_weights)

    # Should raise error for invalid columns
    assert "Invalid cost columns in weights" in str(exc_info.value)


def test_set_objective_valid_weights(sample_gdf: gpd.GeoDataFrame):
    """Test that set_objective correctly sets the objective with valid weights."""
    sample_gdf["cost_emb"] = [10, 20, 30]
    sample_gdf["cost_transit"] = [5, 15, 25]
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    weights = {"cost_emb": 1.0, "cost_transit": 0.5}

    optimizer.set_objective(weights)

    # Should set minimization objective
    assert optimizer.model.ModelSense == gp.GRB.MINIMIZE

    # All pids should be in the objective
    objective = optimizer.model.getObjective()
    pids_in_objective = {
        optimizer._gindex_to_pid[objective.getVar(i).index]
        for i in range(objective.size())
    }
    expected_pids = set(optimizer.pids)
    assert expected_pids == pids_in_objective

    # Objective values should have the correct weights
    for i in range(objective.size()):
        v = objective.getVar(i)
        pid = optimizer._gindex_to_pid[v.index]
        coeff = objective.getCoeff(i)

        expected_coeff = (
            sample_gdf.loc[sample_gdf["pid"] == pid, "cost_emb"].values[0]
            * weights["cost_emb"]
            + sample_gdf.loc[sample_gdf["pid"] == pid, "cost_transit"].values[0]
            * weights["cost_transit"]
        )
        assert abs(coeff - expected_coeff) < 1e-6, (
            f"Wrong coefficient for pid {pid}: got {coeff}, expected {expected_coeff}"
        )
