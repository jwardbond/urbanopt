import geopandas as gpd
import gurobipy as gp
import pytest
from shapely.geometry import Point

from urbanopt import PathwayOptimizer


def test_optimizer_initialization(sample_gdf: gpd.GeoDataFrame):
    """Test that the optimizer initializes correctly with a GeoDataFrame."""
    optimizer = PathwayOptimizer(sample_gdf)

    # Should be initialized with a copy of input data
    assert optimizer.data is not sample_gdf
    assert optimizer.data.equals(sample_gdf)
    assert optimizer.crs == "EPSG:4326"
    assert optimizer.pids == [1, 2, 3]


def test_non_integer_pids():
    """Test that initialization fails when PIDs are not integers."""
    data = {
        "pid": ["1", "2", "3"],  # String PIDs
        "label": ["adu", "bment", "vsplit"],
        "start": ["A", "B", "C"],
        "end": ["X", "Y", "Z"],
        "desc": ["desc1", "desc2", "desc3"],
        "opportunity": [1.0, 2.0, 3.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    with pytest.raises(ValueError) as exc_info:
        PathwayOptimizer(gdf)

    # Should raise error for non-integer PIDs
    assert "All PIDs must be integers" in str(exc_info.value)


def test_float_pids():
    """Test that initialization fails when PIDs are floats."""
    data = {
        "pid": [1.0, 2.0, 3.0],  # Float PIDs
        "label": ["adu", "bment", "vsplit"],
        "start": ["A", "B", "C"],
        "end": ["X", "Y", "Z"],
        "desc": ["desc1", "desc2", "desc3"],
        "opportunity": [1.0, 2.0, 3.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    with pytest.raises(ValueError) as exc_info:
        PathwayOptimizer(gdf)

    # Should raise error for non-integer PIDs
    assert "All PIDs must be integers" in str(exc_info.value)


def test_missing_required_columns():
    """Test that initialization fails when required columns are missing."""
    data = {
        "pid": [1],  # Integer PID
        "start": ["A"],
        "geometry": [Point(0, 0)],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    with pytest.raises(ValueError) as exc_info:
        PathwayOptimizer(gdf)

    # Should raise error with missing columns
    assert "Missing required columns" in str(exc_info.value)
    assert "end" in str(exc_info.value)
    assert "desc" in str(exc_info.value)
    assert "opportunity" in str(exc_info.value)


def test_cost_columns_extraction(sample_gdf: gpd.GeoDataFrame):
    """Test that cost_* columns are correctly extracted."""
    sample_gdf["cost_emb"] = [10, 20, 30]
    sample_gdf["cost_transit"] = [5, 15, 25]
    sample_gdf["not_a_cost"] = [1, 2, 3]

    optimizer = PathwayOptimizer(sample_gdf)

    # Should extract correct cost columns
    assert sorted(optimizer.cost_columns) == ["cost_emb", "cost_transit"]
    assert "not_a_cost" not in optimizer.cost_columns


def test_build_variables(sample_gdf: gpd.GeoDataFrame):
    """Test that build_variables creates binary variables for each pathway."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Test that variables are created correctly
    assert len(optimizer._variables) == len(optimizer.pids)
    assert all(var.VType == gp.GRB.BINARY for var in optimizer._variables.values())
