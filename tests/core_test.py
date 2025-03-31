import geopandas as gpd
import gurobipy as gp
import pytest
from shapely.geometry import Point

from urbanopt import PathwayOptimizer


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame for testing."""
    data = {
        "pid": [1, 2, 3],
        "start": ["A", "B", "C"],
        "end": ["X", "Y", "Z"],
        "desc": ["desc1", "desc2", "desc3"],
        "opportunity": [1.0, 2.0, 3.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


def test_optimizer_initialization(sample_gdf: gpd.GeoDataFrame):
    """Test that the optimizer initializes correctly with a GeoDataFrame."""
    optimizer = PathwayOptimizer(sample_gdf)

    # Check that the GeoDataFrame is stored correctly
    assert optimizer.df is sample_gdf

    # Check that the CRS is stored correctly
    assert optimizer.crs == "EPSG:4326"

    # Check that pids are stored correctly
    assert optimizer.pids == [1, 2, 3]


def test_missing_required_columns():
    """Test that initialization fails when required columns are missing."""
    # Create a GeoDataFrame missing some required columns
    data = {
        "pid": [1],
        "start": ["A"],
        "geometry": [Point(0, 0)],
    }
    gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")

    # Attempt to initialize with missing columns
    with pytest.raises(ValueError) as exc_info:
        PathwayOptimizer(gdf)

    # Check that the error message lists the missing columns
    assert "Missing required columns" in str(exc_info.value)
    assert "end" in str(exc_info.value)
    assert "desc" in str(exc_info.value)
    assert "opportunity" in str(exc_info.value)


def test_cost_columns_extraction(sample_gdf: gpd.GeoDataFrame):
    """Test that cost_* columns are correctly extracted."""
    # Add cost columns to the sample data
    sample_gdf["cost_emb"] = [10, 20, 30]
    sample_gdf["cost_transit"] = [5, 15, 25]
    sample_gdf["not_a_cost"] = [1, 2, 3]  # This should not be included

    optimizer = PathwayOptimizer(sample_gdf)

    # Check that cost columns were correctly identified
    assert sorted(optimizer.cost_columns) == ["cost_emb", "cost_transit"]

    # Check that non-cost columns were not included
    assert "not_a_cost" not in optimizer.cost_columns


def test_build_variables(sample_gdf: gpd.GeoDataFrame):
    """Test that build_variables creates the model and correct number of variables."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Check that model was created
    assert isinstance(optimizer.model, gp.Model)
    assert optimizer.model.ModelName == "pathway_optimizer"

    # Check that variables dictionary was created
    assert isinstance(optimizer.variables, dict)

    # Check that we have one variable per pathway
    assert len(optimizer.variables) == len(optimizer.pids)

    # Check that each variable is binary
    for var in optimizer.variables.values():
        assert var.VType == gp.GRB.BINARY
