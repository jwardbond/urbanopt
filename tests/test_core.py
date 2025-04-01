import geopandas as gpd
import gurobipy as gp
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

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

    # Should be initialized with a copy of input data
    assert optimizer.data is not sample_gdf
    assert optimizer.data.equals(sample_gdf)
    assert optimizer.crs == "EPSG:4326"
    assert optimizer.pids == [1, 2, 3]


def test_missing_required_columns():
    """Test that initialization fails when required columns are missing."""
    data = {
        "pid": [1],
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
    """Test that build_variables creates the model and correct number of variables."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Should initialize model correctly
    assert isinstance(optimizer.model, gp.Model)
    assert optimizer.model.ModelName == "pathway_optimizer"

    # Should create correct variables
    assert isinstance(optimizer.variables, dict)
    assert len(optimizer.variables) == len(optimizer.pids)
    assert all(var.VType == gp.GRB.BINARY for var in optimizer.variables.values())


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


def test_add_max_opportunity_global(sample_gdf: gpd.GeoDataFrame):
    """Test that add_max_opportunity creates a global constraint including all pids."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0

    optimizer.add_max_opportunity(limit)
    optimizer.model.update()

    # Should add one gurobi constraint
    constraint = optimizer.constraints[0]
    assert len(optimizer.constraints) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Constraint should have all variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in optimizer.pids}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    assert used_indices == expected_indices


def test_add_max_opportunity_polygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that add_max_opportunity creates a constraint with polygon boundary including only intersecting pids."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0
    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    optimizer.add_max_opportunity(limit, boundary)
    optimizer.model.update()

    # Should add one gurobi constraint
    constraint = optimizer.constraints[0]
    assert len(optimizer.constraints) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    assert used_indices == expected_indices


def test_add_max_opportunity_multipolygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that add_max_opportunity creates a constraint with multipolygon boundary including only intersecting pids."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0
    boundary = MultiPolygon(
        [
            Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            Polygon([(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)]),
        ],
    )

    optimizer.add_max_opportunity(limit, boundary)
    optimizer.model.update()

    # Should add one gurobi constraint
    constraint = optimizer.constraints[0]
    assert len(optimizer.constraints) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    assert used_indices == expected_indices


def test_add_min_opportunity_global(sample_gdf: gpd.GeoDataFrame):
    """Test that add_min_opportunity creates a global constraint including all pids."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 2.0

    optimizer.add_min_opportunity(limit)
    optimizer.model.update()

    # Should add one gurobi constraint
    constraint = optimizer.constraints[0]
    assert len(optimizer.constraints) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include all variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in optimizer.pids}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    assert used_indices == expected_indices


def test_add_min_opportunity_polygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that add_min_opportunity creates a constraint with polygon boundary including only intersecting pids."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 2.0
    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    optimizer.add_min_opportunity(limit, boundary)
    optimizer.model.update()

    # Should add one gurobi constraint
    constraint = optimizer.constraints[0]
    assert len(optimizer.constraints) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    assert used_indices == expected_indices


def test_add_min_opportunity_multipolygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that add_min_opportunity creates a constraint with multipolygon boundary including only intersecting pids."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 2.0
    boundary = MultiPolygon(
        [
            Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            Polygon([(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)]),
        ],
    )

    optimizer.add_min_opportunity(limit, boundary)
    optimizer.model.update()

    # Should add one gurobi constraint
    constraint = optimizer.constraints[0]
    assert len(optimizer.constraints) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    assert used_indices == expected_indices
