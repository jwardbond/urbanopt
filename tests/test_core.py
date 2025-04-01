import geopandas as gpd
import gurobipy as gp
import pytest
from shapely.geometry import MultiPolygon, Point, Polygon

from urbanopt import PathwayOptimizer


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame for testing."""
    data = {
        "pid": [1, 2, 3],  # Integer PIDs
        "start": ["A", "B", "C"],
        "end": ["X", "Y", "Z"],
        "desc": ["desc1", "desc2", "desc3"],
        "opportunity": [1.0, 2.0, 3.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


# ************************
# Test initialization
# ************************
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


# ************************
# Test objective
# ************************
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


# ************************
# Test constraints
# ************************
def test_add_max_opportunity_global(sample_gdf: gpd.GeoDataFrame):
    """Test that global max opportunity constraint includes all pathways."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0

    constraint = optimizer.add_max_opportunity(limit)
    optimizer.model.update()

    # Should create one constraint in untagged group
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include all pathway variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in optimizer.pids}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    assert used_indices == expected_indices


def test_add_max_opportunity_polygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that max opportunity constraint correctly filters by polygon boundary."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0
    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    constraint = optimizer.add_max_opportunity(limit, boundary)
    optimizer.model.update()

    # Should create one constraint in untagged group
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    assert used_indices == expected_indices


def test_add_max_opportunity_multipolygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that max opportunity constraint correctly filters by multipolygon boundary."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0
    boundary = MultiPolygon(
        [
            Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            Polygon([(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)]),
        ],
    )

    constraint = optimizer.add_max_opportunity(limit, boundary)
    optimizer.model.update()

    # Should create one constraint in untagged group
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    assert used_indices == expected_indices


def test_add_max_opportunity_with_tag(sample_gdf: gpd.GeoDataFrame):
    """Test that max opportunity constraints are correctly tagged."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    c1 = optimizer.add_max_opportunity(4.0, tag="test_tag")
    c2 = optimizer.add_max_opportunity(5.0, tag="test_tag")

    # Should store both constraints under the same tag
    assert "test_tag" in optimizer.constraints
    assert len(optimizer.constraints["test_tag"]) == 2
    assert optimizer.constraints["test_tag"] == [c1, c2]


def test_add_min_opportunity_global(sample_gdf: gpd.GeoDataFrame):
    """Test that add_min_opportunity creates a global constraint including all pids."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 2.0

    constraint = optimizer.add_min_opportunity(limit)
    optimizer.model.update()

    # Should add one constraint to untagged group
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include all variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in optimizer.pids}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    assert used_indices == expected_indices


def test_add_min_opportunity_polygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that min opportunity constraint correctly filters by polygon boundary."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 2.0
    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    constraint = optimizer.add_min_opportunity(limit, boundary)
    optimizer.model.update()

    # Should create one constraint in untagged group
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    assert used_indices == expected_indices


def test_add_min_opportunity_multipolygon_boundary(sample_gdf: gpd.GeoDataFrame):
    """Test that min opportunity constraint correctly filters by multipolygon boundary."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 2.0
    boundary = MultiPolygon(
        [
            Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            Polygon([(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)]),
        ],
    )

    constraint = optimizer.add_min_opportunity(limit, boundary)
    optimizer.model.update()

    # Should create one constraint in untagged group
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only intersecting variables
    expr = optimizer.model.getRow(constraint)
    expected_indices = {optimizer.variables[pid].index for pid in [1, 2]}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    assert used_indices == expected_indices


def test_add_min_opportunity_with_tag(sample_gdf: gpd.GeoDataFrame):
    """Test that min opportunity constraints handle tags correctly."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    c1 = optimizer.add_min_opportunity(2.0, tag="min_tag")
    c2 = optimizer.add_min_opportunity(3.0)  # Untagged

    # Should store constraints in appropriate groups
    assert "min_tag" in optimizer.constraints
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["min_tag"]) == 1
    assert len(optimizer.constraints["untagged"]) == 1
    assert optimizer.constraints["min_tag"][0] == c1
    assert optimizer.constraints["untagged"][0] == c2


def test_add_constraint_helper(sample_gdf: gpd.GeoDataFrame):
    """Test that _add_constraint handles all constraint types correctly."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Should create less-than constraint
    const_le = optimizer._add_constraint(
        pids=optimizer.pids,
        coeff_func=lambda pid: 1.0,  # noqa: ARG005
        sense="le",
        rhs=5.0,
        tag="test_le",
    )
    assert const_le.Sense == "<"

    # Should create greater-than constraint
    const_ge = optimizer._add_constraint(
        pids=optimizer.pids,
        coeff_func=lambda pid: 1.0,  # noqa: ARG005
        sense="ge",
        rhs=2.0,
        tag="test_ge",
    )
    assert const_ge.Sense == ">"

    # Should create equality constraint
    const_eq = optimizer._add_constraint(
        pids=optimizer.pids,
        coeff_func=lambda pid: 1.0,  # noqa: ARG005
        sense="eq",
        rhs=3.0,
        tag="test_eq",
    )
    assert const_eq.Sense == "="

    # Should reject invalid constraint sense
    with pytest.raises(ValueError, match="Invalid constraint sense"):
        optimizer._add_constraint(
            pids=optimizer.pids,
            coeff_func=lambda pid: 1.0,  # noqa: ARG005
            sense="invalid",
            rhs=1.0,
        )


def test_add_constraint_coefficient_function(sample_gdf: gpd.GeoDataFrame):
    """Test that constraint coefficients are correctly computed."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    def custom_coeff(pid: int):
        return (
            optimizer.data.loc[optimizer.data["pid"] == pid, "opportunity"].iloc[0]
            * 2.0
        )

    constr = optimizer._add_constraint(
        pids=optimizer.pids,
        coeff_func=custom_coeff,
        sense="le",
        rhs=10.0,
        tag="test_coeff",
    )

    # Should compute coefficients correctly
    expr = optimizer.model.getRow(constr)
    coeffs = [expr.getCoeff(i) for i in range(expr.size())]
    expected_coeffs = [2.0, 4.0, 6.0]  # 2 * opportunity values
    assert coeffs == expected_coeffs


def test_mixed_constraint_tags(sample_gdf: gpd.GeoDataFrame):
    """Test that different constraint types can be mixed under tags."""
    # Setup
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Action
    c1 = optimizer.add_max_opportunity(5.0, tag="mixed")
    c2 = optimizer.add_min_opportunity(2.0, tag="mixed")
    c3 = optimizer.add_max_opportunity(3.0)  # Untagged
    c4 = optimizer.add_min_opportunity(1.0, tag="other")

    # Should organize constraints by tag correctly
    assert set(optimizer.constraints.keys()) == {"mixed", "other", "untagged"}
    assert optimizer.constraints["mixed"] == [c1, c2]
    assert optimizer.constraints["other"] == [c4]
    assert optimizer.constraints["untagged"] == [c3]
