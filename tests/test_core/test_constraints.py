import geopandas as gpd
import gurobipy as gp
import pytest
from shapely.geometry import MultiPolygon, Polygon, Point

from urbanopt import PathwayOptimizer
from urbanopt.core import _reproject_point


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


def test_mutual_exclusion_basic_intersection(mutual_exclusion_gdf: gpd.GeoDataFrame):
    """Test that mutual exclusion constraints are created for intersecting pathways."""
    optimizer = PathwayOptimizer(mutual_exclusion_gdf)
    optimizer.build_variables()

    # Test basic intersection case between A→X and B→Y pathways
    constraints = optimizer.add_mutual_exclusion(
        (("A", "X"), ("B", "Y")),  # Single pair of intersecting pathway types
        tag="basic",
    )

    # Should create one constraint for the intersecting pair
    assert len(constraints) == 1
    assert len(optimizer.constraints["basic"]) == 1

    # Should have correct constraint structure
    constr = constraints[0]
    expr = optimizer.model.getRow(constr)
    assert expr.size() == 2
    assert constr.RHS == 1.0
    assert constr.Sense == "<"

    # Should include correct variables (pid 1 and pid 3)
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    expected_indices = {optimizer.variables[1].index, optimizer.variables[3].index}
    assert used_indices == expected_indices


def test_mutual_exclusion_no_intersections(mutual_exclusion_gdf: gpd.GeoDataFrame):
    """Test that no constraints are created when pathways don't intersect."""
    optimizer = PathwayOptimizer(mutual_exclusion_gdf)
    optimizer.build_variables()

    # Test between D→W and E→V pathways which don't intersect
    constraints = optimizer.add_mutual_exclusion(
        (("D", "W"), ("E", "V")),  # Single pair of non-intersecting pathway types
        tag="no_intersect",
    )

    # Should not create any constraints
    assert len(constraints) == 0
    assert "no_intersect" not in optimizer.constraints


def test_mutual_exclusion_multiple_pairs(mutual_exclusion_gdf: gpd.GeoDataFrame):
    """Test that mutual exclusion works with multiple pairs by calling method multiple times."""
    optimizer = PathwayOptimizer(mutual_exclusion_gdf)
    optimizer.build_variables()

    # Test multiple pairs with various intersection patterns
    pairs = [
        (("A", "X"), ("B", "Y")),  # Should create 1 constraint (pid 1 ↔ pid 3)
        (("B", "Y"), ("C", "Z")),  # Should create 0 constraints (no intersections)
        (("A", "X"), ("D", "W")),  # Should create 0 constraints (no intersections)
    ]

    all_constraints = []
    for pair in pairs:
        constraints = optimizer.add_mutual_exclusion(pair, tag="multi")
        all_constraints.extend(constraints)

    # Should create correct number of constraints (only one pair intersects)
    assert len(all_constraints) == 1
    assert len(optimizer.constraints["multi"]) == 1

    # Should have correct constraint structure
    constr = all_constraints[0]
    expr = optimizer.model.getRow(constr)
    assert expr.size() == 2
    assert constr.RHS == 1.0
    assert constr.Sense == "<"

    # Should include correct variables (pid 1 and pid 3)
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    expected_indices = {optimizer.variables[1].index, optimizer.variables[3].index}
    assert used_indices == expected_indices


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


def test_max_opportunity_near_point(sample_gdf: gpd.GeoDataFrame):
    """Test that max opportunity near point constraint correctly filters by distance."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0
    point = Point(0.5, 0.5)  # Center point
    distance = 0.7  # Should catch points within sqrt(0.5) of center
    proj_crs = "EPSG:3347"  # Canada Lambert Conformal Conic

    constraint = optimizer.add_max_opportunity_near_point(
        limit,
        point,
        distance,
        proj_crs=proj_crs,
    )
    optimizer.model.update()

    # Should create one constraint in untagged group
    assert "untagged" in optimizer.constraints
    assert len(optimizer.constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only pathways with centroids within distance
    expr = optimizer.model.getRow(constraint)
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    projected_data = sample_gdf.to_crs(proj_crs)
    projected_point = _reproject_point(point, sample_gdf.crs, proj_crs)
    mask = projected_data.geometry.centroid.distance(projected_point) <= distance
    expected_pids = projected_data[mask]["pid"].tolist()
    expected_indices = {optimizer.variables[pid].index for pid in expected_pids}

    assert used_indices == expected_indices


def test_max_opportunity_near_point_crs_validation(sample_gdf: gpd.GeoDataFrame):
    """Test that max opportunity near point constraint validates CRS."""
    # Set geographic CRS
    sample_gdf = sample_gdf.set_crs("EPSG:4326")
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    point = Point(0.5, 0.5)
    distance = 0.7

    # Should raise error if no CRS provided with geographic data
    with pytest.raises(ValueError, match="Must provide projected CRS"):
        optimizer.add_max_opportunity_near_point(4.0, point, distance)

    # Should work with projected CRS provided
    constraint = optimizer.add_max_opportunity_near_point(
        4.0,
        point,
        distance,
        proj_crs="EPSG:3347",
    )
    assert isinstance(constraint, gp.Constr)
