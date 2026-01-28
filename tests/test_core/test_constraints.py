import geopandas as gpd
import gurobipy as gp
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from shapely.geometry import MultiPolygon, Point, Polygon

from urbanopt import PathwayOptimizer
from urbanopt.core import _reproject_point


def test_add_contribution_constraints_global(sample_gdf: gpd.GeoDataFrame):
    """Test that global contribution constraint includes all pathways in a matrix constraint."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0

    constraints = optimizer.add_contribution_constraints(
        limit,
        sense="<=",
    )
    optimizer.model.update()
    constr = constraints[0]
    expr = optimizer.model.getRow(constr)

    # Should return a matrix constraint
    assert isinstance(constraints, gp.MConstr)

    # Should have exactly 1 constraint (one global limit)
    assert constraints.size == 1  # <-- check matrix size

    # Should include all pathway variables
    expected_indices = {
        optimizer._variables[f"x_{pid}"].index for pid in optimizer.pids
    }
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    assert used_indices == expected_indices


@pytest.mark.parametrize(
    "sense_input, expected_sense",
    [
        ("<=", "<"),
        (">=", ">"),
        ("==", "="),
    ],
)
def test_add_contribution_constraints_correct_sense(
    sample_gdf: gpd.GeoDataFrame, sense_input, expected_sense
):
    """Test that contribution constraints are initialized with correct senses ("<=", ">=", "==")."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    limit = 4.0
    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    constraints = optimizer.add_contribution_constraints(
        limit,
        sense=sense_input,
        boundaries=boundary,
    )

    constr = constraints[0]
    assert constr.Sense == expected_sense


@pytest.mark.parametrize(
    "boundary, expected_pids",
    [
        (
            Polygon([(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)]),
            [1],  # Small polygon: only PID 1 inside
        ),
        (
            MultiPolygon(
                [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ]
            ),
            [1, 2, 3],  # Bigger multipolygon: PIDs 1, 2, and 3
        ),
    ],
)
def test_add_contribution_constraints_geom_boundary(
    sample_gdf: gpd.GeoDataFrame,
    boundary: Polygon | MultiPolygon,
    expected_pids: list[int],
):
    """Test that contribution constraint correctly filters different sets of PIDs for Polygon vs MultiPolygon."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    limit = 10.0

    constraints = optimizer.add_contribution_constraints(
        limit, sense="<=", boundaries=boundary
    )
    optimizer.model.update()

    # Should return a matrix constraint
    assert isinstance(constraints, gp.MConstr)

    # Should have exactly 1 constraint
    assert constraints.size == 1

    # Should include only the expected variables
    constr = constraints[0]
    expr = optimizer.model.getRow(constr)
    expected_indices = {optimizer._variables[f"x_{pid}"].index for pid in expected_pids}
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    assert used_indices == expected_indices


def test_add_contribution_constraints_handles_tags(sample_gdf: gpd.GeoDataFrame):
    """Test that contribution constraints are correctly tagged."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    limit1 = 4.0
    limit2 = 5.0
    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    c1 = optimizer.add_contribution_constraints(
        limit1, sense="<=", boundaries=boundary, tag="test_tag"
    )
    c2 = optimizer.add_contribution_constraints(
        limit2, sense="<=", boundaries=boundary, tag="test_tag"
    )

    # Should store both matrix constraints under the same tag
    assert "test_tag" in optimizer._constraints

    # Should return MConstr, but should be stored internally as list (like Gurobi)
    assert isinstance(c1, gp.MConstr)
    assert isinstance(c2, gp.MConstr)

    stored_constraints = optimizer._constraints["test_tag"]
    assert len(stored_constraints) == c1.size + c2.size

    for constr in c1.tolist():
        assert constr in stored_constraints
    for constr in c2.tolist():
        assert constr in stored_constraints


def test_add_contribution_constraints_multiple_limits_and_boundaries(
    sample_gdf: gpd.GeoDataFrame,
):
    """Test that multiple limits and boundaries create a matrix constraint with correct rows and columns."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    limits = [4.0, 5.0]
    boundaries = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # Should intersect PID 1, 2
        Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),  # Should intersect PID 3
    ]

    constraints = optimizer.add_contribution_constraints(
        limits,
        sense="<=",
        boundaries=boundaries,
        tag="multi_limits",
    )
    optimizer.model.update()

    # Should return a matrix constraint
    assert isinstance(constraints, gp.MConstr)

    # Should have exactly 2 constraints (one per limit/boundary)
    assert constraints.size == 2

    # First constraint should involve PID 1 and PID 2
    constr0 = constraints[0]
    expr0 = optimizer.model.getRow(constr0)
    used_indices_0 = {expr0.getVar(j).index for j in range(expr0.size())}
    expected_indices_0 = {
        optimizer._variables["x_1"].index,
        optimizer._variables["x_2"].index,
    }
    assert used_indices_0 == expected_indices_0
    assert constr0.RHS == 4.0

    # Second constraint should involve PID 3
    constr1 = constraints[1]
    expr1 = optimizer.model.getRow(constr1)
    used_indices_1 = {expr1.getVar(j).index for j in range(expr1.size())}
    expected_indices_1 = {optimizer._variables["x_3"].index}
    assert used_indices_1 == expected_indices_1
    assert constr1.RHS == 5.0


# FIXME these zone constraints are bad
def test_zone_difference_constraints_creates_two_constraints(
    sample_gdf: gpd.GeoDataFrame,
):
    """Ensure two MConstrs are returned per geometry pair."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    geom_pairs = [
        (
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        )
    ]
    limits = [10.0]

    constrs1, constrs2 = optimizer.add_zone_difference_constraints(
        geom_pairs,
        sense="<=",
        limits=limits,
        tag="zone_diff",
    )

    assert isinstance(constrs1, gp.MConstr)
    assert isinstance(constrs2, gp.MConstr)
    assert constrs1.size == 1
    assert constrs2.size == 1

    assert "zone_diff_1" in optimizer._constraints
    assert "zone_diff_2" in optimizer._constraints


def test_zone_difference_constraints_tags_are_correct(sample_gdf: gpd.GeoDataFrame):
    """Check that both MConstrs are stored under separate tags."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    pair = (
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
    )

    optimizer.add_zone_difference_constraints(
        [pair], sense="<=", limits=[5.0], tag="zd_test"
    )

    assert "zd_test_1" in optimizer._constraints
    assert "zd_test_2" in optimizer._constraints
    assert isinstance(optimizer._constraints["zd_test_1"], list)
    assert isinstance(optimizer._constraints["zd_test_2"], list)
    assert len(optimizer._constraints["zd_test_1"]) == 1
    assert len(optimizer._constraints["zd_test_2"]) == 1


def test_zone_difference_constraints_handles_non_intersecting(
    sample_gdf: gpd.GeoDataFrame,
):
    """Ensure constraint still works when one of the zones has no intersecting geometry."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    pair = (
        Polygon([(100, 100), (101, 100), (101, 101), (100, 101)]),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
    )  # Only second intersects

    constrs1, constrs2 = optimizer.add_zone_difference_constraints(
        [pair], sense="<=", limits=[999.0], tag="sparse"
    )

    assert isinstance(constrs1, gp.MConstr)
    assert isinstance(constrs2, gp.MConstr)
    assert constrs1.size == 1
    assert constrs2.size == 1


class TestConversionConstraints:
    def test_no_overlap(
        self,
        overlapping_pathways_gdf: gpd.GeoDataFrame,
    ):
        """Test a simple instance where we don't care about overlaps"""
        input_gdf = overlapping_pathways_gdf
        optimizer = PathwayOptimizer(input_gdf)
        optimizer.build_variables()
        optimizer.add_conversion_constraints(start_name="A", sense="<=", limit=1)

        # Constraints should be added as untagged
        assert "untagged" in optimizer._constraints
        assert len(optimizer._constraints["untagged"]) == 1

        # All pathways with start==A should be in the constraint
        constr = optimizer._constraints["untagged"][0]
        expr = optimizer.model.getRow(constr)
        used_vars = {expr.getVar(i).VarName for i in range(expr.size())}
        expected_vars = {
            f"x_{pid}" for pid in input_gdf[input_gdf["start"] == "A"]["pid"]
        }
        assert used_vars == expected_vars

        # The constraint should have the correct sense and rhs
        assert constr.Sense == "<"
        assert constr.RHS == 1.0

    def test_with_overlaps(
        self,
        overlapping_pathways_gdf: gpd.GeoDataFrame,
    ):
        gdf = overlapping_pathways_gdf
        optimizer = PathwayOptimizer(gdf)
        optimizer.build_variables()
        optimizer.add_conversion_constraints(
            start_name="A",
            sense="<=",
            limit=1,
            check_overlaps=True,
            proj_crs="EPSG:3347",
        )

        # x_3 and z_12 and z_45 should be in the constraint
        constr = optimizer._constraints["untagged"][0]
        expr = optimizer.model.getRow(constr)
        used_vars = {expr.getVar(i).VarName for i in range(expr.size())}
        expected_vars = {"x_3", "z_12", "z_456"}
        assert used_vars == expected_vars

        # Dummy variables should be added
        assert "z_12" in optimizer._variables
        assert "z_456" in optimizer._variables

        # Dummy constraints should be added
        dummy_constraints = optimizer._constraints.get("dummy", [])
        assert len(dummy_constraints) == 2

    def test_invalid_start(self, mutual_exclusion_gdf: gpd.geodataframe):
        gdf = mutual_exclusion_gdf
        optimizer = PathwayOptimizer(gdf)
        optimizer.build_variables()
        with pytest.raises(ValueError, match="No pathways start from Z"):
            optimizer.add_conversion_constraints(start_name="Z", sense="<=", limit=1)

    def test_debuff_warning(self, mutual_exclusion_gdf: gpd.GeoDataFrame):
        gdf = mutual_exclusion_gdf
        optimizer = PathwayOptimizer(gdf)
        optimizer.build_variables()
        with pytest.warns(
            UserWarning,
            match="Specifying debuff without setting check_overlaps=True",
        ):
            optimizer.add_conversion_constraints(
                start_name="A",
                sense="<=",
                limit=1,
                debuff=0.1,
            )

    def test_tags(self, mutual_exclusion_gdf: gpd.GeoDataFrame):
        gdf = mutual_exclusion_gdf
        optimizer = PathwayOptimizer(gdf)
        optimizer.build_variables()
        optimizer.add_conversion_constraints(
            start_name="A",
            sense="<=",
            limit=1,
            tag="test_tag",
        )
        optimizer.model.update()

        assert "test_tag" in optimizer._constraints
        assert optimizer._constraints["test_tag"]


def test_add_mutual_exclusion_creates_constraints(
    mutual_exclusion_gdf: gpd.GeoDataFrame,
):
    """Test that mutual exclusion constraints are created for intersecting pathways."""
    optimizer = PathwayOptimizer(mutual_exclusion_gdf)
    optimizer.build_variables()

    # Test basic intersection case between adu and bment pathways
    mconstr = optimizer.add_mutual_exclusion(
        label1="adu",
        label2="bment",
        tag="basic",
    )

    # Should create one constraint for the intersecting pair
    assert isinstance(mconstr, gp.MConstr)
    assert mconstr.shape == (1,)  # One constraint

    # Should have correct constraint structure
    constr = mconstr[0]  # Get first constraint from matrix
    expr = optimizer.model.getRow(constr)
    assert expr.size() == 2
    assert constr.RHS == 1.0
    assert constr.Sense == "<"

    # Should include correct variables (pid 1 and pid 3)
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    expected_indices = {
        optimizer._variables["x_1"].index,
        optimizer._variables["x_3"].index,
    }
    assert used_indices == expected_indices


def test_add_mutual_exclusion_handles_no_intersections(
    mutual_exclusion_gdf: gpd.GeoDataFrame,
):
    """Test that no constraints are created when pathways don't intersect."""
    optimizer = PathwayOptimizer(mutual_exclusion_gdf)
    optimizer.build_variables()

    # Test between hsplit and merge pathways which don't intersect
    mconstr = optimizer.add_mutual_exclusion(
        label1="hsplit",
        label2="merge",
        tag="no_intersect",
    )

    # Should not create any constraints
    assert mconstr is None
    assert "no_intersect" not in optimizer._constraints


def test_add_mutual_exclusion_handles_single_label(
    mutual_exclusion_gdf: gpd.GeoDataFrame,
):
    """Test that mutual exclusion works with a single label."""
    optimizer = PathwayOptimizer(mutual_exclusion_gdf)
    optimizer.build_variables()

    # Test exclusion for adu pathways
    mconstr = optimizer.add_mutual_exclusion(
        label1="adu",
        tag="exclusion",
    )

    # Should create one constraint for the intersecting bment pathway
    assert isinstance(mconstr, gp.MConstr)
    assert mconstr.shape == (1,)  # One constraint

    # Should have correct constraint structure
    constr = mconstr[0]  # Get first constraint from matrix
    expr = optimizer.model.getRow(constr)
    assert expr.size() == 2
    assert constr.RHS == 1.0
    assert constr.Sense == "<"

    # Should include correct variables (pid 1 and pid 3)
    used_indices = {expr.getVar(i).index for i in range(expr.size())}
    expected_indices = {
        optimizer._variables["x_1"].index,
        optimizer._variables["x_3"].index,
    }
    assert used_indices == expected_indices


def test_add_max_contribution_near_point_filters_by_distance(
    sample_gdf: gpd.GeoDataFrame,
):
    """Test that max contribution near point constraint correctly filters by distance."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    limit = 4.0
    point = Point(0.5, 0.5)  # Center point
    distance = 0.7  # Should catch points within sqrt(0.5) of center
    proj_crs = "EPSG:3347"  # Canada Lambert Conformal Conic

    constraint = optimizer.add_max_contribution_near_point(
        limit,
        point,
        distance,
        proj_crs=proj_crs,
    )
    optimizer.model.update()

    # Should create one constraint in untagged group
    assert "untagged" in optimizer._constraints
    assert len(optimizer._constraints["untagged"]) == 1
    assert isinstance(constraint, gp.Constr)
    assert limit == constraint.RHS

    # Should include only pathways with centroids within distance
    expr = optimizer.model.getRow(constraint)
    used_indices = {expr.getVar(i).index for i in range(expr.size())}

    projected_data = sample_gdf.to_crs(proj_crs)
    projected_point = _reproject_point(point, sample_gdf.crs, proj_crs)
    mask = projected_data.geometry.centroid.distance(projected_point) <= distance
    expected_pids = projected_data[mask]["pid"].tolist()
    expected_indices = {optimizer._variables[f"x_{pid}"].index for pid in expected_pids}

    assert used_indices == expected_indices


def test_add_max_contribution_near_point_validates_crs(sample_gdf: gpd.GeoDataFrame):
    """Test that max contribution near point constraint validates CRS."""
    # Set geographic CRS
    sample_gdf = sample_gdf.set_crs("EPSG:4326")
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    point = Point(0.5, 0.5)
    distance = 0.7

    # Should raise error if no CRS provided with geographic data
    with pytest.raises(ValueError, match="Must provide projected CRS"):
        optimizer.add_max_contribution_near_point(4.0, point, distance)

    # Should work with projected CRS provided
    constraint = optimizer.add_max_contribution_near_point(
        4.0,
        point,
        distance,
        proj_crs="EPSG:3347",
    )
    assert isinstance(constraint, gp.Constr)


def test_register_constraint_senses(sample_gdf: gpd.GeoDataFrame):
    """Test that _register_constraint handles all constraint types correctly."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Should create less-than constraint
    varnames = [f"x_{p}" for p in optimizer.pids]
    const_le = optimizer._register_constraint(
        varnames=varnames,
        coeff_map=dict.fromkeys(varnames, 1.0),
        sense="<=",
        rhs=5.0,
        tag="test_le",
    )
    assert const_le.Sense == "<"

    # Should create greater-than constraint
    const_ge = optimizer._register_constraint(
        varnames=varnames,
        coeff_map=dict.fromkeys(varnames, 1.0),
        sense=">=",
        rhs=2.0,
        tag="test_ge",
    )
    assert const_ge.Sense == ">"

    # Should create equality constraint
    const_eq = optimizer._register_constraint(
        varnames=varnames,
        coeff_map=dict.fromkeys(varnames, 1.0),
        sense="==",
        rhs=3.0,
        tag="test_eq",
    )
    assert const_eq.Sense == "="

    # Should reject invalid constraint sense
    with pytest.raises(ValueError, match="Invalid constraint sense"):
        optimizer._register_constraint(
            varnames=varnames,
            coeff_map=dict.fromkeys(varnames, 1.0),
            sense="invalid",
            rhs=1.0,
        )


def test_remove_constraints_removes_by_tag(sample_gdf: gpd.GeoDataFrame):
    """Test that removing constraints by tag actually deletes them from the model and tracking dict."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    boundary = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Add constraints with different tags
    c1 = optimizer.add_contribution_constraints(
        5.0, "<=", boundaries=boundary, tag="test_tag"
    )
    c2 = optimizer.add_contribution_constraints(
        2.0, ">=", boundaries=boundary, tag="test_tag"
    )
    c3 = optimizer.add_contribution_constraints(
        3.0, "<=", boundaries=boundary, tag="other_tag"
    )

    c1_constrs = c1.tolist()
    c2_constrs = c2.tolist()
    c3_constrs = c3.tolist()

    optimizer.remove_constraints("test_tag")

    # Get remaining model constraints
    remaining_constraints = optimizer.model.getConstrs()

    # c1 and c2 constraints should be gone
    for constr in c1_constrs + c2_constrs:
        assert constr not in remaining_constraints

    # c3 constraints should still be present
    for constr in c3_constrs:
        assert constr in remaining_constraints

    # _constraints dictionary should be updated properly
    assert "test_tag" not in optimizer._constraints
    assert "other_tag" in optimizer._constraints


def test_remove_constraints_invalid_tag(sample_gdf: gpd.GeoDataFrame):
    """Test that removing constraints with invalid tag raises error."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.add_contribution_constraints(5.0, "<=", tag="test_tag")

    # Should raise error for non-existent tag
    with pytest.raises(ValueError) as exc_info:
        optimizer.remove_constraints("invalid_tag")

    assert "Tag 'invalid_tag' not found in constraints" in str(exc_info.value)


def test_register_matrix_constraint(sample_gdf: gpd.GeoDataFrame):
    """Test that matrix constraints are correctly registered with less-than sense."""
    # TODO


def test_register_matrix_constraint_less_than(sample_gdf: gpd.GeoDataFrame):
    """Test that matrix constraints are correctly registered with less-than sense."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    pids = [1, 2]
    varnames = [f"x_{pid}" for pid in pids]
    coeffs = csr_matrix([[1.0, 1.0], [2.0, 3.0]])
    rhs = np.array([1.0, 5.0])

    # x1 + x2 <= 1.0
    # 2x1 + 3x2 <= 5.0

    mconstr = optimizer._register_matrix_constraint(
        varnames=varnames,
        coeffs=coeffs,
        sense="<=",
        rhs=rhs,
        tag="test_matrix",
    )
    stored_constraints = optimizer._constraints["test_matrix"]

    # Should store flattened constraints under tag
    assert "test_matrix" in optimizer._constraints
    assert len(stored_constraints) == mconstr.size

    # Should be a matrix constraint object with correct size
    assert isinstance(mconstr, gp.MConstr)
    assert mconstr.shape == (2,)

    # Should have correct constraint properties
    for i in range(mconstr.shape[0]):
        constr = mconstr[i]
        expr = optimizer.model.getRow(constr)
        actual_coeffs = {
            expr.getVar(j).index: expr.getCoeff(j) for j in range(expr.size())
        }

        assert constr.Sense == "<"
        assert abs(constr.RHS - rhs[i]) < 1e-6

        # Each variable should have the correct coefficient
        for j, pid in enumerate(pids):
            var_index = optimizer._pid_to_gindex[pid]
            assert var_index in actual_coeffs
            assert abs(actual_coeffs[var_index] - coeffs[i, j]) < 1e-6

        assert mconstr.tolist()[i] in stored_constraints


def test_register_matrix_constraint_invalid_sense(sample_gdf: gpd.GeoDataFrame):
    """Test that matrix constraint registration rejects invalid sense."""
    import numpy as np
    from scipy.sparse import csr_matrix

    # Setup
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    pids = [1, 2]
    varnames = [f"x_{p}" for p in pids]
    coeffs = csr_matrix([[1.0, 1.0], [2.0, 3.0]])
    rhs = np.array([1.0, 5.0])

    # Should reject invalid sense
    with pytest.raises(ValueError, match="Invalid constraint sense"):
        optimizer._register_matrix_constraint(
            varnames=varnames,
            coeffs=coeffs,
            sense="invalid",
            rhs=rhs,
            tag="test_matrix",
        )
