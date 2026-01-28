import json
import warnings
from pathlib import Path
from typing import Literal

import geopandas as gpd
import gurobipy as gp
import numpy as np
import pandas as pd
from pydantic import BaseModel
from pyproj import Transformer
from scipy.sparse import coo_matrix, csr_array
from shapely.geometry import MultiPolygon, Point, Polygon

from .polygraph import PolyGraph


class ConstraintSchema(BaseModel):
    """Schema for serializing constraints."""

    varnames: list[str]
    coeffs: list[float]
    sense: Literal["<=", ">=", "=="]
    rhs: float


class OptimizerConfigSchema(BaseModel):
    """Schema for serializing a PathwayOptimizer Configuration."""

    varnames: list[str]
    cost_columns: list[str]
    objective: dict[str, float] | None
    constraints: dict[str, list[ConstraintSchema]]


class PathwayOptimizer:
    """Optimizer for housing pathway selection based on multiple criteria.

    This class provides functionality to optimize pathway selection using Gurobi,
    allowing for various constraints and objectives to be defined. It supports
    spatial filtering through boundary polygons and flexible constraint tagging
    for organization.

    Attributes:
        data (gpd.GeoDataFrame): Copy of input GeoDataFrame with pathway data.
        crs: Coordinate reference system from input GeoDataFrame.
        pids (list[int]): List of pathway IDs.
        cost_columns (list[str]): List of cost-related column names.
        model (gp.Model): Gurobi optimization model (initialized by build_variables).

    <!--
    Internal Attributes:
        _variables (dict[str, gp.Var]): Dictionary mapping variable names to Gurobi variables.
            (initialized by build_variables).
        _constraints (dict[str, list[gp.Constr]]): Dictionary mapping tags to lists of constraints.
        _gindex_to_pid (dict[int, int]): Dictionary mapping gurobi variable indices to pathway IDs.
        _pid_to_gindex (dict[int, int]): Dictionary mapping pathway ID to gurobi variable indices.
        _objective_weights (dict[str, float]): Dictionary mapping cost column names to their weights
            in the objective function.
        _has_dummies (bool): True if dummy "z" variables have been added to the problem (e.g. via add_conversion_constraints)
    -->

    Example:
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> data = {
        ...     "pid": [1, 2, 3],
        ...     "start": ["A", "B", "C"],
        ...     "end": ["X", "Y", "Z"],
        ...     "desc": ["desc1", "desc2", "desc3"],
        ...     "contribution": [1.0, 2.0, 3.0],
        ...     "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        ...     "cost_emb": [10, 20, 30],
        ...     "cost_transit": [5, 15, 25]
        ... }
        >>> gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        >>> optimizer = PathwayOptimizer(gdf)
        >>> optimizer.build_variables()
        >>> optimizer.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})
        >>> optimizer.add_contribution_constraints(3.0, tag="constraints")
    """

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        """Initialize the PathwayOptimizer.

        Args:
            gdf: GeoDataFrame containing pathway data with required columns:
                - pid: Pathway identifier (integer)
                - label: Pathway type label (e.g., "adu")
                - start: Start point/area
                - end: End point/area
                - desc: Description
                - contribution: Contribution score
                - geometry: Shapely geometry
                - cost_*: Cost-related columns

        Raises:
            ValueError: If any required columns are missing from the GeoDataFrame,
                       or if any pid values are not integers.
        """
        # Validate required columns
        required_columns = [
            "pid",
            "label",
            "start",
            "end",
            "desc",
            "contribution",
            "geometry",
        ]
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
            raise ValueError(msg)

        # Validate that all PIDs are integers
        if gdf["pid"].dtype.kind != "i":
            msg = "All PIDs must be integers"
            raise ValueError(msg)

        # Store a copy of the input data
        self.data = gdf.copy()

        # Store CRS
        self.crs = str(gdf.crs)

        # Store list of pathway IDs
        self.pids = gdf["pid"].tolist()

        # Extract and store cost columns
        self.cost_columns = [col for col in gdf.columns if col.startswith("cost_")]

        # Initialize the model
        self.model = gp.Model("pathway_optimizer")

        # Initialize constraint tracking
        self._constraints = {}

        # Initialize objective function
        self._objective_weights = {}

        self._has_dummies = False

    @classmethod
    def load(cls, path: str | Path) -> "PathwayOptimizer":
        """Load an optimizer from saved files.

        Args:
            path: Base path for loading files (without extension).
                 Will look for {path}.geoparquet and {path}.json.

        Returns:
            A new PathwayOptimizer instance with restored state.

        Raises:
            FileNotFoundError: If either file is missing.
            ValueError: If saved state is invalid.
        """
        path = Path(path)

        # Load GeoDataFrame first
        parquet_path = path.with_suffix(".geoparquet")
        if not parquet_path.exists():
            msg = f"GeoParquet file not found: {parquet_path}"
            raise FileNotFoundError(msg)
        data = gpd.read_parquet(parquet_path)

        # Load and validate config
        json_path = path.with_suffix(".json")
        if not json_path.exists():
            msg = f"JSON configuration file not found: {json_path}"
            raise FileNotFoundError(msg)
        config_dict = json.loads(json_path.read_text())
        config = OptimizerConfigSchema.model_validate(config_dict)

        # Build model
        optimizer = cls(data)
        optimizer.build_variables()

        # Restore objective if present
        if config.objective:
            optimizer.set_objective(config.objective)

        # Deserialize and restore constraints
        for tag, constr_list in config.constraints.items():
            for constr in constr_list:
                optimizer._deserialize_and_register_constraint(constr=constr, tag=tag)

        return optimizer

    def build_variables(self) -> None:
        """Create binary variables for each pathway and initialize the Gurobi model.

        Creates one binary variable per pathway ID and stores them in self._variables.
        Also creates and stores the Gurobi model in self.model.
        """
        # Initialize variable tracking
        self._variables: dict[str, gp.Var] = {}

        # Initialize index mapping dictionaries
        self._gindex_to_pid = {}
        self._pid_to_gindex = {}

        # Create binary variables for each pathway
        for pid in self.pids:
            var = self.model.addVar(vtype=gp.GRB.BINARY, name=f"x_{pid}")
            self._variables[f"x_{pid}"] = var

            self._gindex_to_pid[var.index] = pid
            self._pid_to_gindex[pid] = var.index

        # Update the model to include the new variables
        self.model.update()

    def set_objective(self, weights: dict[str, float]) -> None:
        """Set the optimization objective using weighted costs.

        Args:
            weights: Dictionary mapping cost column names to their weights.
                    Keys must exist in self.cost_columns.

        Raises:
            ValueError: If any weight key is not a valid cost column.
        """
        # Validate weights
        invalid_weights = [col for col in weights if col not in self.cost_columns]
        if invalid_weights:
            msg = (
                f"Invalid cost columns in weights: {invalid_weights}\n"
                f"Available cost columns: {self.cost_columns}"
            )
            raise ValueError(msg)

        # Store weights internally
        self._objective_weights = weights

        # Create a dictionary to look up costs faster
        cost_matrix = self.data.set_index("pid")[list(weights.keys())].to_dict(
            orient="index",
        )

        # Precomputing weights per pid saves having to
        # construct a huge gurobi linear expression
        varnames = [f"x_{pid}" for pid in self.pids]
        weighted_cost = {
            f"x_{pid}": sum(weights[col] * cost_matrix[pid][col] for col in weights)
            for pid in self.pids
        }

        objective = gp.quicksum(
            weighted_cost[vn] * self._variables[vn] for vn in varnames
        )

        # Set as minimization objective
        self.model.setObjective(objective, gp.GRB.MINIMIZE)

        self.model.update()

    def get_selected_pids(self) -> list[int]:
        """Get the list of selected pathway IDs from the solved model.

        Returns:
            A list of pathway IDs that were selected (variable value = 1)

        Raises:
            RuntimeError: If the model has not been solved yet.
        """
        if self.model.SolCount == 0:
            msg = "Model has not been solved yet"
            raise RuntimeError(msg)

        self.model.update()

        return [
            pid
            for pid in self.pids
            if abs(self._variables[f"x_{pid}"].X - 1.0)
            < 1e-6  # Check if binary variable is 1
        ]

    def get_solution_summary(self) -> dict:
        """Get a summary of the optimization results.

        Returns:
            dict: Dictionary containing the results with the following keys:

                * **objective_value** (float): The final objective value.
                * **total_contribution** (float): Sum of contribution for selected pathways.
                * **solve_time** (float): Time taken to solve the model (seconds).
                * **selected_count** (int): Number of selected pathways.

        Raises:
            RuntimeError: If the model has not been solved yet.
        """
        if self.model.SolCount == 0:
            msg = "Model has not been solved yet"
            raise RuntimeError(msg)

        selected_pids = self.get_selected_pids()
        filtered = self.data[self.data["pid"].isin(selected_pids)]

        total_contribution = filtered["contribution"].sum()

        costs = {c: filtered[c].sum() for c in self.cost_columns}

        return {
            "objective_value": self.model.ObjVal,
            "cost_column_sums": costs,
            "total_contribution": total_contribution,
            "solve_time": self.model.Runtime,
            "selected_count": len(selected_pids),
        }

    def print_solution_summary(self) -> None:
        """Pretty-print a summary of the optimization results.

        Prints a formatted summary including:
        - Objective value
        - Total contribution of selected pathways
        - Solve time in seconds
        - Number of selected pathways

        Raises:
            RuntimeError: If the model has not been solved yet.
        """
        summary = self.get_solution_summary()

        print("Optimization Summary")
        print(f"  Objective Value   : {summary['objective_value']:.2f}")
        print(f"  Total Contribution: {summary['total_contribution']:.2f}")
        print(f"  Solve Time        : {summary['solve_time']:.3f} sec")
        print(f"  Selected Pathways : {summary['selected_count']}")

    def add_contribution_constraints(
        self,
        limits: float | list[float],
        sense: str,
        boundaries: None | Polygon | MultiPolygon | list = None,
        tag: str | None = None,
    ) -> gp.MConstr:
        """Add a constraint limiting the total contribution within a given boundary (or boundaries).

        Always returns a matrix constraint.

        Args:
            limits (float | list[float]): Maximum allowed total contribution (or a list of those).
            sense (str): One of "<=", ">=", or "=="
            boundaries (None | Polygon | MultiPolygon | list, optional): Optional boundary or list of boundaries.
            tag (str | None, optional): Optional tag for constraint tracking or removal.

        Returns:
            A Gurobi matrix constraint object

        Raises:
            TypeError: If boundaries are not valid geometry types.
            ValueError: If number of limits and boundaries do not match.
        """
        # Coerce inputs to lists
        if boundaries is None:
            boundaries = [None]
        if isinstance(limits, (float, int)):
            limits = [limits]

        if isinstance(boundaries, (Polygon, MultiPolygon)):
            boundaries = [boundaries]
        elif not isinstance(boundaries, list):
            msg = f"Unsupported boundaries type: {type(boundaries)}"
            raise TypeError(msg)

        if len(limits) != len(boundaries):
            msg = "Number of limits must match number of boundaries."
            raise ValueError(msg)

        if boundaries == [None]:
            # Case 1: No boundary specified: use all pathways
            filtered_pids = self.pids
            pid_to_group = dict.fromkeys(filtered_pids, 0)  # All go into group 0
        else:
            # Case 2: Boundaries specified → spatial join

            boundary_gdf = gpd.GeoDataFrame(
                {"limit": limits},
                geometry=boundaries,
                crs=self.data.crs,
            )
            boundary_gdf["group_id"] = boundary_gdf.index
            joined = gpd.sjoin(
                self.data[["pid", "contribution", "geometry"]],
                boundary_gdf[["group_id", "geometry"]],
                how="inner",
                predicate="intersects",
            )

            if joined.empty:
                msg = "No pathways intersect the given boundaries."
                raise ValueError(msg)

            filtered_pids = joined["pid"].unique()
            pid_to_group = dict(
                zip(joined["pid"], joined["group_id"], strict=True),
            )

        # Build mapping and sparse matrix
        pid_to_index = {pid: i for i, pid in enumerate(filtered_pids)}
        contribution_map = self.data.set_index("pid")["contribution"].to_dict()

        row_indices = []
        col_indices = []
        data = []

        for pid in filtered_pids:
            col_idx = pid_to_index[pid]
            group_idx = pid_to_group[pid]

            row_indices.append(group_idx)
            col_indices.append(col_idx)
            data.append(contribution_map[pid])

        coeffs = coo_matrix(
            (data, (row_indices, col_indices)),
            shape=(len(limits), len(filtered_pids)),
        ).tocsr()

        rhs_array = np.array(limits)

        varnames = [f"x_{pid}" for pid in filtered_pids]

        return self._register_matrix_constraint(
            varnames=varnames,
            coeffs=coeffs,
            sense=sense,
            rhs=rhs_array,
            tag=tag,
        )

    def add_zone_difference_constraints(
        self,
        geom_pairs: list[tuple[Polygon | MultiPolygon, Polygon | MultiPolygon]],
        sense: str,
        limits: list[float],
        tag: str | None = None,
    ) -> tuple[gp.MConstr, gp.MConstr]:
        """Constrain the (absolute) difference in contribution within two geometries within a given limit.

        For each pair of geometries (zone1, zone2), enforces:
            sum(contribution in zone1) - sum(contribution in zone2) <= limit
            sum(contribution in zone2) - sum(contribution in zone1) <= limit

        Args:
            geom_pairs: List of (geom1, geom2) tuples.
            sense (str): One of "<=", ">=", or "=="
            limits (float | list[float]): Maximum allowed total contribution (or a list of those).
            tag (str | None, optional): Optional tag for constraint tracking or removal.

        Returns:
            Two Gurobi matrix constraint objects for the absolute difference.

        Raises:
            ValueError: If geom_pairs and limits are not the same length.
        """
        # Create GeoDataFrame
        geoms1, geoms2 = zip(*geom_pairs, strict=True)
        indices = list(range(len(geom_pairs)))

        gdf1 = gpd.GeoDataFrame(
            {"row": indices, "side": 1, "geometry": geoms1},
            crs=self.data.crs,
        )
        gdf2 = gpd.GeoDataFrame(
            {"row": indices, "side": -1, "geometry": geoms2},
            crs=self.data.crs,
        )
        combined = gpd.GeoDataFrame(
            pd.concat([gdf1, gdf2], ignore_index=True),
            crs=self.data.crs,
        )
        # Join with data
        pid_data = self.data[["pid", "contribution", "geometry"]].copy()
        combined = combined.sjoin(pid_data, predicate="intersects", how="left")
        combined = combined[~combined["contribution"].isna()]
        combined["pid"] = combined["pid"].astype(int)
        combined["index_right"] = combined["index_right"].astype(int)

        # Duplicates occur when a pathway is right on a border. We will just drop them
        # and pretend like they contribute equally to both
        # HACK
        dupes = combined.duplicated(subset=["row", "pid"], keep=False)
        combined = combined[~dupes]

        # Create maps
        pid_set = combined["pid"].sort_values().unique().tolist()
        pid_to_index = {p: i for i, p in enumerate(pid_set)}
        varnames = [f"x_{pid}" for pid in pid_set]

        # Construct COO matrix
        row_indices = combined["row"].to_list()
        col_indices = combined["pid"].map(pid_to_index).to_list()
        combined["coeff"] = combined["contribution"] * combined["side"]
        values = combined["coeff"].to_list()

        coeffs = coo_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(limits), len(pid_set)),
        ).tocsr()

        rhs_array = np.array(limits, dtype=float)

        # Add both sides of absolute value constraint
        constrs1 = self._register_matrix_constraint(
            varnames=varnames,
            coeffs=coeffs,
            sense=sense,
            rhs=rhs_array,
            tag=f"{tag}_1",
        )

        constrs2 = self._register_matrix_constraint(
            varnames=varnames,
            coeffs=-1 * coeffs,
            sense=sense,
            rhs=rhs_array,
            tag=f"{tag}_2",
        )

        return constrs1, constrs2

    def add_conversion_constraints(
        self,
        start_name: str,
        sense: str,
        limit: int,
        check_overlaps: bool = False,
        proj_crs: str | None = None,
        debuff: float = 0,
        tag: str | None = None,
    ) -> gp.Constr:
        """Constrain the sum of pathways that start from a given "start" type.

        This method can be used to (e.g.) constrain the total number of pathways that start
        from "sfh" to 10, indicating that only 10 sfh can be converted.

        If check_overlaps is True, the method attempts to count overlapping pathways
        (potentially after a 'debuff') as a single unit in the sum. Otherwise,
        it simply constrains the sum of individual selected pathways.

        The constraint is of the form:
            sum(effective_pathways) [sense] limit

        Where 'effective_pathways' are individual pathways (x_pid) or
        indicator variables (z_group) if check_overlaps is True.

        Args:
            start_name (str): String from self.data["start] to filter on.
            sense (str): One of "<=", ">=", or "==" for the constraint.
            limit (int): The right-hand side value for the constraint.
            check_overlaps (bool, optional): If True, attempts to identify and group
                overlapping pathways. Defaults to False.
            proj_crs (str | None, optional): A projected CRS to use for geometric
                operations (buffer, overlap). Required if self.data CRS is geographic.
                Defaults to None.
            debuff (float, optional): A distance to shrink pathway geometries by before checking overlaps.
                Requires check_overlaps=True. Value must be non-negative. Defaults to 0.
            tag (str | None, optional): Optional tag for constraint tracking or removal. Defaults to None.

        Returns:
            The Gurobi constraint object added to the model.

        Raises:
            ValueError: If debuff is negative.
            ValueError: If no pathways start from start_name.
            ValueError: If data CRS is geographic and proj_crs is not provided.
        """
        # Checking arguments
        if debuff < 0:
            msg = "Value for parameter debuff must be >= 0."
            raise ValueError(msg)
        if debuff > 0 and not check_overlaps:
            msg = "Specifying debuff without setting check_overlaps=True will not do anything"
            warnings.warn(msg, stacklevel=2)

        filtered = self.data[self.data["start"] == start_name].copy()
        if len(filtered) < 1:
            msg = f"No pathways start from {start_name}."
            raise ValueError(msg)

        if proj_crs and not check_overlaps:
            msg = "Specifying proj_crs without setting check_overlaps=True will not do anything"
            warnings.warn(msg, stacklevel=2)
        elif proj_crs:
            filtered = filtered.to_crs(proj_crs)

        # Get relevant pids
        if check_overlaps:
            filtered["geometry"] = filtered.buffer(-debuff)

            grph = PolyGraph.from_geoseries(
                filtered.set_index("pid")["geometry"],  # type: ignore[arg-type]
                predicate="intersects",
            )
            id_to_cc, cc_to_ids = grph.get_connected_components_map()

            # every pid should be mapped to the set of its connected components
            pids = {p: cc_to_ids[id_to_cc[p]] for p in list(filtered["pid"].unique())}
        else:
            pids = {p: {p} for p in list(filtered["pid"].unique())}

        var_list: list[str] = []
        visited = set()

        for pid, overlap_set in pids.items():
            # If the pathway doesn't overlap with anything, just add it by itself
            if len(overlap_set) <= 1:
                var_list.append(f"x_{pid}")

            # Otherwise we add an auxillary variable
            elif overlap_set not in visited:
                suffix = "".join(str(i) for i in sorted(overlap_set))
                dummy_name = f"z_{suffix}"
                z = self.model.addVar(vtype=gp.GRB.BINARY, name=dummy_name)
                self._variables[dummy_name] = z

                constr = self.model.addGenConstrOr(
                    z,
                    [self._variables[f"x_{p}"] for p in overlap_set],
                )

                dummy_tag = f"{tag}_dummy" if tag else "dummy"
                self._constraints.setdefault(dummy_tag, []).append(constr)

                var_list.append(dummy_name)
                visited.add(frozenset(overlap_set))

        coeff_map = dict.fromkeys(var_list, 1.0)

        return self._register_constraint(
            varnames=var_list,
            coeff_map=coeff_map,
            sense=sense,
            rhs=limit,
            tag=tag,
        )

    def add_mutual_exclusion(
        self,
        label1: str,
        label2: str | None = None,
        tag: str | None = None,
    ) -> gp.MConstr | None:
        """Add mutual exclusion constraints between pathways based on their labels.

        If label2 is provided, creates constraints ensuring no pathway with label1 can be selected with an intersecting pathway with label2.
        If label2 is not provided, creates constraints ensuring no pathway with label1 can be selected with ANY intersecting pathway (regardless of label).

        Uses GeoPandas spatial join for efficient intersection detection.

        Args:
            label1: First pathway label to find mutual exclusions for.
            label2 (Optional, None): Pathways with which label1 cannot co-occur. Defaults to None (all pathways)
            tag: Optional tag for constraint tracking/removal.

        Returns:
            Gurobi matrix constraint object, or None if no overlaps are found
        """
        label1_paths = self.data[self.data["label"] == label1].copy()
        if label1_paths.empty:
            return None

        if label2 is None:  # i.e., Exclusive with all
            other_paths = self.data[self.data["label"] != label1].copy()
        else:
            other_paths = self.data[self.data["label"] == label2].copy()

        if other_paths.empty:
            return None

        # Perform spatial join: find intersecting geometry pairs
        joined = gpd.sjoin(
            other_paths,
            label1_paths,
            how="inner",
            predicate="intersects",
            lsuffix="other",
            rsuffix="label1",
        )

        # Construct variables
        pids = list(set(joined["pid_label1"]) | set(joined["pid_other"]))
        if not pids:  # No intersecting pathways found
            return None

        index_map = {pid: i for i, pid in enumerate(pids)}

        # Construct coefficient matrix
        seen_pairs = set()
        rhs_list = []
        rows = []
        cols = []
        data = []

        for _, row in joined.iterrows():
            pid1 = row["pid_label1"]
            pid2 = row["pid_other"]

            key = tuple(sorted((pid1, pid2)))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            # Add coefficients for this constraint
            rows.extend([len(rhs_list), len(rhs_list)])
            cols.extend([index_map[pid1], index_map[pid2]])
            data.extend([1.0, 1.0])  # Coefficients are 1.0 for both variables
            rhs_list.append(1.0)

        # Create sparse matrix
        coeffs = coo_matrix(
            (data, (rows, cols)),
            shape=(len(rhs_list), len(pids)),
        ).tocsr()

        # Construct the rhs
        rhs = np.array(rhs_list)

        # Convert to variable names
        varnames = [f"x_{pid}" for pid in pids]

        return self._register_matrix_constraint(
            varnames=varnames,
            coeffs=coeffs,
            rhs=rhs,
            sense="<=",
            tag=tag,
        )

    def add_max_contribution_near_point(
        self,
        limit: float,
        point: Point,
        distance: float,
        proj_crs: str | None = None,
        tag: str | None = None,
    ) -> gp.Constr:
        """Add a constraint limiting total contribution for pathways near a point.

        Args:
            limit: Maximum allowed total contribution across selected pathways.
            point: Shapely Point to measure distance from.
            distance: Maximum distance from point to pathway centroid.
            proj_crs: Optional CRS to project geometries into before distance calculation.
                 If not provided, uses the current CRS which must be projected.
            tag: Optional tag for constraint tracking/removal.

        Returns:
            The created Gurobi constraint object.

        Raises:
            ValueError: If no CRS is provided and current CRS is not projected.
        """
        # If CRS provided, project data and point
        if proj_crs is not None:
            data = self.data.copy().to_crs(proj_crs)
            point = _reproject_point(point, self.crs, proj_crs)
        else:
            if not self.data.crs or self.data.crs.is_geographic:
                msg = "Must provide projected CRS for distance calculation"
                raise ValueError(msg)
            data = self.data

        # Filter pathways by distance from point to centroid
        mask = data.geometry.centroid.distance(point) <= distance
        filtered_pids = data[mask]["pid"].tolist()

        coeff_map = {
            f"x_{pid}": self.data[self.data["pid"] == pid]["contribution"].iloc[0]
            for pid in filtered_pids
        }
        varnames = list(coeff_map.keys())

        return self._register_constraint(
            varnames=varnames,
            coeff_map=coeff_map,
            sense="<=",
            rhs=limit,
            tag=tag,
        )

    def remove_constraints(self, tag: str) -> None:
        """Remove all constraints associated with a given tag.

        Args:
            tag: The tag identifying which constraints to remove.

        Raises:
            ValueError: If the tag does not exist.
        """
        if tag not in self._constraints:
            msg = f"Tag '{tag}' not found in constraints"
            raise ValueError(msg)

        # Remove constraints from model
        for constr in self._constraints[tag]:
            self.model.remove(constr)

        # Remove constraints from tracking dictionary
        del self._constraints[tag]

        # Update model to reflect changes
        self.model.update()

    def solve(self, verbose: bool = False, logfile: str | None = None) -> None:
        """Solve the optimization model.

        This method optimizes the model with the current objective and constraints.
        After solving, use get_selected_pids() to get the selected pathways or
        get_solution_summary() for optimization results.

        Args:
            verbose: If True, prints a summary of the solution after solving.
                    Defaults to False.
            logfile: (str | None, optional) path to a file to log Gurobi output. Defaults to None.

        Raises:
            RuntimeError: If the model is infeasible, unbounded, or fails to solve
                        for any other reason.
        """
        if not verbose:
            self.model.setParam("LogToConsole", 0)
        if logfile:
            logpath = Path(logfile)
            logpath.parent.mkdir(parents=True, exist_ok=True)
            self.model.setParam("LogFile", str(logpath))

        self.model.optimize()
        self.model.update()
        status = self.model.Status

        if status == gp.GRB.INFEASIBLE:
            msg = "Model is infeasible"
            raise RuntimeError(msg)
        if status == gp.GRB.UNBOUNDED:
            msg = "Model is unbounded"
            raise RuntimeError(msg)
        if status != gp.GRB.OPTIMAL:
            msg = f"Optimization failed with status {status}"
            raise RuntimeError(msg)

        if not verbose:
            self.print_solution_summary()

    def save(self, path: str | Path) -> None:
        """Save the optimizer state to disk.

        This method saves:
        1. The GeoDataFrame to a GeoParquet file
        2. Model configuration and constraints to a JSON file

        Args:
            path: Base path for saving files (without extension).
                 Will create {path}.geoparquet and {path}.json.

        Raises:
            RuntimeError: If model has not been built yet.
            ValueError: If path is invalid.
        """
        if not hasattr(self, "model") or not hasattr(self, "_variables"):
            msg = "Model has not been built yet. Call build_variables() first."
            raise RuntimeError(msg)

        path = Path(path)
        if str(path) == "":
            msg = "Path cannot be empty"
            raise ValueError(msg)

        # Save GeoDataFrame to GeoParquet
        self.data.to_parquet(path.with_suffix(".geoparquet"))

        # Build constraint schema
        constraint_config: dict[str, list[ConstraintSchema]] = {
            tag: [self._serialize_constraint(c) for c in cs]
            for tag, cs in self._constraints.items()
        }

        # Build the full schema
        config = OptimizerConfigSchema(
            varnames=list(self._variables.keys()),
            cost_columns=self.cost_columns,
            objective=self._objective_weights
            if self._objective_weights
            else None,  # HACK
            constraints=constraint_config,
        )

        path.with_suffix(".json").write_text(config.model_dump_json(indent=2))

    def debug_model(self, verbose: bool = False, max_vars: int = 20) -> None:
        """Print debug information about the current state of the model.

        Prints basic model information including number of variables,
        constraints, and model status. If the model has been solved, also
        includes variable values.

        Args:
            verbose: If True, prints additional details about variables
                    and constraints. Defaults to False.
            max_vars: Maximum number of variables to print in verbose mode.
                     Defaults to 20.
        """
        # Calculate total constraints by handling both single constraints and MConstraints
        total_constraints = sum(
            len(constr) if hasattr(constr, "__len__") else 1
            for constrs in self._constraints.values()
            for constr in constrs
        )

        print("Gurobi Model Debug Info")
        print(f"- Variables: {len(self._variables)}")
        print(f"- Constraints: {total_constraints}")
        print(f"- Objective set: {bool(self._objective_weights)}")
        print(f"- Model status: {self.model.Status}")

        if verbose:
            if self._objective_weights:
                print("\nObjective Weights:")
                for col, weight in self._objective_weights.items():
                    print(f"  {col}: {weight}")

            print(f"\nVariables (first {max_vars}):")
            for i, (varname, var) in enumerate(self._variables.items()):
                if i >= max_vars:
                    print("  ... (truncated)")
                    break
                print(
                    f"  {varname}: {var.X if self.model.SolCount > 0 else 'not solved'}",
                )

            print("\nConstraint Tags:")
            for tag, constrs in self._constraints.items():
                # Count constraints properly for each tag
                tag_total = sum(len(c) if hasattr(c, "__len__") else 1 for c in constrs)
                print(f"  [{tag}] {tag_total} constraints")

    def export_model(self, path: str) -> None:
        """Export the current model to a human-readable .lp file for debugging.

        Args:
            path: Base path for the output file (without extension).
                 The .lp extension will be added automatically.
                 Can be either a relative or absolute path.
        """
        self.model.write(str(Path(path).with_suffix(".lp")))
        print(f"Model written to: {path}.lp")

    def _register_constraint(
        self,
        varnames: list[str],
        coeff_map: dict[str, float],
        sense: str,  # "<=", ">=", "=="
        rhs: float,
        tag: str | None = None,
        defer_update: bool = False,
    ) -> gp.Constr:
        """Add constraint to Gurobi model.

        Registers a constraint with the gurobi model, and stores a reference to it.

        Args:
            varnames: List of variable names involved in the constraint.
            coeff_map: Dictionary mapping pid to coefficient.
            sense: One of "<=", ">=", or "==".
            rhs: Right-hand-side limit.
            tag: Optional tag for constraint tracking/removal.
            defer_update: If False, calls model.update() before exiting. Defaults to False.

        Returns:
            The Gurobi constraint object.

        Raises:
            ValueError: If sense is not one of "<=", ">=", or "==".
        """
        terms = [coeff_map[vn] * self._variables[vn] for vn in varnames]
        expr = gp.quicksum(terms)

        if sense == "<=":
            constr = self.model.addConstr(expr <= rhs)
        elif sense == ">=":
            constr = self.model.addConstr(expr >= rhs)
        elif sense == "==":
            constr = self.model.addConstr(expr == rhs)
        else:
            msg = f"Invalid constraint sense: {sense}"
            raise ValueError(msg)

        # Store constraint with tag if provided
        self._constraints.setdefault(tag or "untagged", []).append(constr)

        if not defer_update:
            self.model.update()

        return constr

    def _register_matrix_constraint(
        self,
        varnames: list[str],
        coeffs: csr_array,
        sense: str,  # "<=", ">=", "=="
        rhs: np.ndarray,
        tag: str | None = None,
        defer_update: bool = False,
    ) -> gp.MConstr:
        """Register multiple constraints at once using matrix form.

        This is a more efficient way to add many constraints at once compared to
        adding them one by one. It uses Gurobi's matrix constraint API.

        Args:
            varnames: List of variable names involved in the constraints.
            coeffs: Sparse coefficient matrix where each row represents one constraint
                   and each column corresponds to a pathway variable.
            sense: One of "<=", ">=", or "==". The sense for all constraints.
            rhs: Right-hand-side values, one per constraint (row in coeffs).
            tag: Optional tag for constraint tracking/removal.
            defer_update: If False, calls model.update() before exiting. Defaults to False.

        Returns:
            The Gurobi matrix constraint object.

        Raises:
            ValueError: If sense is not one of "<=", ">=", or "==".
        """
        sense_map = {
            "<=": "<",
            ">=": ">",
            "==": "=",
        }
        if sense not in sense_map:
            msg = f"Invalid constraint sense: {sense}"
            raise ValueError(msg)
        else:
            sense = sense_map[sense]

        x = [self._variables[vn] for vn in varnames]

        constrs = self.model.addMConstr(coeffs, x, sense, rhs)  # type: ignore[attr-defined]

        self._constraints.setdefault(tag or "untagged", []).extend(constrs.tolist())

        if not defer_update:
            self.model.update()

        return constrs

    def _serialize_constraint(self, constr: gp.Constr) -> ConstraintSchema:
        """Serialize a Gurobi constraint into a schema.

        Args:
            constr: The Gurobi constraint to serialize.

        Returns:
            The serialized constraint data.
        """
        expr = self.model.getRow(constr)
        varnames, coeffs = [], []
        for i in range(expr.size()):
            varnames.append(expr.getVar(i).VarName)
            coeffs.append(expr.getCoeff(i))

        sense = "<=" if constr.Sense == "<" else ">=" if constr.Sense == ">" else "=="

        return ConstraintSchema(
            varnames=varnames,
            coeffs=coeffs,
            sense=sense,
            rhs=constr.RHS,
        )

    def _deserialize_and_register_constraint(
        self,
        constr: ConstraintSchema,
        tag: str,
    ) -> gp.Constr:
        """Deserialize and register a constraint from a schema.

        Args:
            constr: The constraint schema to deserialize.
            tag: The tag to associate with the constraint.

        Returns:
            The registered Gurobi constraint.
        """
        coeff_map = {
            constr.varnames[i]: constr.coeffs[i] for i, _ in enumerate(constr.varnames)
        }
        return self._register_constraint(
            varnames=constr.varnames,
            coeff_map=coeff_map,
            sense=constr.sense,
            rhs=constr.rhs,
            tag=tag,
        )


def _reproject_point(point: Point, from_crs: str, to_crs: str) -> Point:
    """Reproject a point from one CRS to another.

    Args:
        point: The point to reproject.
        from_crs: Source coordinate reference system.
        to_crs: Target coordinate reference system.

    Returns:
        The reprojected point.
    """
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    x, y = transformer.transform(point.x, point.y)
    return Point(x, y)


def _sjoin_greatest_intersection(
    target_df: gpd.GeoDataFrame,
    source_df: gpd.GeoDataFrame,
    variables: list[str],
) -> gpd.GeoDataFrame:
    """Join variables from source_df based on the largest intersection. In case of a tie it picks the first one.

    From https://pysal.org/tobler/_modules/tobler/area_weighted/area_join.html#area_join

    Args:
        target_df (geopandas.GeoDataFrame): GeoDataFrame containing target values.
        source_df (geopandas.GeoDataFrame): GeoDataFrame containing source values.
        variables (str or list-like): Column(s) in `source_df` dataframe for variable(s) to be joined.

    Returns:
        geopandas.GeoDataFrame: The `target_df` GeoDataFrame with joined variables as additional columns.

    """
    if not pd.api.types.is_list_like(variables):
        variables = [variables]  # type: ignore[assignment]

    for v in variables:
        if v in target_df.columns:
            msg = f"Column '{v}' already present in target_df."
            raise ValueError(msg)

    target_df = target_df.copy()
    target_ix, source_ix = source_df.sindex.query(
        target_df.geometry,
        predicate="intersects",
    )
    areas = (
        target_df.geometry.values[target_ix]  # noqa: PD011
        .intersection(source_df.geometry.values[source_ix])  # noqa: PD011
        .area
    )

    main = []
    for i in range(len(target_df)):  # vectorise this loop?
        mask = target_ix == i
        if np.any(mask):
            main.append(source_ix[mask][np.argmax(areas[mask])])
        else:
            main.append(np.nan)

    main = np.array(main, dtype=float)
    mask = ~np.isnan(main)

    for v in variables:
        arr = np.empty(len(main), dtype=object)
        arr[mask] = source_df[v].to_numpy()[main[mask].astype(int)]
        try:
            arr = arr.astype(source_df[v].dtype)  # type: ignore[assignment]
        except TypeError:
            warnings.warn(
                f"Cannot preserve dtype of '{v}'. Falling back to `dtype=object`.",
                stacklevel=2,
            )
        target_df[v] = arr

    return target_df
