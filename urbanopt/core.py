import json
from pathlib import Path
from typing import Literal

import geopandas as gpd
import gurobipy as gp
from pydantic import BaseModel
from pyproj import Transformer
from shapely.geometry import MultiPolygon, Point, Polygon


class ConstraintSchema(BaseModel):
    """Schema for serializing constraints."""

    pids: list[int]
    coeffs: list[float]
    sense: Literal["<=", ">=", "=="]
    rhs: float


class OptimizerConfigSchema(BaseModel):
    """Schema for serializing a PathywayOptimizer Configuration."""

    pids: list[int]
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
        constraints (dict[str, list[gp.Constr]]): Dictionary mapping tags to lists of constraints.
        model (gp.Model): Gurobi optimization model (initialized by build_variables).
        variables (dict[int, gp.Var]): Dictionary mapping pathway IDs to Gurobi variables
            (initialized by build_variables).

    Internal Attributes:
        _index_to_pid (dict[int, int]): Dictionary mapping variable indices to pathway IDs.
        _pid_to_index (dict[int, int]): Dictionary mapping pathway IDs to variable indices.

    Example:
        >>> import geopandas as gpd
        >>> from shapely.geometry import Point
        >>> data = {
        ...     "pid": [1, 2, 3],
        ...     "start": ["A", "B", "C"],
        ...     "end": ["X", "Y", "Z"],
        ...     "desc": ["desc1", "desc2", "desc3"],
        ...     "opportunity": [1.0, 2.0, 3.0],
        ...     "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        ...     "cost_emb": [10, 20, 30],
        ...     "cost_transit": [5, 15, 25]
        ... }
        >>> gdf = gpd.GeoDataFrame(data, crs="EPSG:4326")
        >>> optimizer = PathwayOptimizer(gdf)
        >>> optimizer.build_variables()
        >>> optimizer.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})
        >>> optimizer.add_max_opportunity(5.0, tag="constraints")
    """

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        """Initialize the PathwayOptimizer.

        Args:
            gdf: GeoDataFrame containing pathway data with required columns:
                - pid: Pathway identifier (integer)
                - start: Start point/area
                - end: End point/area
                - desc: Description
                - opportunity: Opportunity score
                - geometry: Shapely geometry
                - cost_*: Cost-related columns

        Raises:
            ValueError: If any required columns are missing from the GeoDataFrame,
                       or if any pid values are not integers.
        """
        # Validate required columns
        required_columns = ["pid", "start", "end", "desc", "opportunity", "geometry"]
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
        self.crs = gdf.crs

        # Store list of pathway IDs
        self.pids = gdf["pid"].tolist()

        # Extract and store cost columns
        self.cost_columns = [col for col in gdf.columns if col.startswith("cost_")]

        # Initialize constraint tracking
        self.constraints = {}

        # Initialize index mapping dictionaries
        self._index_to_pid = {}
        self._pid_to_index = {}

        # Initialize objective function
        self._objective_weights = {}

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

        if config.pids != data["pid"].tolist():
            msg = "Saved configuration does not match loaded data"
            raise ValueError(msg)

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

        Creates one binary variable per pathway ID and stores them in self.variables.
        Also creates and stores the Gurobi model in self.model.
        """
        # Create the model
        self.model = gp.Model("pathway_optimizer")

        # Create binary variables for each pathway
        self.variables = {
            pid: self.model.addVar(vtype=gp.GRB.BINARY, name=f"x_{pid}")
            for pid in self.pids
        }

        # Build index mapping dictionaries
        for pid, var in self.variables.items():
            self._index_to_pid[var.index] = pid
            self._pid_to_index[pid] = var.index

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
            msg = f"Invalid cost columns in weights: {invalid_weights}"
            raise ValueError(msg)

        # Store weights internally
        self._objective_weights = weights

        # Create a dictionary to look up costs faster
        cost_matrix = self.data.set_index("pid")[list(weights.keys())].to_dict(
            orient="index",
        )

        # Precomputing weights per pid saves having to
        # construct a huge gurobi linear expression
        weighted_cost = {
            pid: sum(weights[col] * cost_matrix[pid][col] for col in weights)
            for pid in self.pids
        }
        objective = gp.quicksum(weighted_cost[pid] for pid in self.pids)

        # Set as minimization objective
        self.model.setObjective(objective, gp.GRB.MINIMIZE)

    def get_selected_pids(self) -> list[int]:
        """Get the list of selected pathway IDs from the solved model.

        Returns:
            List of pathway IDs that were selected (variable value = 1).

        Raises:
            RuntimeError: If the model has not been solved yet.
        """
        if self.model.SolCount == 0:
            msg = "Model has not been solved yet"
            raise RuntimeError(msg)

        self.model.update()

        return [
            pid
            for pid, var in self.variables.items()
            if abs(var.X - 1.0) < 1e-6  # Check if binary variable is 1
        ]

    def get_summary(self) -> dict:
        """Get a summary of the optimization results.

        Returns:
            Dictionary containing:
                - objective_value: The final objective value
                - total_opportunity: Sum of opportunity for selected pathways
                - solve_time: Time taken to solve the model (seconds)
                - selected_count: Number of selected pathways

        Raises:
            RuntimeError: If the model has not been solved yet.
        """
        if self.model.SolCount == 0:
            msg = "Model has not been solved yet"
            raise RuntimeError(msg)

        self.model.update()  # Ensure model state is current before reading values
        selected_pids = self.get_selected_pids()
        total_opportunity = self.data[self.data["pid"].isin(selected_pids)][
            "opportunity"
        ].sum()

        return {
            "objective_value": self.model.ObjVal,
            "total_opportunity": total_opportunity,
            "solve_time": self.model.Runtime,
            "selected_count": len(selected_pids),
        }

    def remove_constraints(self, tag: str) -> None:
        """Remove all constraints associated with a given tag.

        Args:
            tag: The tag identifying which constraints to remove.

        Raises:
            ValueError: If the tag does not exist.
        """
        if tag not in self.constraints:
            msg = f"Tag '{tag}' not found in constraints"
            raise ValueError(msg)

        # Remove constraints from model
        for constr in self.constraints[tag]:
            self.model.remove(constr)

        # Remove constraints from tracking dictionary
        del self.constraints[tag]

        # Update model to reflect changes
        self.model.update()

    def _add_constraint(
        self,
        pids: list[int],
        coeff_map: list[float],
        sense: str,  # "<=", ">=", "=="
        rhs: float,
        tag: str | None = None,
    ) -> gp.Constr:
        """Generic constraint builder.

        Args:
            pids: List of pathway IDs involved.
            coeff_map: Dictionary mapping pid to coefficient.
            sense: One of "<=", ">=", or "==".
            rhs: Right-hand-side limit.
            tag: Optional tag for constraint tracking/removal.

        Returns:
            The created Gurobi constraint object.

        Raises:
            ValueError: If sense is not one of "<=", ">=", or "==".
        """
        expr = gp.quicksum(coeff_map[pid] * self.variables[pid] for pid in pids)

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
        if tag:
            if tag not in self.constraints:
                self.constraints[tag] = []
            self.constraints[tag].append(constr)
        else:
            if "untagged" not in self.constraints:
                self.constraints["untagged"] = []
            self.constraints["untagged"].append(constr)

        self.model.update()
        return constr

    def add_max_opportunity(
        self,
        limit: float,
        boundary: Polygon | MultiPolygon | None = None,
        tag: str | None = None,
    ) -> gp.Constr:
        """Add a constraint limiting the total opportunity.

        Args:
            limit: Maximum allowed total opportunity across all selected pathways.
            boundary: Optional shapely polygon or multipolygon to filter pathways. Only pathways
                     that intersect with this boundary will be included in the constraint.
            tag: Optional tag for constraint tracking/removal.

        Returns:
            The created Gurobi constraint object.
        """
        # Filter pathways by boundary if provided
        if boundary is not None:
            mask = self.data.geometry.intersects(boundary)
            filtered_pids = self.data[mask]["pid"].tolist()
        else:
            filtered_pids = self.pids

        coeff_map = {
            pid: self.data[self.data["pid"] == pid]["opportunity"].iloc[0]
            for pid in filtered_pids
        }

        return self._add_constraint(
            pids=filtered_pids,
            coeff_map=coeff_map,
            sense="<=",
            rhs=limit,
            tag=tag,
        )

    def add_min_opportunity(
        self,
        limit: float,
        boundary: Polygon | MultiPolygon | None = None,
        tag: str | None = None,
    ) -> gp.Constr:
        """Add a constraint requiring a minimum total opportunity.

        Args:
            limit: Minimum required total opportunity across all selected pathways.
            boundary: Optional shapely polygon or multipolygon to filter pathways. Only pathways
                     that intersect with this boundary will be included in the constraint.
            tag: Optional tag for constraint tracking/removal.

        Returns:
            The created Gurobi constraint object.
        """
        # Filter pathways by boundary if provided
        if boundary is not None:
            mask = self.data.geometry.intersects(boundary)
            filtered_pids = self.data[mask]["pid"].tolist()
        else:
            filtered_pids = self.pids

        coeff_map = {
            pid: self.data[self.data["pid"] == pid]["opportunity"].iloc[0]
            for pid in filtered_pids
        }

        return self._add_constraint(
            pids=filtered_pids,
            coeff_map=coeff_map,
            sense=">=",
            rhs=limit,
            tag=tag,
        )

    def add_mutual_exclusion(
        self,
        exclusion_pair: tuple[tuple[str, str], tuple[str, str]],
        tag: str | None = None,
    ) -> list[gp.Constr]:
        """Add mutual exclusion constraints between a pair of pathway types based on start/end types.

        For the given pair of pathway types specified by their start/end points, this method:
        1. Finds all pathways matching each type specification
        2. For each pair of intersecting pathways between the groups, adds a constraint
           ensuring at most one can be selected.

        Args:
            exclusion_pair: A tuple containing two (start, end) tuples specifying which pathway
                         types should be mutually exclusive.
                         Example: (("A", "X"), ("B", "Y")) means pathways from A to X
                         cannot be selected together with pathways from B to Y if they intersect.
            tag: Optional tag for constraint tracking/removal.

        Returns:
            List of created Gurobi constraint objects.
        """
        constraints = []
        (start1, end1), (start2, end2) = exclusion_pair

        # Find pathways matching each type specification
        mask1 = (self.data["start"] == start1) & (self.data["end"] == end1)
        mask2 = (self.data["start"] == start2) & (self.data["end"] == end2)
        pids1 = self.data[mask1]["pid"].tolist()
        pids2 = self.data[mask2]["pid"].tolist()

        # For each pair of pathways between the groups
        for pid1 in pids1:
            geom1 = self.data[self.data["pid"] == pid1]["geometry"].iloc[0]
            for pid2 in pids2:
                geom2 = self.data[self.data["pid"] == pid2]["geometry"].iloc[0]
                if geom1.intersects(geom2):
                    coeff_map = {pid1: 1.0, pid2: 1.0}
                    constr = self._add_constraint(
                        pids=[pid1, pid2],
                        coeff_map=coeff_map,
                        sense="<=",
                        rhs=1.0,
                        tag=tag,
                    )
                    constraints.append(constr)

        return constraints

    def add_pathway_limit(
        self,
        start: str,
        end: str,
        max_count: float,
        boundary: Polygon | MultiPolygon | None = None,
        tag: str | None = None,
    ) -> gp.Constr:
        """Add a constraint limiting the number of selected pathways of a specific type.

        Args:
            start: Start point/area identifier.
            end: End point/area identifier.
            max_count: Maximum number of pathways that can be selected.
            boundary: Optional shapely polygon or multipolygon to filter pathways. Only pathways
                     that intersect with this boundary will be included in the constraint.
            tag: Optional tag for constraint tracking/removal.

        Returns:
            The created Gurobi constraint object.
        """
        # Filter pathways by start/end
        mask = (self.data["start"] == start) & (self.data["end"] == end)

        # Apply boundary filter if provided
        if boundary is not None:
            mask = mask & self.data.geometry.intersects(boundary)

        filtered_pids = self.data[mask]["pid"].tolist()

        coeff_map = dict.fromkeys(filtered_pids, 1.0)

        return self._add_constraint(
            pids=filtered_pids,
            coeff_map=coeff_map,
            sense="<=",
            rhs=max_count,
            tag=tag,
        )

    def add_max_opportunity_near_point(
        self,
        limit: float,
        point: Point,
        distance: float,
        proj_crs: str | None = None,
        tag: str | None = None,
    ) -> gp.Constr:
        """Add a constraint limiting total opportunity for pathways near a point.

        Args:
            limit: Maximum allowed total opportunity across selected pathways.
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
            pid: self.data[self.data["pid"] == pid]["opportunity"].iloc[0]
            for pid in filtered_pids
        }

        return self._add_constraint(
            pids=filtered_pids,
            coeff_map=coeff_map,
            sense="<=",
            rhs=limit,
            tag=tag,
        )

    def solve(self) -> None:
        """Solve the optimization model.

        This method optimizes the model with the current objective and constraints.
        After solving, use get_selected_pids() to get the selected pathways or
        get_summary() for optimization results.

        Raises:
            RuntimeError: If the model is infeasible or unbounded.
        """
        self.model.optimize()
        self.model.update()  # Ensure model state is current before status check
        status = self.model.Status

        if status == gp.GRB.INFEASIBLE:
            msg = "Model is infeasible"
            raise RuntimeError(msg)
        elif status == gp.GRB.UNBOUNDED:
            msg = "Model is unbounded"
            raise RuntimeError(msg)
        elif status != gp.GRB.OPTIMAL:
            msg = f"Optimization failed with status {status}"
            raise RuntimeError(msg)

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
        if not hasattr(self, "model") or not hasattr(self, "variables"):
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
            for tag, cs in self.constraints.items()
        }

        # Build the full schema
        config = OptimizerConfigSchema(
            pids=self.pids,
            cost_columns=self.cost_columns,
            objective=self._objective_weights
            if self._objective_weights
            else None,  # HACK
            constraints=constraint_config,
        )

        path.with_suffix(".json").write_text(config.model_dump_json(indent=2))

    def _serialize_constraint(self, constr: gp.Constr) -> ConstraintSchema:
        expr = self.model.getRow(constr)
        pids, coeffs = [], []
        for i in range(expr.size()):
            var = expr.getVar(i)
            pid = self._index_to_pid[var.index]
            pids.append(pid)
            coeffs.append(expr.getCoeff(i))

        sense = "<=" if constr.Sense == "<" else ">=" if constr.Sense == ">" else "=="

        return ConstraintSchema(pids=pids, coeffs=coeffs, sense=sense, rhs=constr.RHS)

    def _deserialize_and_register_constraint(
        self,
        constr: ConstraintSchema,
        tag: str,
    ) -> gp.Constr:
        coeff_map = {
            constr.pids[i]: constr.coeffs[i] for i, _ in enumerate(constr.pids)
        }
        return self._add_constraint(
            pids=constr.pids,
            coeff_map=coeff_map,
            sense=constr.sense,
            rhs=constr.rhs,
            tag=tag,
        )


def _reproject_point(point: Point, from_crs: str, to_crs: str) -> Point:
    transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
    x, y = transformer.transform(point.x, point.y)
    return Point(x, y)
