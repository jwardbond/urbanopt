import geopandas as gpd
import gurobipy as gp
from shapely.geometry import Polygon, MultiPolygon
from typing import Callable


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
        if not gdf["pid"].dtype.kind == "i":
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
        self.constraints = {}  # Changed to dict to support tags

    def _add_constraint(
        self,
        pids: list[int],
        coeff_func: Callable[
            [int], float
        ],  # e.g. lambda pid: 1 or lambda pid: self.data.loc[self.data["pid"] == pid, "opportunity"].iloc[0]
        sense: str,  # "le", "ge", "eq"
        rhs: float,
        tag: str | None = None,
    ) -> gp.Constr:
        """Generic constraint builder.

        Args:
            pids: List of pathway IDs involved.
            coeff_func: Function returning coefficient per pid.
            sense: One of "le", "ge", or "eq".
            rhs: Right-hand-side limit.
            tag: Optional tag for constraint tracking/removal.

        Returns:
            The created Gurobi constraint object.

        Raises:
            ValueError: If sense is not one of "le", "ge", or "eq".
        """
        expr = gp.quicksum(coeff_func(pid) * self.variables[pid] for pid in pids)

        if sense == "le":
            constr = self.model.addConstr(expr <= rhs)
        elif sense == "ge":
            constr = self.model.addConstr(expr >= rhs)
        elif sense == "eq":
            constr = self.model.addConstr(expr == rhs)
        else:
            raise ValueError(f"Invalid constraint sense: {sense}")

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
        invalid_weights = [
            col for col in weights.keys() if col not in self.cost_columns
        ]
        if invalid_weights:
            msg = f"Invalid cost columns in weights: {invalid_weights}"
            raise ValueError(msg)

        # Create weighted sum expression
        objective = gp.quicksum(
            weights[col] * self.data[col][i] * self.variables[pid]
            for col in weights
            for i, pid in enumerate(self.pids)
        )

        # Set as minimization objective
        self.model.setObjective(objective, gp.GRB.MINIMIZE)

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

        def opportunity_coeff(pid: int) -> float:
            return self.data[self.data["pid"] == pid]["opportunity"].iloc[0]

        return self._add_constraint(
            pids=filtered_pids,
            coeff_func=opportunity_coeff,
            sense="le",
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

        def opportunity_coeff(pid: int) -> float:
            return self.data[self.data["pid"] == pid]["opportunity"].iloc[0]

        return self._add_constraint(
            pids=filtered_pids,
            coeff_func=opportunity_coeff,
            sense="ge",
            rhs=limit,
            tag=tag,
        )
