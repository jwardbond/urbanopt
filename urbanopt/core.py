import geopandas as gpd
import gurobipy as gp
from shapely.geometry import Polygon, MultiPolygon


class PathwayOptimizer:
    """Optimizer for housing pathway selection based on multiple criteria."""

    def __init__(self, gdf: gpd.GeoDataFrame) -> None:
        """Initialize the PathwayOptimizer.

        Args:
            gdf: GeoDataFrame containing pathway data with required columns:
                - pid: Pathway identifier
                - start: Start point/area
                - end: End point/area
                - desc: Description
                - opportunity: Opportunity score
                - geometry: Shapely geometry
                - cost_*: Cost-related columns

        Raises:
            ValueError: If any required columns are missing from the GeoDataFrame.
        """
        # Validate required columns
        required_columns = ["pid", "start", "end", "desc", "opportunity", "geometry"]
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        if missing_columns:
            msg = f"Missing required columns: {missing_columns}"
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
        self.constraints = []

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
        self, limit: float, boundary: Polygon | MultiPolygon | None = None
    ) -> None:
        """Add a constraint limiting the total opportunity.

        Args:
            limit: Maximum allowed total opportunity across all selected pathways.
            boundary: Optional shapely polygon or multipolygon to filter pathways. Only pathways
                     that intersect with this boundary will be included in the constraint.
        """
        # Filter pathways by boundary if provided
        if boundary is not None:
            mask = self.data.geometry.intersects(boundary)
            filtered_pids = self.data[mask]["pid"].tolist()
        else:
            filtered_pids = self.pids

        # Create opportunity sum expression for filtered pathways
        opportunity_sum = gp.quicksum(
            self.data[self.data["pid"] == pid]["opportunity"].iloc[0]
            * self.variables[pid]
            for pid in filtered_pids
        )

        # Add constraint and store it
        constraint = self.model.addConstr(opportunity_sum <= limit)
        self.constraints.append(constraint)

        # Update the model to include the new constraint
        self.model.update()

    def add_min_opportunity(
        self, limit: float, boundary: Polygon | MultiPolygon | None = None
    ) -> None:
        """Add a constraint requiring a minimum total opportunity.

        Args:
            limit: Minimum required total opportunity across all selected pathways.
            boundary: Optional shapely polygon or multipolygon to filter pathways. Only pathways
                     that intersect with this boundary will be included in the constraint.
        """
        # Filter pathways by boundary if provided
        if boundary is not None:
            mask = self.data.geometry.intersects(boundary)
            filtered_pids = self.data[mask]["pid"].tolist()
        else:
            filtered_pids = self.pids

        # Create opportunity sum expression for filtered pathways
        opportunity_sum = gp.quicksum(
            self.data[self.data["pid"] == pid]["opportunity"].iloc[0]
            * self.variables[pid]
            for pid in filtered_pids
        )

        # Add constraint and store it
        constraint = self.model.addConstr(opportunity_sum >= limit)
        self.constraints.append(constraint)

        # Update the model to include the new constraint
        self.model.update()
