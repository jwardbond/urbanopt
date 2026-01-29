# UrbanOPT

## Installation
```python
pip install urbanopt
```

## Usage

```python
import geopandas as gpd
from urbanopt import UrbanOPT

# Load your pathway data
gdf = gpd.read_file("pathways.geojson")

# Initialize optimizer
opt = UrbanOPT(gdf)
opt.build_variables()

# Set objective to minimize costs
opt.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})

# Add constraints
opt.add_contribution_constraints(
    limits=1000, 
    sense=">=", 
    tag="min_population"
    )

opt.add_mutual_exclusion_constraints(
    label1="house", 
    label2="condo_tower", 
    tag="exclusions"
    )

# Solve and get results
opt.solve()
print(opt.get_solution_summary())

# Save the model
opt.save("results/my_optimization")
```