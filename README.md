<div align="center">

![urbanopt logo](static/logo_no_bg.png)

[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/yourusername/urbanopt/releases)

*Optimization framework for urban housing pathway selection*

[Documentation](https://urbanopt.readthedocs.io) | [Installation](#installation) | [Examples](docs/examples)

</div>

---

## Overview

`urbanopt` is a Python package for housing development optimization. It uses binary linear programming (via Gurobi) to help planners make data-driven decisions about where and how housing should be built.

### Key Features

- **Geospatial Consntraints**: Constrain developments based on geographic boundaries and spatial relationships
- **High Resolution**: This one technically depends on how you set the data up, but UrbanOPT was designed to accomodate vectorized geometries of *individual development opportunities*
- **Efficient Solving**: Uses Gurobi as a backend to solve city-scale problems to optimality
- **Multiobjective Support**: Currently supports scalarized MOO, which works well with the rapid solve times

## Installation

```bash
# Installation instructions coming soon
pip install urbanopt

# Requires Gurobi license (academic licenses available free)
```

See the [installation guide](docs/installation.md) for detailed setup instructions. *(coming soon)*

## Quick Start

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
opt.add_contribution_constraints(limits=1000, sense=">=", tag="min_contribution")
opt.add_mutual_exclusion_constraints(label1="adu", label2="sfh", tag="exclusions")

# Solve and get results
opt.solve()
print(opt.get_solution_summary())

# Save the model
opt.save("results/my_optimization")
```

## Documentation

Full documentation is available at [urbanopt.readthedocs.io](https://urbanopt.readthedocs.io) *(coming soon)*

- [User Guide](docs/guide.md)
- [API Reference](docs/api.md)
- [Examples](docs/examples)

## Citation *(coming soon)*

If you use `urbanopt` in your research, please cite:

```bibtex
...
```