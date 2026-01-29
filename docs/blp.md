# Binary Linear Programming Formulation

The goal, then, is to select the subset of these development opportunities that minimizes an objective while

1. being above (below) a minimum (maximum) contribution, and
2. obeying geospatial constraints.

The BLP model is as follows, with variables outlined in the table below.

## Constants and Variables

| Symbol | Definition |
| :--- | :--- |
| **Sets** | |
| \(\mathcal{E}(\mathcal{P'})\) | Set of all unordered pairs of overlapping development opportunities with pathway types \(\mathcal{P'} \subseteq \mathcal{P}\) |
| \(\mathcal{G}(\mathcal{P'})\) | Set of all connected components (groups) of overlapping development opportunities with pathways \(\mathcal{P'} \subseteq \mathcal{P}\) |
| \(\mathcal{I}\) | Set of development opportunities, indexed by \(i\) |
| \(\mathcal{I}_n\) | Set of development opportunities within neighbourhood \(n \in \mathcal{N}\) |
| \(\mathcal{I}_p\) | Set of development opportunities of pathway \(p \in \mathcal{P}\) |
| \(\mathcal{I}^0(\mathcal{P'})\) | Set of development opportunities with pathways \(\mathcal{P'} \subseteq \mathcal{P}\) that have no overlaps, i.e., \(i\) is not in any pair within \(\mathcal{E}(\mathcal{P'})\) |
| \(\mathcal{J}\) | Set of cost types, indexed by \(j\) |
| \(\mathcal{N}\) | Set of bounded neighbourhoods, indexed by \(n\) |
| \(\mathcal{P}\) | Set of development opportunity types (pathways), indexed by \(p\) |
| \(\mathcal{P}_{\text{ex}}\) | Set of pathways that are mutually exclusive (e.g., cannot be developed at the same location), \(\mathcal{P}_{\text{ex}} \subseteq \mathcal{P}\) |
| **Constants** | |
| \(c_{i,j}\) | Cost of type \(j \in \mathcal{J}\) for opportunity \(i \in \mathcal{I}\) |
| \(O_{\mathcal{P'}}\) | Maximum number of opportunities that can be developed with pathways \(\mathcal{P'} \subseteq \mathcal{P}\) |
| \(\pi_i\) | Contribution (e.g., population increase) achieved by developing opportunity \(i \in \mathcal{I}\) |
| \(\Pi\) | Minimum required population increase within the whole region |
| \(\Pi_n\) | Maximum population increase within neighbourhood \(n \in \mathcal{N}\) |
| **Variables** | |
| \(x_i\) | Equals 1 if opportunity \(i\) is chosen for development, 0 otherwise |
| \(z_g\) | Equals 1 if any \(x_i\) in group \(g \in \mathcal{G}(\mathcal{P'})\) is selected, 0 otherwise |

## Formulation

### Objective Function

\[
\underset{\mathbf{x}}{\text{minimize}} \quad \sum_{j}\sum_{i}\lambda_j c_{i,j}\cdot x_i \tag{1}
\]

Minimizes the total cost of developments where each cost type is given weight \(\lambda_j\).

### Constraint 1: Minimum Population Target

\[
\sum_{i \in \mathcal{I}} \pi_i x_i \geq \Pi \tag{2}
\]

Ensures that the added housing across the entire region is at least \(\Pi\).

### Constraint 2: Neighbourhood Population Limits

\[
\sum_{i \in \mathcal{I}_n}x_i \leq \Pi_n \quad \forall \; n \in \mathcal{N} \tag{3}
\]

Sets *maximum* contribution increases on a per-neighbourhood basis.

### Constraint 3: Mutual Exclusivity

\[
x_i + x_{i'} \leq 1 \quad \forall \; (i, i') \in \mathcal{E}(\mathcal{P}_{\text{ex}}) \tag{4}
\]

Prevents two overlapping development opportunities from being constructed at the same time, but only insofar as those opportunities are incompatible. For instance, while you can't build a single family home (SFH) and a condo tower at the same spot, there is no reason you cannot add a basement suite and a garden suite to the same SFH property.

### Constraint 4: Group Indicators

\[
z_g = \bigvee_{i\in g} x_i \quad \forall\; g \in \mathcal{G}(\mathcal{P}') \tag{5}
\]

Creates dummy variables which are used only when developing maximum contribution constraints and only in the case were certain development opportunities are allowed to co-occur at the same location.

### Constraint 5: Pathway Capacity

\[
\sum_{i \in \mathcal{I}^0(\mathcal{P'})}x_i + \sum_{g \in \mathcal{G}(\mathcal{P'})}z_g \leq O_{\mathcal{P'}} \tag{6}
\]

Limits the total number of development opportunities for a subset of pathways.