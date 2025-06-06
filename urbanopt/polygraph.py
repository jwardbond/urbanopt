"""Creating an manipulating graphs made from geodata."""

from typing import Self

import geopandas as gpd


class PolyGraph:
    """A graph created from geodata.

    Typically created using PolyGraph.create_from_geoseries()

    Supported predicates for building the graph are:
        - `intersects`
        - `overlaps`

    Attributes:
        adj_list: The generated graph. Stored as an adjacency list ({id: {neighbour ids}}).
            ids correspond to indices in the original geoseries.
    """

    def __init__(self, adj_list: dict):
        """Create PolyGraph object.

        Args:
            adj_list (dict): The graph of adjacent polygons in an adjaceny list form ({id: {neighbour ids}}).
        """
        self.adj_list = adj_list

    @classmethod
    def from_geoseries(
        cls,
        gs: gpd.GeoSeries,
        predicate: str = "intersects",
    ) -> Self:
        """Create a PolyGraph object from a geoseries.

        Args:
            gs (gpd.GeoSeries): Input geoseries. Preferrably in a projected crs
            predicate (str, optional): Binary predicate to use to determine neighbouring geometries.
                Defaults to "intersects"
        """
        gs = gs.copy()

        original_index = gs.index.to_list()
        sindex = gs.sindex

        graph = {}
        for idx, poly in gs.items():
            neighbour_sidxs = sindex.query(poly, predicate)
            neighbours = {original_index[i] for i in neighbour_sidxs}

            # Some predicates will produce self-edges
            if idx in neighbours:
                neighbours.remove(idx)

            graph[idx] = neighbours

        return PolyGraph(adj_list=graph)

    def get_connected_components_map(self) -> dict:
        """Generates {id: cc_id} and {cc_id: {ids}} maps.

        cc_id is the connected component to which a given polygon id belongs.

        Returns:
            dict: The {id: cc_id} map
        """
        graph = self.adj_list

        id_to_cc = {}
        cc_to_ids = {}

        group_id = 0
        for v in graph:
            if v not in id_to_cc:
                connected_ids = self.get_connected(v)
                id_to_cc.update(dict.fromkeys(connected_ids, group_id))
                cc_to_ids[group_id] = connected_ids
                group_id += 1

        return id_to_cc, cc_to_ids

    def get_connected(self, root: int) -> set[int]:
        """Given a root node, generates a list of connected nodes with DFS."""
        stack = [root]
        visited = set()

        while len(stack) > 0:
            v = stack.pop(-1)
            if v not in visited:
                visited.add(v)
                stack = stack + list(self.adj_list[v])

        return visited
