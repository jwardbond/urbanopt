import geopandas as gpd
import pytest
import shapely

from urbanopt.polygraph import PolyGraph


class TestPolyGraphIntersects:
    def test_create_from_three_overlapping_polygons(self):
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = shapely.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        poly3 = shapely.Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = PolyGraph.create_from_geoseries(gs)
        assert graph.adj_list[0] == {1, 2}
        assert graph.adj_list[1] == {0, 2}
        assert graph.adj_list[2] == {0, 1}

    def test_create_from_touching(self):
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        poly3 = shapely.Polygon([(2, 1), (3, 1), (3, 2), (2, 2)])
        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = PolyGraph.create_from_geoseries(gs)
        assert graph.adj_list[0] == {1}
        assert graph.adj_list[1] == {0, 2}
        assert graph.adj_list[2] == {1}

    def test_create_from_transitive_overlapping_polygons(self):
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = shapely.Polygon([(2, 0), (4, 0), (4, 2), (2, 2)])
        poly3 = shapely.Polygon([(4, 0), (6, 0), (6, 2), (4, 2)])
        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = PolyGraph.create_from_geoseries(gs)
        assert graph.adj_list[0] == {1}
        assert graph.adj_list[1] == {0, 2}
        assert graph.adj_list[2] == {1}

    def test_create_from_empty_geoseries(self):
        gs = gpd.GeoSeries([])
        graph = PolyGraph.create_from_geoseries(gs)
        assert graph.adj_list == {}

    def test_create_from_non_sequential_index(self):
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)])
        poly3 = shapely.Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        gs = gpd.GeoSeries({0: poly1, 2: poly2, 3: poly3})
        graph = PolyGraph.create_from_geoseries(gs)
        assert set(graph.adj_list.keys()) == {0, 2, 3}


class TestPolyGraphOverlaps:
    def test_create_from_three_overlapping_polygons(self):
        poly1 = shapely.Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        poly2 = shapely.Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
        poly3 = shapely.Polygon([(2, 2), (4, 2), (4, 4), (2, 4)])
        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = PolyGraph.create_from_geoseries(gs, predicate="overlaps")
        assert graph.adj_list[0] == {1}
        assert graph.adj_list[1] == {0, 2}
        assert graph.adj_list[2] == {1}

    def test_create_from_touching(self):
        poly1 = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = shapely.Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
        poly3 = shapely.Polygon([(2, 1), (3, 1), (3, 2), (2, 2)])
        gs = gpd.GeoSeries([poly1, poly2, poly3])
        graph = PolyGraph.create_from_geoseries(gs, predicate="overlaps")
        assert graph.adj_list[0] == set()
        assert graph.adj_list[1] == set()
        assert graph.adj_list[2] == set()


class TestConnectedComponents:
    def test_dfs_connected_nodes(self):
        pg = PolyGraph({0: {1, 2}, 1: {0, 2}, 2: {0, 1}, 3: set()})
        connected = pg.get_connected(0)
        assert set(connected) == {0, 1, 2}

    def test_isolated_root_node(self):
        pg = PolyGraph({0: {1, 2}, 1: {0, 2}, 2: {0, 1}, 3: set()})
        connected = pg.get_connected(3)
        assert set(connected) == {3}

    def test_invalid_root_node(self):
        pg = PolyGraph({0: {1, 2}, 1: {0, 2}, 2: {0, 1}, 3: set()})
        with pytest.raises(KeyError):
            pg.get_connected(99)


class TestCreateCCMap:
    def test_multiple_components_mapping(self):
        adj_list = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4], 4: [3]}
        pg = PolyGraph(adj_list)
        expected = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1}
        assert pg.create_connected_components_map() == expected

    def test_empty_graph_mapping(self):
        pg = PolyGraph(adj_list={})
        assert pg.create_connected_components_map() == {}

    def test_isolated_nodes_mapping(self):
        adj_list = {0: [], 1: [], 2: []}
        pg = PolyGraph(adj_list)
        expected = {0: 0, 1: 1, 2: 2}
        assert pg.create_connected_components_map() == expected
