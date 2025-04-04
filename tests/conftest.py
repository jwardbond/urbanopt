import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame for testing."""
    data = {
        "pid": [1, 2, 3],  # Integer PIDs
        "label": ["adu", "bment", "vsplit"],
        "start": ["A", "B", "C"],
        "end": ["X", "Y", "Z"],
        "desc": ["desc1", "desc2", "desc3"],
        "opportunity": [1.0, 2.0, 3.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        "cost_emb": [10, 20, 30],
        "cost_transit": [5, 15, 25],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def mutual_exclusion_gdf():
    """Create a GeoDataFrame with various pathway intersection scenarios for testing."""
    data = {
        "pid": list(range(1, 8)),  # 7 pathways
        "label": [
            "adu",
            "adu",
            "bment",
            "bment",
            "vsplit",
            "hsplit",
            "merge",
        ],
        "start": ["A", "A", "B", "B", "C", "D", "E"],
        "end": ["X", "X", "Y", "Y", "Z", "W", "V"],
        "desc": [f"d{i}" for i in range(1, 8)],
        "opportunity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "geometry": [
            # Group 1: adu pathways (pids 1,2)
            Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),  # int w. 3
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),  # no int.
            # Group 2: bment pathways (pids 3,4)
            Polygon([(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]),  # int w. 1
            Polygon([(6, 6), (7, 6), (7, 7), (6, 7)]),  # no int.
            # Group 3: Additional pathways for multiple pair testing
            Polygon([(8, 8), (9, 8), (9, 9), (8, 9)]),  # vsplit: no int.
            Polygon([(10, 10), (11, 10), (11, 11), (10, 11)]),  # hsplit: no int.
            Polygon([(12, 12), (13, 12), (13, 13), (12, 13)]),  # merge: no int.
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")
