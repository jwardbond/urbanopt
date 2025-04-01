import geopandas as gpd
import pytest
from shapely.geometry import Point, Polygon


@pytest.fixture
def sample_gdf():
    """Create a sample GeoDataFrame for testing."""
    data = {
        "pid": [1, 2, 3],  # Integer PIDs
        "start": ["A", "B", "C"],
        "end": ["X", "Y", "Z"],
        "desc": ["desc1", "desc2", "desc3"],
        "opportunity": [1.0, 2.0, 3.0],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")


@pytest.fixture
def mutual_exclusion_gdf():
    """Create a GeoDataFrame with various pathway intersection scenarios for testing."""
    data = {
        "pid": list(range(1, 8)),  # 7 pathways
        "start": ["A", "A", "B", "B", "C", "D", "E"],
        "end": ["X", "X", "Y", "Y", "Z", "W", "V"],
        "desc": [f"d{i}" for i in range(1, 8)],
        "opportunity": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "geometry": [
            # Group 1: A→X pathways (pids 1,2)
            Polygon(
                [(0, 0), (1, 0), (1, 1), (0, 1)]
            ),  # pid 1: intersects with pid 3 only
            Polygon([(4, 4), (5, 4), (5, 5), (4, 5)]),  # pid 2: no intersections
            # Group 2: B→Y pathways (pids 3,4)
            Polygon(
                [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
            ),  # pid 3: intersects with pid 1 only
            Polygon([(6, 6), (7, 6), (7, 7), (6, 7)]),  # pid 4: no intersections
            # Group 3: Additional pathways for multiple pair testing
            Polygon([(8, 8), (9, 8), (9, 9), (8, 9)]),  # pid 5 (C→Z): no intersections
            Polygon(
                [(10, 10), (11, 10), (11, 11), (10, 11)]
            ),  # pid 6 (D→W): no intersections
            Polygon(
                [(12, 12), (13, 12), (13, 13), (12, 13)]
            ),  # pid 7 (E→V): completely isolated
        ],
    }
    return gpd.GeoDataFrame(data, crs="EPSG:4326")
