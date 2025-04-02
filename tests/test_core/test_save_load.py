import json
import tempfile
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import Point

from urbanopt import PathwayOptimizer


def test_save_creates_files(sample_gdf: gpd.GeoDataFrame):
    """Test that save creates both required files."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_optimizer"
        optimizer.save(path)

        # Should create both files
        assert path.with_suffix(".geoparquet").exists()
        assert path.with_suffix(".json").exists()


def test_load_restores_data(sample_gdf: gpd.GeoDataFrame):
    """Test that load restores basic data correctly."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_optimizer"
        optimizer.save(path)
        loaded = PathwayOptimizer.load(path)

        # Should restore data correctly
        assert loaded.data.equals(optimizer.data)
        assert loaded.pids == optimizer.pids
        assert loaded.cost_columns == optimizer.cost_columns


def test_load_restores_objective(sample_gdf: gpd.GeoDataFrame):
    """Test that load restores objective function correctly."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()
    optimizer.set_objective({"cost_emb": 1.0, "cost_transit": 0.5})

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_optimizer"
        optimizer.save(path)
        loaded = PathwayOptimizer.load(path)

        # Should restore objective correctly
        loaded_obj = loaded.model.getObjective()
        orig_obj = optimizer.model.getObjective()
        assert loaded_obj.size() == orig_obj.size()
        for i in range(orig_obj.size()):
            assert abs(loaded_obj.getCoeff(i) - orig_obj.getCoeff(i)) < 1e-6


def test_load_restores_constraints(sample_gdf: gpd.GeoDataFrame):
    """Test that load restores constraints correctly."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Add constraints with different tags
    optimizer.add_max_opportunity(5.0, tag="test_tag")
    optimizer.add_min_opportunity(2.0, tag="test_tag")
    optimizer.add_max_opportunity(3.0, tag="other_tag")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_optimizer"
        optimizer.save(path)
        loaded = PathwayOptimizer.load(path)

        # Should restore constraints correctly
        assert set(loaded._constraints.keys()) == set(optimizer._constraints.keys())
        for tag in optimizer._constraints:
            assert len(loaded._constraints[tag]) == len(optimizer._constraints[tag])


def test_save_constraint_sense(sample_gdf: gpd.GeoDataFrame):
    """Test that constraint senses are correctly saved and loaded."""
    optimizer = PathwayOptimizer(sample_gdf)
    optimizer.build_variables()

    # Add constraints with different senses
    optimizer.add_max_opportunity(5.0, tag="le_test")  # Uses "<="
    optimizer.add_min_opportunity(2.0, tag="ge_test")  # Uses ">="

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_optimizer"
        optimizer.save(path)

        # Read and check the JSON file directly
        json_path = path.with_suffix(".json")
        with json_path.open() as f:
            config = json.load(f)

        # Should save correct constraint senses
        assert config["constraints"]["le_test"][0]["sense"] == "<=", (
            "Max constraint should use '<=' sense"
        )
        assert config["constraints"]["ge_test"][0]["sense"] == ">=", (
            "Min constraint should use '>=' sense"
        )

        # Load and verify constraints work
        loaded = PathwayOptimizer.load(path)
        assert len(loaded._constraints["le_test"]) == 1
        assert len(loaded._constraints["ge_test"]) == 1


def test_save_without_build():
    """Test that saving without building variables raises error."""
    data = gpd.GeoDataFrame(
        {
            "pid": [1],
            "start": ["A"],
            "end": ["X"],
            "desc": ["test"],
            "opportunity": [1.0],
            "geometry": [Point(0, 0)],
            "cost_emb": [10],
            "cost_transit": [5],
        }
    )
    optimizer = PathwayOptimizer(data)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test_optimizer"
        with pytest.raises(RuntimeError, match="Model has not been built yet"):
            optimizer.save(path)


def test_load_missing_files():
    """Test that loading with missing files raises appropriate errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nonexistent"

        # Should raise FileNotFoundError for missing geoparquet
        with pytest.raises(FileNotFoundError, match="GeoParquet file not found"):
            PathwayOptimizer.load(path)

        # Create geoparquet but no json
        data = gpd.GeoDataFrame(
            {
                "pid": [1],
                "start": ["A"],
                "end": ["X"],
                "desc": ["test"],
                "opportunity": [1.0],
                "geometry": [Point(0, 0)],
                "cost_emb": [10],
                "cost_transit": [5],
            },
        )
        data.to_parquet(path.with_suffix(".geoparquet"))

        # Should raise FileNotFoundError for missing json
        with pytest.raises(
            FileNotFoundError, match="JSON configuration file not found"
        ):
            PathwayOptimizer.load(path)
