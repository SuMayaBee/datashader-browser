from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import datashader as ds
from datashader_browser import TilePyramid, build_pyramid

X_RANGE = (-10.0, 10.0)
Y_RANGE = (-5.0, 5.0)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    size = 30_011
    x = np.concatenate([rng.normal(0, 3, size), [10.0, -10.0, 10.0, 25.0, np.nan]])
    y = np.concatenate([rng.normal(0, 1.5, size), [5.0, -5.0, 0.0, 0.0, 1.0]])
    value = rng.normal(20, 3, len(x))
    value[::17] = np.nan
    return pd.DataFrame(
        {"x": x.astype(np.float32), "y": y.astype(np.float32), "value": value}
    )


def chunks(frame: pd.DataFrame, size: int = 4_001):
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


@pytest.fixture(scope="module")
def pyramid(frame, tmp_path_factory) -> TilePyramid:
    out = tmp_path_factory.mktemp("pyramid")
    return build_pyramid(
        chunks(frame),
        out,
        "x",
        "y",
        x_range=X_RANGE,
        y_range=Y_RANGE,
        max_level=3,
        tile_size=16,
        values=["value"],
    )


def native(frame, level, agg, tile_size=16):
    n = tile_size * 2**level
    canvas = ds.Canvas(plot_width=n, plot_height=n, x_range=X_RANGE, y_range=Y_RANGE)
    return canvas.points(frame, "x", "y", agg=agg)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
@pytest.mark.parametrize("agg", [ds.count(), ds.count("value"), ds.min("value"), ds.max("value")])
def test_every_level_is_identical_to_datashader(frame, pyramid, level, agg):
    actual = pyramid.aggregate(level=level, agg=agg)
    expected = native(frame, level, agg)
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.values, expected.values)
    np.testing.assert_allclose(actual["x"].values, expected["x"].values)
    np.testing.assert_allclose(actual["y"].values, expected["y"].values)


@pytest.mark.parametrize("level", [0, 3])
@pytest.mark.parametrize("agg", [ds.sum("value"), ds.mean("value")])
def test_float_reductions_match_datashader(frame, pyramid, level, agg):
    actual = pyramid.aggregate(level=level, agg=agg)
    expected = native(frame, level, agg)
    np.testing.assert_allclose(actual.values, expected.values, rtol=1e-12, equal_nan=True)


def test_viewport_is_a_slice_of_the_level(frame, pyramid):
    view = pyramid.aggregate((-2.3, 4.1), (-1.2, 0.7), width=40, height=10)
    level = view.attrs["pyramid_level"]
    full = native(frame, level, ds.count())
    expected = full.sel(x=view["x"], y=view["y"])
    xr.testing.assert_equal(view.drop_attrs().rename(None), expected.drop_attrs().rename(None))
    assert view.attrs["x_range"][0] <= -2.3 and view.attrs["x_range"][1] >= 4.1
    assert view.shape[1] >= 40 and view.shape[0] >= 10


def test_level_choice_and_under_resolution(pyramid):
    assert pyramid.choose_level(X_RANGE, Y_RANGE, 16, 16) == 0
    assert pyramid.choose_level(X_RANGE, Y_RANGE, 100, 100) == 3
    deep = pyramid.aggregate((0, 0.5), (0, 0.5), width=200, height=200)
    assert deep.attrs["pyramid_level"] == 3
    assert deep.attrs["under_resolved"]


def test_only_intersecting_tiles_are_read(frame, tmp_path):
    pyramid = build_pyramid(
        chunks(frame), tmp_path, "x", "y",
        x_range=X_RANGE, y_range=Y_RANGE, max_level=3, tile_size=16,
    )
    pyramid.aggregate((0, 1), (0, 0.5), level=3)
    assert 1 <= pyramid.tiles_read <= 4


def test_threaded_build_matches_serial(frame, pyramid, tmp_path):
    threaded = build_pyramid(
        chunks(frame, 997), tmp_path, "x", "y",
        x_range=X_RANGE, y_range=Y_RANGE, max_level=3, tile_size=16,
        values=["value"], workers=4, merge_threshold=1_000,
    )
    for level in range(4):
        for agg in (ds.count(), ds.max("value")):
            np.testing.assert_array_equal(
                threaded.aggregate(level=level, agg=agg).values,
                pyramid.aggregate(level=level, agg=agg).values,
            )


def test_manifest_records_rows(frame, pyramid):
    manifest = json.loads((pyramid_path(pyramid) / "manifest.json").read_text())
    assert manifest["source_rows"] == len(frame)
    # Out-of-range and NaN coordinates are dropped, exactly as in Datashader.
    assert manifest["binned_rows"] == int(native(frame, 0, ds.count()).sum())


def test_missing_value_column_is_explained(pyramid):
    with pytest.raises(ValueError, match="rebuild"):
        pyramid.aggregate(agg=ds.sum("other"))


def test_arrow_batches_are_accepted(frame, tmp_path):
    pa = pytest.importorskip("pyarrow")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pyramid = build_pyramid(
        table.to_batches(max_chunksize=5_000), tmp_path, "x", "y",
        x_range=X_RANGE, y_range=Y_RANGE, max_level=2, tile_size=16,
    )
    np.testing.assert_array_equal(
        pyramid.aggregate(level=2).values, native(frame, 2, ds.count()).values
    )


def pyramid_path(pyramid: TilePyramid):
    from pathlib import Path

    return Path(pyramid.location)
