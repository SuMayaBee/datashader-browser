from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import datashader as ds
from datashader_browser import Progress, StreamingCanvas


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    size = 20_003
    return pd.DataFrame(
        {
            "x": rng.uniform(-10, 10, size),
            "y": rng.uniform(-5, 5, size),
            "value": rng.normal(20, 3, size),
        }
    )


def chunks(frame: pd.DataFrame, size: int = 997):
    for start in range(0, len(frame), size):
        yield frame.iloc[start : start + size]


@pytest.mark.parametrize(
    "reduction",
    [ds.count(), ds.sum("value"), ds.mean("value"), ds.min("value"), ds.max("value")],
)
def test_streaming_points_matches_datashader(frame, reduction):
    kwargs = dict(plot_width=321, plot_height=177, x_range=(-10, 10), y_range=(-5, 5))
    expected = ds.Canvas(**kwargs).points(frame, "x", "y", agg=reduction)
    actual = StreamingCanvas(**kwargs).points(chunks(frame), "x", "y", agg=reduction)

    xr.testing.assert_equal(actual.drop_attrs(), expected.drop_attrs())
    assert actual.attrs["source_rows"] == len(frame)


def test_progress_is_reported(frame):
    updates: list[Progress] = []
    result = StreamingCanvas(
        plot_width=10,
        plot_height=10,
        x_range=(-10, 10),
        y_range=(-5, 5),
    ).points(chunks(frame, 10_000), "x", "y", on_progress=updates.append)

    assert result.attrs["source_chunks"] == 3
    assert updates[-1] == Progress(chunks=3, rows=len(frame))


def test_empty_source_is_rejected():
    canvas = StreamingCanvas(x_range=(0, 1), y_range=(0, 1))
    with pytest.raises(ValueError, match="at least one"):
        canvas.points([], "x", "y")


def test_schema_change_is_rejected():
    first = pd.DataFrame({"x": [0.1], "y": [0.2]})
    second = pd.DataFrame({"x": [0.3], "y": np.array([1], dtype="int64")})
    canvas = StreamingCanvas(x_range=(0, 1), y_range=(0, 1))
    with pytest.raises(TypeError, match="matching Datashader schemas"):
        canvas.points([first, second], "x", "y")

