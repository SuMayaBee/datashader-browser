from __future__ import annotations

import datashader as ds
import numpy as np
import pandas as pd
import pytest

from datashader_browser import ParquetSource, Progress
from datashader_browser.holoviews import datashade


@pytest.fixture
def interactive_source(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    frame = pd.DataFrame(
        {
            "x": np.arange(10, dtype="float64"),
            "y": np.arange(10, dtype="float64"),
            "value": np.arange(10, dtype="float64") + 1,
        }
    )
    path = tmp_path / "interactive.parquet"
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=4
    )
    return ParquetSource(path, batch_size=3)


def test_datashade_returns_dynamic_rgb_and_responds_to_ranges(interactive_source):
    hv = pytest.importorskip("holoviews")
    updates: list[Progress] = []
    dynamic = datashade(
        interactive_source,
        "x",
        "y",
        agg=ds.mean("value"),
        width=100,
        height=80,
        on_progress=updates.append,
    )

    initial = dynamic[()]
    assert isinstance(dynamic, hv.DynamicMap)
    assert isinstance(initial, hv.RGB)
    assert initial.data.attrs["source_rows"] == 10
    assert updates[-1].rows == 10

    dynamic.streams[0].event(x_range=(2.0, 5.0), y_range=(2.0, 5.0))
    zoomed = dynamic[()]
    assert isinstance(zoomed, hv.RGB)
    assert zoomed.data.attrs["x_range"] == (2.0, 5.0)
    assert zoomed.data.attrs["source_rows"] == 4
    assert updates[-1].rows == 4


def test_datashade_handles_an_empty_viewport(interactive_source):
    pytest.importorskip("holoviews")
    dynamic = datashade(
        interactive_source,
        "x",
        "y",
        x_range=(20.0, 30.0),
        y_range=(20.0, 30.0),
        width=20,
        height=20,
    )

    image = dynamic[()]
    assert image.data.attrs["source_rows"] == 0


def test_datashade_validates_source_and_dimensions(interactive_source):
    with pytest.raises(TypeError, match="ParquetSource"):
        datashade(object(), "x", "y")
    with pytest.raises(ValueError, match="greater than zero"):
        datashade(interactive_source, "x", "y", width=0)
