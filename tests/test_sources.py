from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from datashader_browser import ParquetSource


@pytest.fixture
def parquet_path(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    frame = pd.DataFrame(
        {
            "x": np.arange(10, dtype="float64"),
            "y": np.arange(10, dtype="float64") * 2,
            "value": np.arange(10, dtype="float64") + 100,
        }
    )
    path = tmp_path / "points.parquet"
    pq.write_table(
        pa.Table.from_pandas(frame, preserve_index=False), path, row_group_size=4
    )
    return path


def test_parquet_source_is_repeatable_and_projected(parquet_path):
    source = ParquetSource(parquet_path, batch_size=3)

    first = list(source.scan(columns=["x", "y"]))
    second = list(source.scan(columns=["x", "y"]))

    assert source.num_rows == 10
    assert source.schema.names == ["x", "y", "value"]
    assert sum(batch.num_rows for batch in first) == 10
    assert sum(batch.num_rows for batch in second) == 10
    assert all(batch.schema.names == ["x", "y"] for batch in first)


def test_parquet_source_reports_and_caches_bounds(parquet_path):
    source = ParquetSource(parquet_path)

    assert source.bounds("x", "y") == ((-0.0, 9.0), (-0.0, 18.0))
    assert source.bounds("x", "y") == ((-0.0, 9.0), (-0.0, 18.0))


def test_parquet_source_filters_each_bounded_batch(parquet_path):
    source = ParquetSource(parquet_path, batch_size=3)
    batches = list(
        source.scan(
            columns=["value"],
            x="x",
            y="y",
            x_range=(2.0, 6.0),
            y_range=(4.0, 12.0),
        )
    )

    assert sum(batch.num_rows for batch in batches) == 5
    assert all(batch.num_rows <= 3 for batch in batches)
    assert batches[0].schema.names == ["value", "x", "y"]


def test_parquet_source_preserves_schema_for_empty_viewport(parquet_path):
    source = ParquetSource(parquet_path, batch_size=3)
    batches = list(
        source.scan(
            columns=["x", "y"],
            x="x",
            y="y",
            x_range=(20.0, 30.0),
            y_range=(20.0, 30.0),
        )
    )

    assert len(batches) == 1
    assert batches[0].num_rows == 0
    assert batches[0].schema.names == ["x", "y"]


def test_parquet_source_validates_filter_arguments(parquet_path):
    source = ParquetSource(parquet_path)
    with pytest.raises(ValueError, match="provided together"):
        list(source.scan(x_range=(0, 1)))
    with pytest.raises(ValueError, match="minimum"):
        list(source.scan(x="x", x_range=(1, 1)))
