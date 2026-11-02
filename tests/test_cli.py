from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import datashader as ds
from datashader_browser import TilePyramid
from datashader_browser.cli import main


def test_build_tiles_from_parquet_uses_statistics_extent(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"x": rng.uniform(0, 1, 3_000), "y": rng.uniform(5, 6, 3_000),
                          "v": rng.normal(size=3_000)})
    path = tmp_path / "points.parquet"
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path)

    out = tmp_path / "tiles"
    assert main(["build-tiles", str(path), "--x", "x", "--y", "y", "--out", str(out),
                 "--max-level", "1", "--tile-size", "8", "--values", "v",
                 "--batch-size", "500", "--workers", "2"]) == 0

    pyramid = TilePyramid(out)
    assert pyramid.manifest["binned_rows"] == len(frame)
    canvas = ds.Canvas(16, 16, x_range=pyramid.x_range, y_range=pyramid.y_range)
    expected = canvas.points(frame, "x", "y", agg=ds.max("v"))
    np.testing.assert_array_equal(pyramid.aggregate(level=1, agg=ds.max("v")).values,
                                  expected.values)


def test_build_tiles_requires_ranges_for_csv(tmp_path):
    path = tmp_path / "points.csv"
    pd.DataFrame({"x": [0.0], "y": [0.0]}).to_csv(path, index=False)
    with pytest.raises(SystemExit):
        main(["build-tiles", str(path), "--x", "x", "--y", "y", "--out", str(tmp_path / "t")])
