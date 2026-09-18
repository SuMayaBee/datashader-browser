# datashader-browser

Run Datashader-powered visualizations entirely in a browser with Python and
WebAssembly—without assuming that the complete dataset fits in browser memory.

This is an early, independent project. It is not yet an official HoloViz
project and is not affiliated with or endorsed by the Datashader maintainers.

[Try the live browser demo](https://sumayabee.github.io/datashader-browser/).
Open `streaming-points.ipynb` and run all cells.

## What works now

The first vertical slice is a bounded-memory point aggregator:

```python
import datashader as ds
import datashader.transfer_functions as tf

from datashader_browser import StreamingCanvas, iter_csv_chunks

source = iter_csv_chunks(
    "points.csv",
    chunksize=250_000,
    columns=["longitude", "latitude", "value"],
)

canvas = StreamingCanvas(
    plot_width=1000,
    plot_height=600,
    x_range=(-180, 180),
    y_range=(-90, 90),
)
aggregate = canvas.points(source, "longitude", "latitude", agg=ds.mean("value"))
image = tf.shade(aggregate)
```

Only one input chunk and Datashader's fixed-size aggregate buffers are live at
once. The implementation calls Datashader's own compiler, so point binning and
reductions match native Datashader instead of being reimplemented.

Currently verified reductions are `count`, `sum`, `mean`, `min`, and `max`.
Explicit `x_range` and `y_range` values are required to preserve a true
single-pass pipeline.

## Browser runtime

The initial deployment target is JupyterLite with the Xeus-Python WebAssembly
kernel. Its environment is described in [`environment.yml`](environment.yml)
and uses the experimental Numba and llvmlite packages from emscripten-forge.
The kernel executes in a Web Worker, keeping aggregation off the browser UI
thread. The first visit downloads a sizeable scientific-Python environment, so
the cold kernel start can take roughly a minute; subsequent use benefits from
the browser cache.

```bash
uv sync --extra test
uv run pytest

# Build the browser site (requires the build dependencies below).
uv pip install -r requirements-build.txt
uv run jupyter lite build \
  --XeusAddon.environment_file=environment.yml \
  --contents content \
  --output-dir dist
```

## Large-data strategy

“In the browser” does not mean that an unlimited dataset can be copied into a
Pandas DataFrame. The project uses three execution levels:

1. **Stream:** scan CSV or Arrow batches through a bounded-memory Datashader
   accumulator. Total row count affects time, not peak aggregate memory.
2. **Prune:** use Parquet row groups, spatial partitions, and DuckDB-Wasm to
   read only the viewport and required columns.
3. **Tile:** for billion-row interactive datasets, fetch spatial partitions or
   precomputed multiresolution aggregates instead of rescanning the source on
   every zoom.

See [Architecture](docs/architecture.md), [Research](docs/research.md), and the
[roadmap](ROADMAP.md) for the engineering decisions and current limitations.

## Status

This repository is an alpha-stage research implementation. The WebAssembly
Numba/llvmlite stack works in current demonstrations but its patches have not
yet been fully upstreamed. Production use should wait for reproducible browser
builds, compatibility CI, and measured browser memory limits.

## License

BSD-3-Clause. Datashader is a separate BSD-3-Clause project owned by its
respective copyright holders and contributors.
