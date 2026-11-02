# Draft proposal: a public incremental aggregation API for Datashader

*Status: draft for discussion with Datashader maintainers. Not yet posted.*

## Problem

`Canvas.points` (and the other glyph methods) aggregate a complete
DataFrame in one call. Data that does not fit in memory needs Dask. That is
the right tool on a cluster, but it is heavy or unavailable in several
growing environments:

- WebAssembly runtimes (JupyterLite / Xeus-Python), where Numba and
  llvmlite now work thanks to the emscripten-forge patches discussed in
  holoviz/datashader#1200 and numba/numba#3284, but Dask's scheduler model
  is a poor fit for a single Web Worker with a bounded heap;
- streaming sources (Arrow record batches, Kafka, sensor feeds) that arrive
  incrementally and should update an existing aggregate;
- simple scripts that read a large CSV or Parquet file in chunks and do not
  want a Dask dependency.

`datashader-browser` implements this today by calling private internals:
`datashader.compiler.compile_components`, `Glyph._build_extend` and
`datashader.core._bypixel_sanitise`. That pins the project to one Datashader
minor series. A small public API would remove the coupling and make the
capability available to every Datashader user.

## Proposed API

```python
canvas = ds.Canvas(plot_width=800, plot_height=500, x_range=xr, y_range=yr)

acc = canvas.accumulator("points", x="x", y="y", agg=ds.mean("value"))
for chunk in chunks:            # pandas DataFrames (and possibly Arrow batches)
    acc.update(chunk)
partial = acc.result()          # finalized aggregate so far; state is preserved
acc.merge(other_acc)            # combine accumulators built in parallel
agg = acc.result()
```

Semantics:

- Explicit `x_range` and `y_range` are required (a single pass cannot infer
  them); the error message says so.
- `update` validates that each chunk has the schema of the first.
- `result()` finalizes a copy, so it can be called repeatedly for
  progressive rendering.
- `merge` uses the reduction's existing `combine` functions, the same path
  Dask partitions use today, so parallel accumulation needs no new math.
- Initially points only, CPU only, and reductions that do not use row
  indices. Lines need overlap between chunks and can follow later.

## Implementation sketch

The pieces already exist inside `Canvas.points` / `bypixel`: compile the
reduction once (`compile_components`), build the glyph's `extend` once,
allocate `create(shape)` bases once, call `extend` per chunk and `finalize`
at the end. The accumulator is a thin object that holds those closures and
the bases. `datashader-browser`'s `StreamingCanvas` (~150 lines) is a working
prototype with tests showing results identical to `Canvas.points` for
`count`, `sum`, `mean`, `min` and `max`.

## Why upstream rather than downstream

- No private-API coupling, so no version pinning for downstream projects.
- Benefits native users (chunked CSV/Parquet without Dask), not only the
  browser.
- Gives the browser work a supported foundation once the Numba/llvmlite
  WebAssembly patches land upstream.

## Open questions for maintainers

1. Should `update` accept Arrow record batches directly, avoiding a pandas
   conversion per chunk?
2. Is `merge` worth exposing in the first version, or only `update`/`result`?
3. Naming: `Canvas.accumulator(...)`, `ds.Accumulator(canvas, ...)`, or an
   `incremental=True` mode on existing methods?
