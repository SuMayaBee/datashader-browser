# Architecture

## Objective

Preserve Datashader's Python API and aggregation semantics while running in a
browser with bounded memory and without a Python server.

The browser is a constrained execution environment. A WebAssembly process has
a finite heap, CPU-heavy Python can block its worker, remote files are subject
to CORS, and scanning a billion records for every pan event is not interactive.
The design therefore separates rendering from data access.

## Execution model

```text
Browser UI / Bokeh / Panel
          |
          | viewport + reduction request
          v
Xeus-Python Web Worker
  StreamingCanvas -> Datashader compiler -> aggregate -> shade
          ^
          | bounded chunks / Arrow IPC
          |
Data worker (planned)
  DuckDB-Wasm -> Parquet / CSV / OPFS / HTTP range requests
```

The current implementation covers the Xeus-Python aggregation box. The data
worker is the next milestone.

## Why streaming changes the memory equation

For an input of `N` rows, chunk size `C`, canvas width `W`, and height `H`, peak
working data is approximately:

```text
Python/Wasm runtime + decoded columns for C rows + reduction bases for W * H bins
```

It is not proportional to `N`, provided the source can be streamed. A
1000-by-1000 count aggregate is about 4 MB before metadata. A mean typically
needs both a floating-point sum and a count, roughly 12 MB for the same canvas.
Input chunks and temporary DataFrames are normally the larger controllable
cost.

Categorical aggregation multiplies aggregate memory by the number of
categories. The API will require an explicit category budget rather than
allowing an unbounded third dimension.

## Current aggregation path

`StreamingCanvas.points`:

1. Reads the first chunk to establish a Datashader schema.
2. Compiles the requested glyph and reduction once using Datashader's compiler.
3. Allocates aggregate bases once for the fixed canvas.
4. Extends those bases with one DataFrame chunk at a time.
5. Finalizes an xarray aggregate only after the stream ends.

This is preferable to combining already-finalized images: reductions such as
mean require multiple associative bases, and image combination would lose that
information.

## Large-source modes

### Memory mode

Use ordinary Datashader for a DataFrame already small enough for the browser.
This is simplest and preserves the complete API.

### Stream mode

Use `StreamingCanvas` with CSV chunks or Arrow record batches. This permits a
large sequential scan while bounding peak data memory. It is suitable for a
one-off render or datasets that can be scanned quickly.

### Pruned mode

DuckDB-Wasm will query only `x`, `y`, and reduction columns and apply the
viewport predicate before returning Arrow record batches. Parquet row-group
statistics help only when the data layout makes the predicate selective;
spatially sorted or partitioned data is therefore strongly preferred.

### Indexed/tiled mode

For repeated pan and zoom over very large sources, the dataset must be divided
by space and optionally by zoom level. A small manifest maps canvas bounds to
Parquet partitions or precomputed aggregate tiles. The browser fetches only
intersecting partitions and caches recent data in OPFS.

No client-only implementation can make a remote billion-row unsorted CSV
interactive: every viewport would otherwise require downloading and scanning
the entire file.

## Responsiveness and cancellation

Both Xeus-Python and DuckDB-Wasm run in workers. Each viewport request receives
a monotonically increasing generation ID. When a new request arrives, old
results are discarded and cancellable work is stopped. Rendering can first use
a coarse level or a row sample, followed by the full aggregate.

## Compatibility policy

Datashader's compiled internals are currently needed to retain reduction state
across chunks. This project pins Datashader to the 0.19 series and tests every
supported reduction against the normal `Canvas.points` result. A future
upstream public streaming accumulator would remove this dependency on private
interfaces.

