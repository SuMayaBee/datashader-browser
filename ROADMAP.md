# Roadmap

## 0.1: bounded-memory correctness

- Point glyphs over iterables of Pandas chunks.
- Count, sum, mean, min, and max parity with Datashader 0.19.1.
- Progress and cancellation hooks.
- Xeus-Python/JupyterLite environment using emscripten-forge Numba.
- Browser build and deployment CI.

## 0.2: browser data plane

- Local Parquet record-batch source via PyArrow in the Xeus-Python worker.
- Repeatable `ParquetSource` with projected scans, bounds, and batch-local
  viewport predicates.
- HoloViews `DynamicMap`/`RangeXY` adapter for local zoom and pan rerendering.
- DuckDB-Wasm worker for local and remote Parquet.
- Column projection and viewport predicates.
- Transferable buffers between the data worker and Python worker.
- Generation IDs so obsolete pan/zoom jobs are cancelled or ignored.

## 0.3: indexed exploration

- [x] Multi-resolution aggregate pyramid (`build_pyramid`, `TilePyramid`)
  storing reduction state, with every level identical to native Datashader.
- [x] `datashader-browser build-tiles` CLI with threaded binning and writing.
- [x] HoloViews adapter reads only intersecting tiles; raw-scan fallback past
  the deepest level.
- [x] Progressive partial aggregates from `StreamingCanvas` (`on_partial`).
- [ ] Separate files per field so remote viewers download only the fields a
  reduction needs.
- [ ] Lower build memory (external sort or per-band builds) for billion-row inputs.
- [ ] Cache fetched tiles in OPFS across sessions.
- [ ] Spatially partitioned raw Parquet so the deep-zoom fallback reads only
  intersecting partitions.
- [ ] Propose a public incremental aggregation API upstream
  (draft: `docs/upstream-proposal.md`).

## Later

- Lines with one-row partition overlap.
- Categorical aggregation with explicit category budgets.
- Panel application integration outside the JupyterLab shell.
- Optional GPU aggregation experiments after CPU parity and memory behavior are
  characterized.

Polygons, trimesh, quadmesh, Dask, CUDA, and arbitrary row-index reductions are
outside the initial scope.
