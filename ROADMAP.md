# Roadmap

## 0.1: bounded-memory correctness

- Point glyphs over iterables of Pandas chunks.
- Count, sum, mean, min, and max parity with Datashader 0.19.1.
- Progress and cancellation hooks.
- Xeus-Python/JupyterLite environment using emscripten-forge Numba.
- Browser build and deployment CI.

## 0.2: browser data plane

- Arrow IPC record-batch source.
- DuckDB-Wasm worker for local and remote Parquet.
- Column projection and viewport predicates.
- Transferable buffers between the data worker and Python worker.
- Generation IDs so obsolete pan/zoom jobs are cancelled or ignored.

## 0.3: indexed exploration

- Spatially partitioned Parquet manifest format.
- Partition pruning by canvas bounds.
- Cache recent partitions in OPFS.
- Progressive low-resolution then full-resolution rendering.

## Later

- Lines with one-row partition overlap.
- Categorical aggregation with explicit category budgets.
- Panel/Bokeh integration outside the JupyterLab shell.
- Optional GPU aggregation experiments after CPU parity and memory behavior are
  characterized.

Polygons, trimesh, quadmesh, Dask, CUDA, and arbitrary row-index reductions are
outside the initial scope.

