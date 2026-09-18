# Engineering research

Research performed September 2026. Only primary project documentation,
repositories, and papers are used below.

## Datashader and Numba in WebAssembly

- [Datashader issue #1200](https://github.com/holoviz/datashader/issues/1200)
  tracks JupyterLite/Panelite support. In August 2026, maintainers confirmed a
  Datashader demonstration running in JupyterLite.
- [Numba issue #3284](https://github.com/numba/numba/issues/3284) contains the
  active WebAssembly JIT work. The current implementation relies on patch sets
  that still need upstream review.
- [emscripten-forge's Numba recipe](https://github.com/emscripten-forge/recipes/tree/main/recipes/recipes_emscripten/numba)
  currently applies ten WASM-specific patches, including execution-engine,
  ABI, persistent-cache, and parallel-pipeline changes.
- [jupyterlite-xeus environments](https://jupyterlite-xeus.readthedocs.io/en/latest/environment.html)
  can preinstall emscripten-forge packages into a static JupyterLite site. The
  Xeus kernel runs in a Web Worker.

Conclusion: browser Datashader is feasible now as an experimental Xeus-Python
deployment, but shipping must pin the complete WASM environment until Numba
and llvmlite support is upstream.

## DuckDB-Wasm

- [DuckDB-Wasm overview](https://duckdb.org/docs/stable/clients/wasm/overview)
  documents an in-browser analytical database with a 4 GB wasm32 ceiling and
  lower browser-specific practical limits.
- [Data ingestion](https://duckdb.org/docs/current/clients/wasm/data_ingestion)
  supports browser file handles, remote Parquet, Arrow, CSV, and JSON.
- [Streaming queries](https://duckdb.org/docs/current/clients/wasm/query)
  expose Arrow record batches instead of materializing a complete result.
- [The DuckDB-Wasm VLDB paper](https://duckdb.org/pdf/VLDB2022-kohn-duckdb-wasm.pdf)
  describes worker execution, Arrow exchange, and HTTP range reads.

Adopted ideas: worker isolation, Arrow batches, column projection, lazy Parquet
reads, and OPFS caching. DuckDB is a data-access layer; Datashader remains the
rasterization layer.

## Mosaic

- [Mosaic](https://idl.uw.edu/mosaic/) supports interactive exploration of
  millions or billions of records by pushing view-dependent queries to DuckDB,
  including DuckDB-Wasm in the browser.
- [Mosaic's architecture paper](https://idl.cs.washington.edu/files/2024-Mosaic-TVCG.pdf)
  uses a coordinator to consolidate and optimize requests from linked views.

Adopted ideas: describe each view as a query, centralize viewport requests,
cancel stale queries, and support both local-WASM and future server connectors.

## Perspective

- [Perspective](https://github.com/perspective-dev/perspective) uses a
  WebAssembly query engine, Arrow, asynchronous clients, and worker-hosted
  computation for large or streaming data.

Adopted ideas: keep the main thread free, communicate using columnar buffers,
and request only windowed or aggregated results rather than serializing the
whole dataset through the UI.

## deck.gl

- [deck.gl aggregation layers](https://deck.gl/docs/api-reference/aggregation-layers/overview)
  demonstrate browser GPU aggregation for grids, hexagons, contours, and
  heatmaps. Its documentation notes 32-bit precision differences and that GPU
  aggregation does not expose all underlying points.

Conclusion: GPU aggregation is a valuable later backend, but it should not be
the correctness baseline for a Datashader-compatible project.

## Engineering decision

The project is Python-first because compatibility with Datashader and HoloViz
is the primary requirement. Browser-side JavaScript remains necessary for file
handles, workers, transferable buffers, DuckDB-Wasm, and deployment plumbing,
but it is an adapter layer rather than a second rasterization implementation.

