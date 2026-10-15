# Datashader Browser Lab

Start with `streaming-parquet.ipynb` to create, stream, and render a Parquet
dataset entirely in this browser tab. `interactive-parquet.ipynb` connects a
repeatable Parquet source to HoloViews zoom and pan events. `streaming-points.ipynb`
is the smaller in-memory-generated example. `nyc-taxi-browser.ipynb` contains
the full 11.8-million-row HoloViz NYC Taxi workflow and instructions for
supplying its 385 MB source file.

The Xeus-Python kernel and all Python packages are WebAssembly assets served by
the static site; no Python server is involved.
