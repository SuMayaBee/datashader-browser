"""Single-pass, bounded-memory Datashader aggregation.

The browser cannot assume that the complete dataset fits in its WebAssembly
heap.  This module keeps one input chunk and Datashader's fixed-size aggregate
state resident at a time.  It intentionally starts with point glyphs: they do
not require overlap between adjacent chunks and therefore provide a sound
foundation for later Arrow and DuckDB-Wasm sources.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import pandas as pd

import datashader as ds
from datashader.compiler import compile_components
from datashader.core import _bypixel_sanitise
from datashader.glyphs import Point


@dataclass(frozen=True, slots=True)
class Progress:
    """Progress reported after an input chunk has been aggregated."""

    chunks: int
    rows: int


ProgressCallback = Callable[[Progress], None]
CancelCallback = Callable[[], bool]


def iter_csv_chunks(
    path: str | Path,
    *,
    chunksize: int = 250_000,
    columns: Iterable[str] | None = None,
    **read_csv_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Yield bounded-size DataFrames from a CSV in the browser filesystem.

    Parameters mirror :func:`pandas.read_csv`. Only requested columns are
    decoded, which is important in memory-constrained WebAssembly runtimes.
    """

    if chunksize <= 0:
        raise ValueError("chunksize must be greater than zero")
    usecols = None if columns is None else list(columns)
    yield from pd.read_csv(
        path,
        chunksize=chunksize,
        usecols=usecols,
        **read_csv_kwargs,
    )


def iter_parquet_batches(
    source: Any,
    *,
    batch_size: int = 250_000,
    columns: Sequence[str] | None = None,
    row_groups: Sequence[int] | None = None,
    use_threads: bool = False,
) -> Iterator[Any]:
    """Yield bounded-size Arrow record batches from a Parquet source.

    ``source`` may be a path or any seekable file object accepted by
    :class:`pyarrow.parquet.ParquetFile`.  In JupyterLite/Xeus-Python this
    includes files uploaded or mounted in the browser filesystem.  PyArrow is
    imported lazily so CSV-only users do not need the optional dependency.

    Arrow batches are intentionally returned without first materializing a
    complete Pandas DataFrame. :meth:`StreamingCanvas.points` converts one
    batch at a time immediately before aggregation.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Parquet streaming requires PyArrow; install "
            "datashader-browser[data]"
        ) from exc

    parquet = pq.ParquetFile(source)
    yield from parquet.iter_batches(
        batch_size=batch_size,
        row_groups=None if row_groups is None else list(row_groups),
        columns=None if columns is None else list(columns),
        use_threads=use_threads,
    )


class StreamingCanvas:
    """A Datashader canvas that consumes Pandas or Arrow chunks.

    Unlike ``datashader.Canvas.points``, the full source is never retained.
    The current implementation requires explicit ranges so aggregation stays
    single-pass. Use source metadata, a cheap bounds query, or a preparatory
    pass to determine those ranges.
    """

    def __init__(
        self,
        *,
        plot_width: int = 600,
        plot_height: int = 600,
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        x_axis_type: str = "linear",
        y_axis_type: str = "linear",
    ) -> None:
        self._canvas = ds.Canvas(
            plot_width=plot_width,
            plot_height=plot_height,
            x_range=x_range,
            y_range=y_range,
            x_axis_type=x_axis_type,
            y_axis_type=y_axis_type,
        )

    @property
    def canvas(self) -> ds.Canvas:
        """The underlying Datashader canvas."""

        return self._canvas

    def points(
        self,
        chunks: Iterable[Any],
        x: str,
        y: str,
        *,
        agg: Any | None = None,
        on_progress: ProgressCallback | None = None,
        cancelled: CancelCallback | None = None,
    ):
        """Aggregate point chunks while retaining only fixed-size state.

        The aggregation uses Datashader's compiler directly, so binning and
        reduction behavior match the native package rather than a separate
        reimplementation.
        """

        reduction = ds.count() if agg is None else agg
        glyph = Point(x, y)
        iterator = iter(chunks)

        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("chunks must contain at least one DataFrame") from exc

        first, dshape = _prepare_chunk(first, glyph, reduction)
        schema = dshape.measure
        glyph.validate(schema)
        reduction.validate(schema)
        self._canvas.validate()

        if reduction.uses_row_index(cuda=False, partitioned=False):
            raise NotImplementedError(
                "row-index reductions are not supported by the streaming points MVP"
            )

        (
            create,
            info,
            append,
            _combine,
            finalize,
            antialias_stage_2,
            antialias_stage_2_funcs,
            _column_names,
        ) = compile_components(
            reduction,
            schema,
            glyph,
            antialias=False,
            cuda=False,
            partitioned=False,
        )

        extend = glyph._build_extend(  # noqa: SLF001 - Datashader extension API
            self._canvas.x_axis.mapper,
            self._canvas.y_axis.mapper,
            info,
            append,
            antialias_stage_2,
            antialias_stage_2_funcs,
        )

        x_range = self._canvas.x_range
        y_range = self._canvas.y_range
        assert x_range is not None and y_range is not None
        self._canvas.validate_ranges(x_range, y_range)

        width = self._canvas.plot_width
        height = self._canvas.plot_height
        x_st = self._canvas.x_axis.compute_scale_and_translate(x_range, width)
        y_st = self._canvas.y_axis.compute_scale_and_translate(y_range, height)
        transform = x_st + y_st
        bounds = x_range + y_range
        bases = create((height, width))

        processed_chunks = 0
        processed_rows = 0

        for chunk in _prepend(first, iterator):
            if cancelled is not None and cancelled():
                raise RuntimeError("aggregation cancelled")
            prepared, chunk_dshape = _prepare_chunk(chunk, glyph, reduction)
            if chunk_dshape.measure != schema:
                raise TypeError(
                    "all chunks must have matching Datashader schemas; "
                    f"expected {schema}, received {chunk_dshape.measure}"
                )
            extend(bases, prepared, transform, bounds)
            processed_chunks += 1
            processed_rows += len(prepared)
            if on_progress is not None:
                on_progress(Progress(processed_chunks, processed_rows))

        x_axis = self._canvas.x_axis.compute_index(x_st, width)
        y_axis = self._canvas.y_axis.compute_index(y_st, height)
        return finalize(
            bases,
            cuda=False,
            coords={glyph.x_label: x_axis, glyph.y_label: y_axis},
            dims=[glyph.y_label, glyph.x_label],
            attrs={
                "x_range": x_range,
                "y_range": y_range,
                "source_chunks": processed_chunks,
                "source_rows": processed_rows,
            },
        )


def _prepare_chunk(chunk: Any, glyph: Point, reduction: Any):
    if not isinstance(chunk, pd.DataFrame):
        to_pandas = getattr(chunk, "to_pandas", None)
        if to_pandas is None or not callable(to_pandas):
            raise TypeError(
                "each chunk must be a pandas DataFrame or Arrow-compatible "
                f"record batch, received {type(chunk)!r}"
            )
        # PyArrow's direct Arrow-to-Pandas bridge can stall in the current
        # Emscripten runtime.  Building the bounded DataFrame from Python
        # column values avoids that bridge while preserving the batch limit.
        if sys.platform == "emscripten" and callable(getattr(chunk, "to_pydict", None)):
            chunk = pd.DataFrame(chunk.to_pydict())
        else:
            chunk = to_pandas()
        if not isinstance(chunk, pd.DataFrame):
            raise TypeError(
                "chunk.to_pandas() must return a pandas DataFrame, received "
                f"{type(chunk)!r}"
            )
    return _bypixel_sanitise(chunk, glyph, reduction)


def _prepend(first: Any, rest: Iterator[Any]) -> Iterator[Any]:
    yield first
    yield from rest
