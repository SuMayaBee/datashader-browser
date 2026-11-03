"""Multi-resolution aggregate pyramids for instant zoom over large point data.

A pyramid stores Datashader *reduction state* (counts, sums, minima, maxima),
not shaded images.  It is built once with ordinary Python and served as small
static files.  A browser client fetches only the tiles that intersect the
viewport, merges them, and shades the result locally, so zooming no longer
rescans the source data.

Exactness
---------
Level ``z`` covers the full extent with a ``tile_size * 2**z`` square grid.
Points are binned with Datashader's own scale/translate parameters and its
truncate-then-clamp rule.  Doubling the grid size multiplies every floating
point term of ``x * s + t`` by exactly two, so pooling 2x2 bins from level
``z + 1`` reproduces the bins Datashader computes at level ``z``.  Every level
is therefore identical to ``ds.Canvas(plot_width=n, plot_height=n,
x_range=extent_x, y_range=extent_y).points(...)`` with ``n = tile_size *
2**z``; a viewport is a slice of that aggregate.  Counts, minima and maxima are
bit-identical.  Sums and means agree up to floating-point summation order.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

import datashader as ds

from .streaming import Progress

Range = tuple[float, float]

FORMAT = "datashader-browser-pyramid"
FORMAT_VERSION = 1
COUNT = "count"

_ADD = np.add
_MIN = np.fmin
_MAX = np.fmax


def _field_ops(values: Sequence[str]) -> dict[str, Any]:
    ops: dict[str, Any] = {COUNT: _ADD}
    for column in values:
        ops[f"{column}:count"] = _ADD
        ops[f"{column}:sum"] = _ADD
        ops[f"{column}:min"] = _MIN
        ops[f"{column}:max"] = _MAX
    return ops


def _empty_value(field: str) -> Any:
    return 0 if field.endswith("count") else np.nan


def _field_dtype(field: str) -> Any:
    return np.uint32 if field.endswith("count") else np.float64


def _scale_translate(extent: Range, n: int) -> tuple[float, float]:
    # Same arithmetic as datashader.core.Axis.compute_scale_and_translate for
    # a linear axis, so bin assignment is reproduced exactly.
    start, end = float(extent[0]), float(extent[1])
    s = n / (end - start)
    return s, -start * s


@dataclass(slots=True)
class _Bins:
    """Sparse reduction state: sorted unique bin keys plus one array per field."""

    keys: np.ndarray
    fields: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.keys)


def _reduce(keys: np.ndarray, fields: dict[str, np.ndarray], ops: dict[str, Any]) -> _Bins:
    if len(keys) == 0:
        return _Bins(keys.astype(np.int64), {k: v[:0] for k, v in fields.items()})
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
    reduced = {
        name: ops[name].reduceat(values[order], starts)
        for name, values in fields.items()
    }
    return _Bins(sorted_keys[starts], reduced)


def _concat_reduce(parts: Sequence[_Bins], ops: dict[str, Any]) -> _Bins:
    if len(parts) == 1:
        return parts[0]
    keys = np.concatenate([part.keys for part in parts])
    fields = {
        name: np.concatenate([part.fields[name] for part in parts]) for name in ops
    }
    return _reduce(keys, fields, ops)


def _column(chunk: Any, name: str) -> np.ndarray:
    if isinstance(chunk, pd.DataFrame):
        return chunk[name].to_numpy()
    column = chunk.column(name) if hasattr(chunk, "column") else chunk[name]
    return column.to_numpy(zero_copy_only=False)


def _bin_chunk(
    chunk: Any,
    *,
    x: str,
    y: str,
    values: Sequence[str],
    x_range: Range,
    y_range: Range,
    n: int,
    ops: dict[str, Any],
) -> tuple[_Bins, int]:
    xs = np.asarray(_column(chunk, x), dtype=np.float64)
    ys = np.asarray(_column(chunk, y), dtype=np.float64)
    # NaN coordinates fail every comparison and are dropped, as in Datashader.
    mask = (x_range[0] <= xs) & (xs <= x_range[1])
    mask &= (y_range[0] <= ys) & (ys <= y_range[1])
    sx, tx = _scale_translate(x_range, n)
    sy, ty = _scale_translate(y_range, n)
    xi = np.trunc(xs[mask] * sx + tx).astype(np.int64)
    yi = np.trunc(ys[mask] * sy + ty).astype(np.int64)
    np.minimum(xi, n - 1, out=xi)
    np.minimum(yi, n - 1, out=yi)
    keys = yi * n + xi

    fields: dict[str, np.ndarray] = {COUNT: np.ones(len(keys), dtype=np.int64)}
    for column in values:
        v = np.asarray(_column(chunk, column), dtype=np.float64)[mask]
        valid = ~np.isnan(v)
        fields[f"{column}:count"] = valid.astype(np.int64)
        fields[f"{column}:sum"] = np.where(valid, v, 0.0)
        fields[f"{column}:min"] = v
        fields[f"{column}:max"] = v
    return _reduce(keys, fields, ops), len(xs)


def _pool(bins: _Bins, n: int, ops: dict[str, Any]) -> _Bins:
    yi, xi = np.divmod(bins.keys, n)
    parent = (yi >> 1) * (n // 2) + (xi >> 1)
    return _reduce(parent, bins.fields, ops)


def build_pyramid(
    chunks: Iterable[Any],
    out_dir: str | Path,
    x: str,
    y: str,
    *,
    x_range: Range,
    y_range: Range,
    max_level: int = 6,
    tile_size: int = 256,
    values: Sequence[str] = (),
    workers: int = 1,
    merge_threshold: int = 8_000_000,
    on_progress: Callable[[Progress], None] | None = None,
) -> "TilePyramid":
    """Aggregate a chunked point source into a static tile pyramid.

    ``chunks`` is any iterable of Pandas DataFrames or Arrow record batches,
    such as :func:`iter_parquet_batches`.  Peak memory is proportional to the
    number of non-empty bins at ``max_level`` plus one chunk per worker, not
    to the number of rows.  ``values`` lists numeric columns for which sum,
    mean, min and max (and non-null counts) are stored in addition to the
    point count.  ``workers > 1`` bins chunks on a thread pool (native Python
    only; NumPy releases the GIL for the heavy operations).
    """

    if max_level < 0 or max_level > 14:
        raise ValueError("max_level must be between 0 and 14")
    if tile_size <= 0 or tile_size & (tile_size - 1):
        raise ValueError("tile_size must be a positive power of two")
    x_range = (float(x_range[0]), float(x_range[1]))
    y_range = (float(y_range[0]), float(y_range[1]))
    for name, extent in (("x_range", x_range), ("y_range", y_range)):
        if not all(math.isfinite(v) for v in extent) or extent[0] >= extent[1]:
            raise ValueError(f"{name} must be two finite, increasing values")

    values = list(dict.fromkeys(values))
    if x in values or y in values:
        raise ValueError("coordinate columns cannot also be value columns")
    ops = _field_ops(values)
    n = tile_size * 2**max_level
    bin_kwargs = dict(
        x=x, y=y, values=values, x_range=x_range, y_range=y_range, n=n, ops=ops
    )

    buffered: list[_Bins] = []
    buffered_size = 0
    progress = [0, 0]

    def collect(result: tuple[_Bins, int]) -> None:
        nonlocal buffered, buffered_size
        part, rows = result
        buffered.append(part)
        buffered_size += len(part)
        if buffered_size > merge_threshold and len(buffered) > 1:
            merged = _concat_reduce(buffered, ops)
            buffered, buffered_size = [merged], len(merged)
        progress[0] += 1
        progress[1] += rows
        if on_progress is not None:
            on_progress(Progress(progress[0], progress[1]))

    if workers <= 1:
        for chunk in chunks:
            collect(_bin_chunk(chunk, **bin_kwargs))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = []
            for chunk in chunks:
                pending.append(pool.submit(_bin_chunk, chunk, **bin_kwargs))
                # Bound in-flight chunks so memory stays proportional to workers.
                while len(pending) >= 2 * workers:
                    collect(pending.pop(0).result())
            for future in pending:
                collect(future.result())

    if not buffered:
        raise ValueError("chunks must contain at least one chunk")
    level_bins = _concat_reduce(buffered, ops)
    binned_rows = int(level_bins.fields[COUNT].sum())

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    levels: dict[str, list[list[int]]] = {}
    writer = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        for level in range(max_level, -1, -1):
            size = tile_size * 2**level
            levels[str(level)] = _write_level(
                out, level, level_bins, size, tile_size, writer
            )
            if level:
                level_bins = _pool(level_bins, size, ops)
    finally:
        if writer is not None:
            writer.shutdown()

    manifest = {
        "format": FORMAT,
        "version": FORMAT_VERSION,
        "x": x,
        "y": y,
        "x_range": list(x_range),
        "y_range": list(y_range),
        "tile_size": tile_size,
        "max_level": max_level,
        "values": values,
        "fields": list(ops),
        "source_rows": progress[1],
        "binned_rows": binned_rows,
        "datashader_version": ds.__version__,
        "levels": {k: levels[k] for k in sorted(levels, key=int)},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, separators=(",", ":")))
    return TilePyramid(out)


def _write_level(
    out: Path,
    level: int,
    bins: _Bins,
    size: int,
    tile_size: int,
    pool: ThreadPoolExecutor | None = None,
) -> list[list[int]]:
    """Write one level as sparse tiles: in-tile bin indices plus field values."""

    yi, xi = np.divmod(bins.keys, size)
    tile_row, local_y = np.divmod(yi, tile_size)
    tile_col, local_x = np.divmod(xi, tile_size)
    tile_id = tile_row * (size // tile_size) + tile_col
    order = np.argsort(tile_id, kind="stable")
    tile_id = tile_id[order]
    index_dtype = np.uint16 if tile_size * tile_size <= 2**16 else np.uint32
    local = (local_y * tile_size + local_x)[order].astype(index_dtype)
    fields = {name.replace(":", "__"): data[order] for name, data in bins.fields.items()}
    for name in fields:
        if name.endswith("count"):
            fields[name] = fields[name].astype(np.uint32)
    starts = np.flatnonzero(np.r_[True, tile_id[1:] != tile_id[:-1]])
    ends = np.r_[starts[1:], len(tile_id)]
    per_side = size // tile_size

    level_dir = out / str(level)
    level_dir.mkdir(parents=True, exist_ok=True)
    tiles = [divmod(int(tile_id[start]), per_side) for start in starts]

    def write(job: tuple[int, int, int, int]) -> None:
        row, col, start, end = job
        arrays = {name: data[start:end] for name, data in fields.items()}
        np.savez_compressed(level_dir / f"{row}-{col}.npz", index=local[start:end], **arrays)

    jobs = [(row, col, int(a), int(b)) for (row, col), a, b in zip(tiles, starts, ends)]
    if pool is None:
        for job in jobs:
            write(job)
    else:
        list(pool.map(write, jobs))
    return [[row, col] for row, col in tiles]


def _default_reader(base: str) -> Callable[[str], bytes]:
    if base.startswith(("http://", "https://")):
        if sys.platform == "emscripten":
            return _browser_http_reader(base)
        from urllib.request import urlopen

        def read(name: str) -> bytes:
            with urlopen(f"{base.rstrip('/')}/{name}") as response:
                return response.read()

        return read

    root = Path(base)
    return lambda name: (root / name).read_bytes()


def _browser_http_reader(base: str) -> Callable[[str], bytes]:  # pragma: no cover
    # Xeus-Python runs in a Web Worker, where synchronous XHR with an
    # ArrayBuffer response is permitted.
    import pyjs

    def read(name: str) -> bytes:
        request = pyjs.js.XMLHttpRequest.new()
        request.open("GET", f"{base.rstrip('/')}/{name}", False)
        request.responseType = "arraybuffer"
        request.send()
        if not 200 <= int(request.status) < 300:
            raise FileNotFoundError(f"{name}: HTTP {request.status}")
        return bytes(pyjs.to_py(pyjs.js.Uint8Array.new(request.response)))

    return read


AsyncReader = Callable[[str], Awaitable[bytes]]


def _default_async_reader(base: str) -> AsyncReader | None:
    """Concurrent reads for an http(s) pyramid in the browser (``fetch``)."""

    if sys.platform != "emscripten" or not base.startswith(("http://", "https://")):
        return None
    return _browser_fetch_reader(base)  # pragma: no cover


def _browser_fetch_reader(base: str) -> AsyncReader:  # pragma: no cover
    # pyjs makes JavaScript promises awaitable on the browser event loop, so
    # many fetches can be in flight at once (unlike synchronous XHR).
    import pyjs

    async def read(name: str) -> bytes:
        response = await pyjs.js.fetch(f"{base.rstrip('/')}/{name}")
        if not bool(response.ok):
            raise FileNotFoundError(f"{name}: HTTP {response.status}")
        buffer = await response.arrayBuffer()
        return bytes(pyjs.to_py(pyjs.js.Uint8Array.new(buffer)))

    return read


class TilePyramid:
    """Read a tile pyramid and assemble Datashader-compatible aggregates.

    ``location`` is a local directory (including the JupyterLite browser
    filesystem) or an ``http(s)`` URL of a statically hosted pyramid.  Only
    tiles intersecting a requested viewport are read, and recently used tiles
    are kept in a small LRU cache.
    """

    def __init__(
        self,
        location: str | Path,
        *,
        reader: Callable[[str], bytes] | None = None,
        async_reader: AsyncReader | None = None,
        cache_tiles: int = 256,
        max_concurrent_reads: int = 16,
    ) -> None:
        self.location = str(location)
        self._read = reader or _default_reader(self.location)
        self._read_async = async_reader or (
            None if reader is not None else _default_async_reader(self.location)
        )
        self._max_concurrent_reads = max_concurrent_reads
        self._inflight: dict[tuple[int, int, int], Any] = {}
        self.manifest = json.loads(self._read("manifest.json"))
        if self.manifest.get("format") != FORMAT:
            raise ValueError(f"{location} is not a {FORMAT} manifest")
        if self.manifest.get("version") != FORMAT_VERSION:
            raise ValueError(f"unsupported pyramid version {self.manifest.get('version')}")
        self._present = {
            int(level): {tuple(tile) for tile in tiles}
            for level, tiles in self.manifest["levels"].items()
        }
        self._cache: OrderedDict[tuple[int, int, int], dict[str, Any]] = OrderedDict()
        self._cache_tiles = cache_tiles
        self.tiles_read = 0

    @property
    def x(self) -> str:
        return self.manifest["x"]

    @property
    def y(self) -> str:
        return self.manifest["y"]

    @property
    def x_range(self) -> Range:
        return tuple(self.manifest["x_range"])  # type: ignore[return-value]

    @property
    def y_range(self) -> Range:
        return tuple(self.manifest["y_range"])  # type: ignore[return-value]

    @property
    def max_level(self) -> int:
        return int(self.manifest["max_level"])

    @property
    def tile_size(self) -> int:
        return int(self.manifest["tile_size"])

    def level_size(self, level: int) -> int:
        """Number of bins per axis at ``level``."""

        return self.tile_size * 2**level

    def choose_level(self, x_range: Range, y_range: Range, width: int, height: int) -> int:
        """Return the coarsest level that gives at least ``width x height`` bins."""

        fx = (x_range[1] - x_range[0]) / (self.x_range[1] - self.x_range[0])
        fy = (y_range[1] - y_range[0]) / (self.y_range[1] - self.y_range[0])
        for level in range(self.max_level + 1):
            size = self.level_size(level)
            if fx * size >= width and fy * size >= height:
                return level
        return self.max_level

    def resolution(self, x_range: Range, y_range: Range, level: int | None = None) -> tuple[float, float]:
        """Bins per axis that ``level`` (default ``max_level``) provides for a viewport."""

        size = self.level_size(self.max_level if level is None else level)
        fx = (x_range[1] - x_range[0]) / (self.x_range[1] - self.x_range[0])
        fy = (y_range[1] - y_range[0]) / (self.y_range[1] - self.y_range[0])
        return fx * size, fy * size

    def _span(self, level: int, x_range: Range, y_range: Range):
        size = self.level_size(level)
        sx, tx = _scale_translate(self.x_range, size)
        sy, ty = _scale_translate(self.y_range, size)
        x0, x1 = _bin_span(x_range, sx, tx, size)
        y0, y1 = _bin_span(y_range, sy, ty, size)
        return (x0, x1, y0, y1), (sx, tx, sy, ty)

    def _tiles_for(self, level: int, bins: tuple[int, int, int, int]) -> list[tuple[int, int, int]]:
        """Tiles that exist in the manifest and intersect a bin span."""

        x0, x1, y0, y1 = bins
        ts = self.tile_size
        present = self._present.get(level, set())
        return [
            (level, row, col)
            for row in range(y0 // ts, (y1 - 1) // ts + 1)
            for col in range(x0 // ts, (x1 - 1) // ts + 1)
            if (row, col) in present
        ]

    def _normalize(self, x_range, y_range, width, height, agg, level):
        reduction = ds.count() if agg is None else agg
        x_range = self.x_range if x_range is None else (float(x_range[0]), float(x_range[1]))
        y_range = self.y_range if y_range is None else (float(y_range[0]), float(y_range[1]))
        if level is None:
            level = self.choose_level(x_range, y_range, width, height)
        if not 0 <= level <= self.max_level:
            raise ValueError(f"level must be between 0 and {self.max_level}")
        return reduction, x_range, y_range, level

    def tiles_needed(
        self,
        x_range: Range | None = None,
        y_range: Range | None = None,
        *,
        width: int = 600,
        height: int = 600,
        level: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """Return the ``(level, row, col)`` tiles a viewport needs."""

        _, x_range, y_range, level = self._normalize(x_range, y_range, width, height, None, level)
        bins, _ = self._span(level, x_range, y_range)
        return self._tiles_for(level, bins)

    def is_loaded(self, key: tuple[int, int, int]) -> bool:
        """Whether a tile's bytes are cached (absent tiles count as loaded)."""

        level, row, col = key
        return key in self._cache or (row, col) not in self._present.get(level, ())

    def aggregate(
        self,
        x_range: Range | None = None,
        y_range: Range | None = None,
        *,
        width: int = 600,
        height: int = 600,
        agg: Any | None = None,
        level: int | None = None,
        cached_only: bool = False,
    ):
        """Return an aggregate covering the viewport as an ``xarray.DataArray``.

        The viewport is snapped outward to the bin edges of the chosen level,
        so the result has at least ``width x height`` bins when the pyramid is
        deep enough (and at most roughly twice that).  ``attrs`` record the
        level, the snapped ranges and whether the result is under-resolved.

        With ``cached_only=True`` no tile is read.  The result uses the deepest
        level at or above the chosen one whose tiles are all cached (a coarser
        but complete image), and ``attrs["missing_tiles"]`` lists the tiles of
        the chosen level still to fetch, e.g. with :meth:`fetch_tiles`.
        """

        reduction, x_range, y_range, level = self._normalize(
            x_range, y_range, width, height, agg, level
        )
        missing: list[tuple[int, int, int]] = []
        if cached_only:
            bins, _ = self._span(level, x_range, y_range)
            missing = [k for k in self._tiles_for(level, bins) if not self.is_loaded(k)]
            if missing:
                for fallback in range(level - 1, -1, -1):
                    bins, _ = self._span(fallback, x_range, y_range)
                    if all(self.is_loaded(k) for k in self._tiles_for(fallback, bins)):
                        break
                else:
                    fallback = level  # nothing complete: draw what is cached
                return self._assemble(
                    reduction, x_range, y_range, width, height, fallback,
                    missing=missing, target_level=level, read=False,
                )
        return self._assemble(reduction, x_range, y_range, width, height, level, missing=missing)

    async def fetch_tiles(self, keys: Iterable[tuple[int, int, int]]) -> int:
        """Read tiles concurrently into the cache; return how many were read.

        In the browser an ``http(s)`` pyramid uses ``fetch`` so the requests
        overlap instead of paying one network round trip each.  Elsewhere,
        reads run on a thread pool.  Tiles already cached or in flight are not
        requested twice.
        """

        import asyncio

        semaphore = asyncio.Semaphore(self._max_concurrent_reads)
        loop = asyncio.get_running_loop()

        async def read(name: str) -> bytes:
            async with semaphore:
                if self._read_async is not None:
                    return await self._read_async(name)
                if sys.platform == "emscripten":  # no threads in the browser
                    return self._read(name)
                return await loop.run_in_executor(None, self._read, name)

        async def load(key: tuple[int, int, int]) -> bool:
            level, row, col = key
            data = await read(f"{level}/{row}-{col}.npz")
            self._store(key, {"__npz__": data})
            return True

        waiters = []
        started = 0
        for key in dict.fromkeys(keys):
            if self.is_loaded(key):
                continue
            future = self._inflight.get(key)
            if future is None:
                future = asyncio.ensure_future(load(key))
                self._inflight[key] = future
                future.add_done_callback(lambda _, key=key: self._inflight.pop(key, None))
                started += 1
            waiters.append(future)
        if waiters:
            await asyncio.gather(*waiters)
        return started

    async def aggregate_async(
        self,
        x_range: Range | None = None,
        y_range: Range | None = None,
        *,
        width: int = 600,
        height: int = 600,
        agg: Any | None = None,
        level: int | None = None,
    ):
        """Like :meth:`aggregate`, but fetch the needed tiles concurrently first."""

        await self.fetch_tiles(
            self.tiles_needed(x_range, y_range, width=width, height=height, level=level)
        )
        return self.aggregate(x_range, y_range, width=width, height=height, agg=agg, level=level)

    def _assemble(
        self,
        reduction: Any,
        x_range: Range,
        y_range: Range,
        width: int,
        height: int,
        level: int,
        *,
        missing: list[tuple[int, int, int]],
        target_level: int | None = None,
        read: bool = True,
    ):
        field_names, finalize, name = self._reduction_fields(reduction)
        (x0, x1, y0, y1), (sx, tx, sy, ty) = self._span(level, x_range, y_range)
        arrays = {
            field: np.full((y1 - y0, x1 - x0), _empty_value(field), _field_dtype(field))
            for field in field_names
        }
        ts = self.tile_size
        for key in self._tiles_for(level, (x0, x1, y0, y1)):
            if not read and key not in self._cache:
                continue
            _, row, col = key
            tile = self._tile(level, row, col, field_names)
            ty0, tx0 = row * ts, col * ts
            sy0, sy1 = max(y0, ty0), min(y1, ty0 + ts)
            sx0, sx1 = max(x0, tx0), min(x1, tx0 + ts)
            for field in field_names:
                arrays[field][sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = tile[field][
                    sy0 - ty0 : sy1 - ty0, sx0 - tx0 : sx1 - tx0
                ]

        import xarray as xr

        data = finalize(arrays)
        xs = (np.arange(x0, x1) + 0.5 - tx) / sx
        ys = (np.arange(y0, y1) + 0.5 - ty) / sy
        snapped_x = ((x0 - tx) / sx, (x1 - tx) / sx)
        snapped_y = ((y0 - ty) / sy, (y1 - ty) / sy)
        goal = level if target_level is None else target_level
        bins_x, bins_y = self.resolution(x_range, y_range, goal)
        return xr.DataArray(
            data,
            coords={self.y: ys, self.x: xs},
            dims=[self.y, self.x],
            name=name,
            attrs={
                "x_range": snapped_x,
                "y_range": snapped_y,
                "pyramid_level": level,
                "target_level": goal,
                "missing_tiles": missing,
                "under_resolved": bool(bins_x < width / 2 or bins_y < height / 2),
            },
        )

    def _reduction_fields(self, reduction: Any):
        kind = type(reduction).__name__
        column = getattr(reduction, "column", None)
        values = self.manifest["values"]
        if kind == "count" and column is None:
            return [COUNT], lambda a: a[COUNT], COUNT
        if column not in values:
            raise ValueError(
                f"column {column!r} was not stored in this pyramid; "
                f"rebuild with values={[*values, column]!r}"
            )
        prefix = f"{column}:"
        if kind == "count":
            return [prefix + "count"], lambda a: a[prefix + "count"], kind
        if kind in ("min", "max"):
            field = prefix + kind
            return [field], lambda a: a[field], kind
        if kind == "sum":
            # Datashader reports NaN, not 0, where a bin has no non-null values.
            return (
                [prefix + "sum", prefix + "count"],
                lambda a: np.where(a[prefix + "count"] > 0, a[prefix + "sum"], np.nan),
                kind,
            )
        if kind == "mean":

            def mean(a: dict[str, np.ndarray]) -> np.ndarray:
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.where(
                        a[prefix + "count"] > 0,
                        a[prefix + "sum"] / a[prefix + "count"],
                        np.nan,
                    )

            return [prefix + "sum", prefix + "count"], mean, kind
        raise NotImplementedError(
            f"{kind} is not supported by tile pyramids; use count, sum, mean, min or max"
        )

    def _tile(
        self, level: int, row: int, col: int, fields: Sequence[str]
    ) -> dict[str, np.ndarray] | None:
        """Return dense arrays for the requested fields of one tile.

        Each field in an ``.npz`` is a separately compressed member, so only
        the fields a reduction needs are decompressed.
        """

        if (row, col) not in self._present.get(level, ()):
            return None
        key = (level, row, col)
        cached = self._cache.get(key)
        if cached is None:
            cached = self._store(key, {"__npz__": self._read(f"{level}/{row}-{col}.npz")})
        else:
            self._cache.move_to_end(key)

        missing = [field for field in fields if field not in cached]
        if missing:
            ts = self.tile_size
            with np.load(io.BytesIO(cached["__npz__"])) as npz:
                index = npz["index"]
                for name in missing:
                    dense = np.full(ts * ts, _empty_value(name), _field_dtype(name))
                    dense[index] = npz[name.replace(":", "__")]
                    cached[name] = dense.reshape(ts, ts)
        return {field: cached[field] for field in fields}


    def _store(self, key: tuple[int, int, int], entry: dict[str, Any]) -> dict[str, Any]:
        self.tiles_read += 1
        self._cache[key] = entry
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_tiles:
            self._cache.popitem(last=False)
        return entry


def _bin_span(viewport: Range, s: float, t: float, size: int) -> tuple[int, int]:
    lo = math.floor(viewport[0] * s + t)
    hi = math.ceil(viewport[1] * s + t)
    lo = min(max(lo, 0), size - 1)
    hi = min(max(hi, lo + 1), size)
    return lo, hi
