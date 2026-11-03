"""Optional HoloViews integration for local browser-side rerendering."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Any

import datashader as ds

from .browser import IN_BROWSER, install_kernel_opened_comms, repair_tornado_stub
from .pyramid import TilePyramid
from .sources import ParquetSource, Range
from .streaming import CancelCallback, ProgressCallback, StreamingCanvas

DEFAULT_CMAP = ["#111827", "#1d4ed8", "#06b6d4", "#facc15", "#fff7ed"]


def datashade(
    source: ParquetSource | TilePyramid,
    x: str | None = None,
    y: str | None = None,
    *,
    agg: Any | None = None,
    x_range: Range | None = None,
    y_range: Range | None = None,
    width: int = 800,
    height: int = 500,
    cmap: Sequence[str] | str | None = None,
    how: str = "eq_hist",
    fallback: ParquetSource | None = None,
    progressive: bool | None = None,
    on_progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
    transform: Callable[[Any], Any] | None = None,
) -> Any:
    """Return a HoloViews DynamicMap rerendered for each visible viewport.

    HoloViews and Bokeh provide the interaction and display layers. This
    adapter supplies only the missing bridge from ``RangeXY`` events to one of
    two data planes:

    * :class:`ParquetSource` - a bounded-memory scan of the file on every
      viewport change. Viewport predicates are evaluated per Arrow batch, so
      work is reduced but the whole file is still read.
    * :class:`TilePyramid` - only the precomputed aggregate tiles that
      intersect the viewport are read, so pan and zoom cost is independent of
      the number of source rows. When the viewport is zoomed past the deepest
      level, ``fallback`` (if given) is scanned for full resolution.

    ``progressive`` (default: on in the browser) makes pyramid rendering
    non-blocking: each viewport is drawn at once from cached tiles (using a
    coarser level where needed), the missing tiles are fetched concurrently
    in the background, and the plot is redrawn when they arrive. Without it,
    tiles are read one after another before the plot updates.
    """

    pyramid = source if isinstance(source, TilePyramid) else None
    if pyramid is None and not isinstance(source, ParquetSource):
        raise TypeError("source must be a ParquetSource or TilePyramid")
    if fallback is not None and not isinstance(fallback, ParquetSource):
        raise TypeError("fallback must be a ParquetSource")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")
    if pyramid is not None:
        x = pyramid.x if x is None else x
        y = pyramid.y if y is None else y
        if (x, y) != (pyramid.x, pyramid.y):
            raise ValueError(f"pyramid was built for ({pyramid.x!r}, {pyramid.y!r})")
    elif x is None or y is None:
        raise ValueError("x and y are required for a ParquetSource")

    repair_tornado_stub()
    try:
        import holoviews as hv
        from holoviews.operation.datashader import shade as hv_shade
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Interactive rendering requires HoloViews; install "
            "datashader-browser[holoviews]"
        ) from exc

    if "bokeh" not in hv.Store.renderers:
        if IN_BROWSER:
            # Embed BokehJS/Panel in the output. Loading them from the CDN is
            # asynchronous in JupyterLite, and a plot displayed before the
            # download finishes stays blank ("Bokeh or Panel library is missing").
            hv.extension("bokeh", inline=True)
        else:
            hv.extension("bokeh")
    # Xeus-Python drops page-opened comms, so zoom events need this (browser only).
    install_kernel_opened_comms()

    if x_range is None or y_range is None:
        if pyramid is not None:
            source_x_range, source_y_range = pyramid.x_range, pyramid.y_range
        else:
            source_x_range, source_y_range = source.bounds(x, y)
        x_range = source_x_range if x_range is None else x_range
        y_range = source_y_range if y_range is None else y_range

    initial_x_range = _coerce_range(x_range)
    initial_y_range = _coerce_range(y_range)
    reduction = ds.count() if agg is None else agg
    columns = _required_columns(x, y, reduction)

    def scan(scan_source: ParquetSource, visible_x: Range, visible_y: Range) -> Any:
        canvas = StreamingCanvas(
            plot_width=width,
            plot_height=height,
            x_range=visible_x,
            y_range=visible_y,
        )
        aggregate = canvas.points(
            scan_source.scan(
                columns=columns,
                x=x,
                y=y,
                x_range=visible_x,
                y_range=visible_y,
            ),
            x,
            y,
            agg=reduction,
            on_progress=on_progress,
            cancelled=cancelled,
        )
        aggregate.attrs["data_plane"] = "scan"
        return aggregate

    use_progressive = IN_BROWSER if progressive is None else progressive
    refresh = hv.streams.Stream.define("TileRefresh", generation=0)()
    background: set[Any] = set()
    latest: dict[str, Any] = {"view": None, "generation": 0}

    def fetch_then_refresh(view: tuple[Range, Range], missing: list) -> None:
        async def run() -> None:
            await pyramid.fetch_tiles(missing)
            # Redraw only if the user has not moved on to another viewport.
            if latest["view"] == view:
                latest["generation"] += 1
                refresh.event(generation=latest["generation"])

        def done(task: Any) -> None:
            background.discard(task)
            if not task.cancelled() and task.exception() is not None:
                print(f"datashader-browser: tile fetch failed: {task.exception()!r}")

        task = asyncio.ensure_future(run())
        background.add(task)
        task.add_done_callback(done)

    def render(
        x_range: Range | None = initial_x_range,
        y_range: Range | None = initial_y_range,
        generation: int = 0,
    ) -> Any:
        visible_x = _coerce_range(x_range, fallback=initial_x_range)
        visible_y = _coerce_range(y_range, fallback=initial_y_range)
        if pyramid is None:
            aggregate = scan(source, visible_x, visible_y)
        else:
            progressive_now = use_progressive and _event_loop_running()
            aggregate = pyramid.aggregate(
                visible_x, visible_y, width=width, height=height, agg=reduction,
                cached_only=progressive_now,
            )
            aggregate.attrs["data_plane"] = "tiles"
            if progressive_now:
                latest["view"] = (visible_x, visible_y)
                if aggregate.attrs["missing_tiles"]:
                    fetch_then_refresh(latest["view"], aggregate.attrs["missing_tiles"])
            if (
                fallback is not None
                and aggregate.attrs["under_resolved"]
                and not aggregate.attrs["missing_tiles"]
            ):
                aggregate = scan(fallback, visible_x, visible_y)
        image_x = aggregate.attrs["x_range"]
        image_y = aggregate.attrs["y_range"]
        if transform is not None:
            aggregate = transform(aggregate)

        # HoloViews bounds use (left, bottom, right, top), not grouped ranges.
        bounds = (image_x[0], image_y[0], image_x[1], image_y[1])
        image = hv.Image(aggregate, kdims=[x, y], bounds=bounds)
        rgb = hv_shade(
            image,
            cmap=DEFAULT_CMAP if cmap is None else cmap,
            cnorm=how,
        )
        rgb.data.attrs.update(aggregate.attrs)
        return rgb

    ranges = hv.streams.RangeXY(
        x_range=initial_x_range,
        y_range=initial_y_range,
    )
    dynamic = hv.DynamicMap(render, streams=[ranges, refresh])
    return dynamic.opts(
        framewise=True,
        width=width,
        height=height,
        active_tools=["wheel_zoom"],
    )


def _event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        pass
    try:
        # The browser's pyjs WebLoop runs continuously but is not "running"
        # from the point of view of synchronous code executed in a cell.
        return asyncio.get_event_loop().is_running()
    except RuntimeError:
        return False


def _required_columns(x: str, y: str, reduction: Any) -> list[str]:
    columns = [x, y]
    column = getattr(reduction, "column", None)
    if isinstance(column, str) and column not in columns:
        columns.append(column)
    return columns


def _coerce_range(value: Any, *, fallback: Range | None = None) -> Range:
    if value is None or len(value) != 2 or value[0] is None or value[1] is None:
        if fallback is None:
            raise ValueError("viewport ranges must contain two values")
        return fallback
    result = (float(value[0]), float(value[1]))
    if result[0] >= result[1]:
        if fallback is None:
            raise ValueError("viewport range minimum must be less than its maximum")
        return fallback
    return result
