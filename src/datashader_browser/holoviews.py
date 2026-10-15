"""Optional HoloViews integration for local browser-side rerendering."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import datashader as ds

from .sources import ParquetSource, Range
from .streaming import CancelCallback, ProgressCallback, StreamingCanvas

DEFAULT_CMAP = ["#111827", "#1d4ed8", "#06b6d4", "#facc15", "#fff7ed"]


def datashade(
    source: ParquetSource,
    x: str,
    y: str,
    *,
    agg: Any | None = None,
    x_range: Range | None = None,
    y_range: Range | None = None,
    width: int = 800,
    height: int = 500,
    cmap: Sequence[str] | str | None = None,
    how: str = "eq_hist",
    on_progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
    transform: Callable[[Any], Any] | None = None,
) -> Any:
    """Return a HoloViews DynamicMap rerendered for each visible viewport.

    HoloViews and Bokeh provide the interaction and display layers. This
    adapter supplies only the missing bridge from ``RangeXY`` events to a
    repeatable bounded-memory Parquet scan and ``StreamingCanvas``.

    Viewport predicates are evaluated per Arrow batch. They reduce aggregation
    work, but the current PyArrow backend still scans the source file for each
    render. Spatial pruning remains a future data-plane optimization.
    """

    if not isinstance(source, ParquetSource):
        raise TypeError("source must be a ParquetSource")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be greater than zero")

    try:
        import holoviews as hv
        from holoviews.operation.datashader import shade as hv_shade
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise ImportError(
            "Interactive rendering requires HoloViews; install "
            "datashader-browser[holoviews]"
        ) from exc

    if "bokeh" not in hv.Store.renderers:
        hv.extension("bokeh")

    if x_range is None or y_range is None:
        source_x_range, source_y_range = source.bounds(x, y)
        x_range = source_x_range if x_range is None else x_range
        y_range = source_y_range if y_range is None else y_range

    initial_x_range = _coerce_range(x_range)
    initial_y_range = _coerce_range(y_range)
    reduction = ds.count() if agg is None else agg
    columns = _required_columns(x, y, reduction)

    def render(
        x_range: Range | None = initial_x_range,
        y_range: Range | None = initial_y_range,
    ) -> Any:
        visible_x = _coerce_range(x_range, fallback=initial_x_range)
        visible_y = _coerce_range(y_range, fallback=initial_y_range)
        canvas = StreamingCanvas(
            plot_width=width,
            plot_height=height,
            x_range=visible_x,
            y_range=visible_y,
        )
        aggregate = canvas.points(
            source.scan(
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
        if transform is not None:
            aggregate = transform(aggregate)

        # HoloViews bounds use (left, bottom, right, top), not grouped ranges.
        bounds = (visible_x[0], visible_y[0], visible_x[1], visible_y[1])
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
    dynamic = hv.DynamicMap(render, streams=[ranges])
    return dynamic.opts(
        framewise=True,
        width=width,
        height=height,
        active_tools=["wheel_zoom"],
    )


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
