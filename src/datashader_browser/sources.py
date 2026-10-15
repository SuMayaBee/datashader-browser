"""Repeatable bounded-memory data sources for interactive browser rendering."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .streaming import iter_parquet_batches

Range = tuple[float, float]
Bounds = tuple[Range, Range]


@dataclass(slots=True)
class ParquetSource:
    """A repeatable, projected Parquet record-batch source.

    The source stores only a path and scan options. Each call to :meth:`scan`
    opens a fresh iterator, which is required when a HoloViews viewport event
    triggers another render. Optional viewport filtering is applied to each
    bounded Arrow batch; it reduces aggregation work but still performs a
    sequential file scan until a spatially indexed backend is added.
    """

    path: str | Path
    batch_size: int = 250_000
    row_groups: Sequence[int] | None = None
    use_threads: bool = False
    _bounds_cache: dict[tuple[str, str], Bounds] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

    @property
    def schema(self) -> Any:
        """Return the Arrow schema without materializing table rows."""

        return self._parquet_file().schema_arrow

    @property
    def num_rows(self) -> int:
        """Return the number of rows selected by this source."""

        parquet = self._parquet_file()
        if self.row_groups is None:
            return parquet.metadata.num_rows
        return sum(
            parquet.metadata.row_group(index).num_rows for index in self.row_groups
        )

    def scan(
        self,
        *,
        columns: Sequence[str] | None = None,
        x: str | None = None,
        y: str | None = None,
        x_range: Range | None = None,
        y_range: Range | None = None,
    ) -> Iterator[Any]:
        """Yield projected Arrow batches, optionally filtered to a viewport.

        Filtering is deliberately batch-local, so peak decoded input memory is
        controlled by ``batch_size``. Coordinate columns are automatically
        included when their ranges are used as predicates.
        """

        if (x_range is None) != (x is None):
            raise ValueError("x and x_range must be provided together")
        if (y_range is None) != (y is None):
            raise ValueError("y and y_range must be provided together")
        _validate_range("x_range", x_range)
        _validate_range("y_range", y_range)

        projected = None if columns is None else list(dict.fromkeys(columns))
        if projected is not None:
            for coordinate in (
                x if x_range is not None else None,
                y if y_range is not None else None,
            ):
                if coordinate is not None and coordinate not in projected:
                    projected.append(coordinate)

        batches = iter_parquet_batches(
            self.path,
            batch_size=self.batch_size,
            columns=projected,
            row_groups=self.row_groups,
            use_threads=self.use_threads,
        )
        if x_range is None and y_range is None:
            yield from batches
            return

        try:
            import pyarrow.compute as pc
        except ImportError as exc:  # pragma: no cover - guarded by parquet extra
            raise ImportError(
                "Viewport filtering requires PyArrow; install datashader-browser[data]"
            ) from exc

        first_empty = None
        yielded = False
        for batch in batches:
            mask = None
            if x_range is not None:
                assert x is not None
                x_mask = pc.and_kleene(
                    pc.greater_equal(batch[x], x_range[0]),
                    pc.less_equal(batch[x], x_range[1]),
                )
                mask = x_mask
            if y_range is not None:
                assert y is not None
                y_mask = pc.and_kleene(
                    pc.greater_equal(batch[y], y_range[0]),
                    pc.less_equal(batch[y], y_range[1]),
                )
                mask = y_mask if mask is None else pc.and_kleene(mask, y_mask)
            filtered = batch.filter(mask)
            if filtered.num_rows:
                yielded = True
                yield filtered
            elif first_empty is None:
                first_empty = filtered

        # StreamingCanvas needs one typed chunk to compile an empty viewport.
        if not yielded and first_empty is not None:
            yield first_empty

    def bounds(self, x: str, y: str) -> Bounds:
        """Return finite coordinate bounds, using Parquet statistics when possible."""

        key = (x, y)
        cached = self._bounds_cache.get(key)
        if cached is not None:
            return cached

        bounds = self._statistics_bounds(x, y)
        if bounds is None:
            bounds = self._scan_bounds(x, y)
        self._bounds_cache[key] = bounds
        return bounds

    def _parquet_file(self) -> Any:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise ImportError(
                "Parquet sources require PyArrow; install datashader-browser[data]"
            ) from exc
        return pq.ParquetFile(self.path)

    def _statistics_bounds(self, x: str, y: str) -> Bounds | None:
        parquet = self._parquet_file()
        names = parquet.schema_arrow.names
        try:
            indexes = (names.index(x), names.index(y))
        except ValueError as exc:
            missing = x if x not in names else y
            raise KeyError(f"column {missing!r} is not present in {self.path}") from exc

        row_groups = (
            range(parquet.metadata.num_row_groups)
            if self.row_groups is None
            else self.row_groups
        )
        minima = [np.inf, np.inf]
        maxima = [-np.inf, -np.inf]
        for row_group in row_groups:
            metadata = parquet.metadata.row_group(row_group)
            for axis, column_index in enumerate(indexes):
                statistics = metadata.column(column_index).statistics
                if statistics is None or not statistics.has_min_max:
                    return None
                try:
                    minimum = float(statistics.min)
                    maximum = float(statistics.max)
                except (TypeError, ValueError):
                    return None
                if not np.isfinite(minimum) or not np.isfinite(maximum):
                    return None
                minima[axis] = min(minima[axis], minimum)
                maxima[axis] = max(maxima[axis], maximum)

        if any(not np.isfinite(value) for value in (*minima, *maxima)):
            return None
        return _expand_degenerate_bounds(
            ((minima[0], maxima[0]), (minima[1], maxima[1]))
        )

    def _scan_bounds(self, x: str, y: str) -> Bounds:
        minima = [np.inf, np.inf]
        maxima = [-np.inf, -np.inf]
        for batch in self.scan(columns=[x, y]):
            for axis, column in enumerate((x, y)):
                values = batch[column].to_numpy(zero_copy_only=False)
                if values.size == 0:
                    continue
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    continue
                minima[axis] = min(minima[axis], float(finite.min()))
                maxima[axis] = max(maxima[axis], float(finite.max()))

        if any(not np.isfinite(value) for value in (*minima, *maxima)):
            raise ValueError(f"columns {x!r} and {y!r} contain no finite coordinates")
        return _expand_degenerate_bounds(
            ((minima[0], maxima[0]), (minima[1], maxima[1]))
        )


def _validate_range(name: str, value: Range | None) -> None:
    if value is None:
        return
    if len(value) != 2 or not all(np.isfinite(bound) for bound in value):
        raise ValueError(f"{name} must contain two finite values")
    if value[0] >= value[1]:
        raise ValueError(f"{name} minimum must be less than its maximum")


def _expand_degenerate_bounds(bounds: Bounds) -> Bounds:
    expanded: list[Range] = []
    for minimum, maximum in bounds:
        if minimum == maximum:
            padding = max(abs(minimum) * 1e-9, 1e-9)
            minimum -= padding
            maximum += padding
        expanded.append((minimum, maximum))
    return expanded[0], expanded[1]
