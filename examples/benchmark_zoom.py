"""Compare per-viewport cost: full Parquet scan vs. tile pyramid (NYC Taxi).

Build the pyramid first::

    uv run datashader-browser build-tiles nyc_taxi_wide.parq \\
        --x pickup_x --y pickup_y \\
        --x-range -8256000 -8209000 --y-range 4964000 4990000 \\
        --max-level 6 --values fare_amount --workers 8 --out .cache/nyc-tiles

then run::

    uv run python examples/benchmark_zoom.py --data nyc_taxi_wide.parq --tiles .cache/nyc-tiles

The script also checks that a pyramid level is identical to native Datashader
on the real data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

import datashader as ds
from datashader_browser import ParquetSource, StreamingCanvas, TilePyramid

X_RANGE = (-8_256_000.0, -8_209_000.0)
Y_RANGE = (4_964_000.0, 4_990_000.0)

# A zoom sequence from the whole city down to a few blocks of Midtown.
VIEWPORTS = [
    ("NYC", X_RANGE, Y_RANGE),
    ("Manhattan", (-8_242_000.0, -8_228_000.0), (4_965_000.0, 4_982_000.0)),
    ("Midtown", (-8_237_000.0, -8_233_000.0), (4_973_000.0, 4_977_000.0)),
    ("Times Sq", (-8_236_200.0, -8_235_200.0), (4_975_000.0, 4_975_800.0)),
    ("Block", (-8_235_900.0, -8_235_600.0), (4_975_300.0, 4_975_500.0)),
]


def scan(source: ParquetSource, x_range, y_range, width, height, agg):
    canvas = StreamingCanvas(
        plot_width=width, plot_height=height, x_range=x_range, y_range=y_range
    )
    columns = ["pickup_x", "pickup_y", *([agg.column] if agg.column else [])]
    return canvas.points(
        source.scan(columns=columns, x="pickup_x", y="pickup_y",
                    x_range=x_range, y_range=y_range),
        "pickup_x", "pickup_y", agg=agg,
    )


def check_exactness(data: Path, pyramid: TilePyramid, level: int) -> dict:
    table = pq.read_table(data, columns=["pickup_x", "pickup_y", "fare_amount"])
    frame = table.to_pandas()
    n = pyramid.level_size(level)
    canvas = ds.Canvas(plot_width=n, plot_height=n, x_range=X_RANGE, y_range=Y_RANGE)
    result = {}
    for agg in (ds.count(), ds.max("fare_amount"), ds.mean("fare_amount")):
        expected = canvas.points(frame, "pickup_x", "pickup_y", agg=agg).values
        actual = pyramid.aggregate(level=level, agg=agg).values
        if type(agg).__name__ == "mean":
            ok = np.allclose(actual, expected, rtol=1e-9, equal_nan=True)
        else:
            ok = np.array_equal(actual, expected, equal_nan=True)
        result[str(agg)] = bool(ok)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=500)
    parser.add_argument("--exact-level", type=int, default=3)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    source = ParquetSource(args.data, batch_size=250_000)
    agg = ds.count()
    rows = []

    # Warm Numba compilation so neither path is charged for it.
    scan(source, VIEWPORTS[-1][1], VIEWPORTS[-1][2], 10, 10, agg)

    session = TilePyramid(args.tiles)  # one pan/zoom session: cache persists
    for name, x_range, y_range in VIEWPORTS:
        started = time.perf_counter()
        scanned = scan(source, x_range, y_range, args.width, args.height, agg)
        scan_s = time.perf_counter() - started

        cold = TilePyramid(args.tiles)  # nothing cached: worst case
        started = time.perf_counter()
        tiled = cold.aggregate(x_range, y_range, width=args.width,
                               height=args.height, agg=agg)
        cold_s = time.perf_counter() - started
        tile_bytes = sum(
            (args.tiles / str(tiled.attrs["pyramid_level"]) / f"{r}-{c}.npz").stat().st_size
            for (_, r, c) in cold._cache
        )

        before = session.tiles_read
        started = time.perf_counter()
        session.aggregate(x_range, y_range, width=args.width,
                          height=args.height, agg=agg)
        session_s = time.perf_counter() - started
        rows.append({
            "viewport": name,
            "scan_s": round(scan_s, 3),
            "tiles_cold_s": round(cold_s, 4),
            "tiles_session_s": round(session_s, 4),
            "level": tiled.attrs["pyramid_level"],
            "tiles_fetched_cold": cold.tiles_read,
            "tiles_fetched_session": session.tiles_read - before,
            "tile_mib_cold": round(tile_bytes / 2**20, 2),
            "under_resolved": tiled.attrs["under_resolved"],
            "points_in_view": int(scanned.sum()),
        })

    print(f"{'viewport':10} {'points':>11} {'scan s':>7} {'cold s':>7} "
          f"{'session s':>9} {'lvl':>3} {'tiles':>5} {'MiB':>6}  under-res")
    for r in rows:
        print(f"{r['viewport']:10} {r['points_in_view']:11,} {r['scan_s']:7.3f} "
              f"{r['tiles_cold_s']:7.4f} {r['tiles_session_s']:9.4f} {r['level']:3d} "
              f"{r['tiles_fetched_cold']:5d} {r['tile_mib_cold']:6.2f}  {r['under_resolved']}")

    exact = check_exactness(args.data, session, args.exact_level)
    print(f"level {args.exact_level} identical to native Datashader:", exact)
    if args.json:
        args.json.write_text(json.dumps({"viewports": rows, "exact": exact}, indent=2))


if __name__ == "__main__":
    main()
