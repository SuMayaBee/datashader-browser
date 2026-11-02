"""Command-line interface: ``datashader-browser build-tiles``."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

from .pyramid import build_pyramid
from .sources import ParquetSource
from .streaming import Progress, iter_csv_chunks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="datashader-browser")
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build-tiles",
        help="aggregate a Parquet or CSV point file into a static tile pyramid",
    )
    build.add_argument("source", type=Path, help="Parquet or CSV file")
    build.add_argument("--x", required=True, help="x coordinate column")
    build.add_argument("--y", required=True, help="y coordinate column")
    build.add_argument("--out", type=Path, required=True, help="output directory")
    build.add_argument("--x-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    build.add_argument("--y-range", type=float, nargs=2, metavar=("MIN", "MAX"))
    build.add_argument("--max-level", type=int, default=6)
    build.add_argument("--tile-size", type=int, default=256)
    build.add_argument(
        "--values", nargs="*", default=[],
        help="numeric columns to store for sum/mean/min/max",
    )
    build.add_argument("--batch-size", type=int, default=500_000)
    build.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    columns = [args.x, args.y, *args.values]
    if args.source.suffix.lower() == ".csv":
        if args.x_range is None or args.y_range is None:
            parser.error("--x-range and --y-range are required for CSV input")
        chunks = iter_csv_chunks(args.source, chunksize=args.batch_size, columns=columns)
        x_range, y_range = args.x_range, args.y_range
    else:
        source = ParquetSource(args.source, batch_size=args.batch_size)
        if args.x_range is None or args.y_range is None:
            bounds = source.bounds(args.x, args.y)
            print(f"extent from data: x={bounds[0]} y={bounds[1]}", file=sys.stderr)
        x_range = args.x_range or bounds[0]
        y_range = args.y_range or bounds[1]
        chunks = source.scan(columns=columns)

    started = time.perf_counter()

    def report(progress: Progress) -> None:
        print(f"\r{progress.rows:,} rows", end="", file=sys.stderr, flush=True)

    pyramid = build_pyramid(
        chunks,
        args.out,
        args.x,
        args.y,
        x_range=tuple(x_range),
        y_range=tuple(y_range),
        max_level=args.max_level,
        tile_size=args.tile_size,
        values=args.values,
        workers=args.workers,
        on_progress=report,
    )
    manifest = pyramid.manifest
    tiles = sum(len(t) for t in manifest["levels"].values())
    size = sum(p.stat().st_size for p in Path(args.out).rglob("*") if p.is_file())
    print(
        f"\nwrote {tiles:,} tiles over {manifest['max_level'] + 1} levels "
        f"({size / 2**20:.1f} MiB) for {manifest['binned_rows']:,} of "
        f"{manifest['source_rows']:,} rows in {time.perf_counter() - started:.1f}s "
        f"-> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
