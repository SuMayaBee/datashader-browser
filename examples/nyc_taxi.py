"""Render the HoloViz January 2015 NYC Taxi dataset in bounded memory.

Run with::

    uv sync --extra data
    uv run python examples/nyc_taxi.py --download

The Parquet file is cached under ``.cache/data`` and is not committed.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
import urllib.request
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

import datashader.transfer_functions as tf
from datashader_browser import Progress, StreamingCanvas


DATA_URL = "https://datasets.holoviz.org/nyc_taxi/v1/nyc_taxi_wide.parq"
DEFAULT_DATA = Path(".cache/data/nyc_taxi_wide.parq")
DEFAULT_OUTPUT = Path("docs/assets/nyc-taxi-pickups.png")

# Published-data bounds in Web Mercator with a small fixed margin. Keeping the
# bounds explicit preserves a true single-pass render.
X_RANGE = (-8_256_000, -8_209_000)
Y_RANGE = (4_964_000, 4_990_000)


def download(url: str, destination: Path) -> None:
    """Download the published dataset without retaining it in the repository."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "datashader-browser-research/0.1"},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        while block := response.read(1024 * 1024):
            output.write(block)


def parquet_chunks(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    """Decode only the coordinate columns, one Arrow batch at a time."""

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["pickup_x", "pickup_y"],
        use_threads=True,
    ):
        yield batch.to_pandas()


def peak_rss_mib() -> float:
    """Return peak resident memory on Linux in MiB."""

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        if not args.download:
            raise SystemExit(f"{args.data} is missing; rerun with --download")
        download(DATA_URL, args.data)

    parquet = pq.ParquetFile(args.data)
    expected_rows = parquet.metadata.num_rows
    started = time.perf_counter()

    def report(progress: Progress) -> None:
        if progress.chunks % 10 == 0 or progress.rows == expected_rows:
            print(f"aggregated {progress.rows:,}/{expected_rows:,} rows")

    canvas = StreamingCanvas(
        plot_width=1_200,
        plot_height=700,
        x_range=X_RANGE,
        y_range=Y_RANGE,
    )
    aggregate = canvas.points(
        parquet_chunks(args.data, args.batch_size),
        "pickup_x",
        "pickup_y",
        on_progress=report,
    )

    image = tf.shade(
        aggregate,
        cmap=["#111827", "#1d4ed8", "#06b6d4", "#facc15", "#fff7ed"],
        how="eq_hist",
    )
    image = tf.set_background(image, "#050814")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.to_pil().save(args.output)

    result = {
        "source": DATA_URL,
        "compressed_bytes": args.data.stat().st_size,
        "rows": aggregate.attrs["source_rows"],
        "chunks": aggregate.attrs["source_chunks"],
        "batch_size": args.batch_size,
        "canvas": [canvas.canvas.plot_width, canvas.canvas.plot_height],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "peak_rss_mib": round(peak_rss_mib(), 1),
        "output": str(args.output),
    }
    print(json.dumps(result, indent=2))

    if result["rows"] != expected_rows:
        raise RuntimeError(
            f"row-count mismatch: expected {expected_rows}, aggregated {result['rows']}"
        )


if __name__ == "__main__":
    main()
