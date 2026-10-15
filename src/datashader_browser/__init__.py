"""Run Datashader over bounded-memory streams in a browser Python runtime."""

from .sources import ParquetSource
from .streaming import Progress, StreamingCanvas, iter_csv_chunks, iter_parquet_batches

__all__ = [
    "ParquetSource",
    "Progress",
    "StreamingCanvas",
    "iter_csv_chunks",
    "iter_parquet_batches",
]
__version__ = "0.2.0a0"
