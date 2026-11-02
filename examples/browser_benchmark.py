"""Measure raw-scan vs. tile-pyramid zoom inside a real browser (JupyterLite).

The script serves a built JupyterLite site plus the data from one local HTTP
server, opens the JupyterLite REPL in headless Chrome with the Xeus-Python
WebAssembly kernel, runs the benchmark there and prints the JSON results.

    uv run --with playwright python examples/browser_benchmark.py \\
        --site dist --parquet nyc_taxi_wide.parq --tiles .cache/nyc-tiles

Use ``--skip-scan`` to measure only the pyramid (no 385 MB download).
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
from urllib.parse import quote

MARKER = "@@BENCHMARK@@"

KERNEL_CODE = r'''
import json, time, pyjs
import datashader as ds
from datashader_browser import ParquetSource, StreamingCanvas, TilePyramid

ORIGIN = str(pyjs.js.self.location.origin)
WIDTH, HEIGHT = 800, 500
VIEWPORTS = __VIEWPORTS__
SKIP_SCAN = __SKIP_SCAN__
result = {"user_agent": str(pyjs.js.navigator.userAgent), "viewports": []}

def fetch_to_file(url, path):
    request = pyjs.js.XMLHttpRequest.new()
    request.open("GET", url, False)
    request.responseType = "arraybuffer"
    request.send()
    data = pyjs.to_py(pyjs.js.Uint8Array.new(request.response))
    with open(path, "wb") as handle:
        handle.write(memoryview(data))
    return len(data)

def scan(source, xr_, yr_, agg):
    canvas = StreamingCanvas(plot_width=WIDTH, plot_height=HEIGHT, x_range=xr_, y_range=yr_)
    return canvas.points(
        source.scan(columns=["pickup_x", "pickup_y"], x="pickup_x", y="pickup_y",
                    x_range=xr_, y_range=yr_),
        "pickup_x", "pickup_y", agg=agg)

if not SKIP_SCAN:
    t = time.perf_counter()
    size = fetch_to_file(ORIGIN + "/data/nyc.parq", "/tmp/nyc.parq")
    result["parquet_download_s"] = time.perf_counter() - t
    result["parquet_mib"] = size / 2**20
    source = ParquetSource("/tmp/nyc.parq", batch_size=250_000)
    t = time.perf_counter()
    scan(source, tuple(VIEWPORTS[-1][1]), tuple(VIEWPORTS[-1][2]), ds.count())
    result["scan_warmup_s"] = time.perf_counter() - t

t = time.perf_counter()
pyramid = TilePyramid(ORIGIN + "/data/tiles")
result["manifest_s"] = time.perf_counter() - t
t = time.perf_counter()
pyramid.aggregate(level=0)
result["tiles_warmup_s"] = time.perf_counter() - t

for name, xr_, yr_ in VIEWPORTS:
    row = {"viewport": name}
    xr_, yr_ = tuple(xr_), tuple(yr_)
    if not SKIP_SCAN:
        t = time.perf_counter()
        agg = scan(source, xr_, yr_, ds.count())
        row["scan_s"] = time.perf_counter() - t
        row["points_in_view"] = int(agg.sum())
    cold = TilePyramid(ORIGIN + "/data/tiles")
    t = time.perf_counter()
    tiled = cold.aggregate(xr_, yr_, width=WIDTH, height=HEIGHT)
    row["tiles_cold_s"] = time.perf_counter() - t
    row["tiles_fetched"] = cold.tiles_read
    t = time.perf_counter()
    pyramid.aggregate(xr_, yr_, width=WIDTH, height=HEIGHT)
    row["tiles_session_s"] = time.perf_counter() - t
    row["level"] = tiled.attrs["pyramid_level"]
    result["viewports"].append(row)

print("__MARKER__" + json.dumps(result) + "__MARKER__")
'''


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - silence per-request logging
        pass

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def serve(root: Path) -> tuple[http.server.ThreadingHTTPServer, int]:
    handler = functools.partial(QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def main() -> None:
    from benchmark_zoom import VIEWPORTS

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=Path, required=True, help="built JupyterLite dir")
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--tiles", type=Path, required=True)
    parser.add_argument("--skip-scan", action="store_true")
    parser.add_argument("--timeout-min", type=float, default=30)
    parser.add_argument("--chrome", default=shutil.which("google-chrome"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        os.symlink(args.site.resolve(), root / "lite")
        (root / "data").mkdir()
        os.symlink(args.parquet.resolve(), root / "data" / "nyc.parq")
        os.symlink(args.tiles.resolve(), root / "data" / "tiles")
        server, port = serve(root)

        code = (
            KERNEL_CODE.replace("__VIEWPORTS__", repr([list(v) for v in VIEWPORTS]))
            .replace("__SKIP_SCAN__", repr(args.skip_scan))
            .replace("__MARKER__", "@@" + "BENCHMARK@@")
        )
        url = (
            f"http://127.0.0.1:{port}/lite/repl/index.html"
            f"?kernel=xpython&toolbar=1&code={quote(code)}"
        )

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chrome, headless=True)
            page = browser.new_page()
            started = time.perf_counter()
            page.goto(url)
            deadline = time.time() + args.timeout_min * 60
            text = ""
            while time.time() < deadline:
                text = page.locator(".jp-OutputArea").all_inner_texts()
                text = "\n".join(text)
                if text.count(MARKER) >= 2 or "Traceback" in text or "Error" in text:
                    break
                page.wait_for_timeout(2_000)
            total = time.perf_counter() - started
            browser.close()
        server.shutdown()

    if text.count(MARKER) < 2:
        raise SystemExit(f"benchmark did not finish after {total:.0f}s:\n{text[-4000:]}")
    result = json.loads(text.split(MARKER)[1])
    result["page_total_s"] = total
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
