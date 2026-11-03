"""End-to-end check of the NYC tile demo in JupyterLab (headless Chrome).

Serves a built site under ``/datashader-browser/`` (the GitHub Pages prefix),
opens ``nyc-taxi-tiles.ipynb``, runs it, zooms with the mouse wheel and checks
that the kernel re-rendered from deeper pyramid levels.  The site must contain
``tiles/nyc-taxi`` (see the Pages workflow).

    uv run --with playwright python examples/browser_demo_check.py --site dist --shots .cache/shots
    uv run --with playwright python examples/browser_demo_check.py \
        --url https://sumayabee.github.io/datashader-browser --shots .cache/shots
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
from pathlib import Path
import re
import shutil
import tempfile
import threading
import time

PREFIX = "datashader-browser"


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    latency_s = 0.0

    def log_message(self, *args):
        pass

    def do_GET(self):
        # Simulate CDN round-trip latency for tile requests (local mode only).
        if self.latency_s and self.path.endswith(".npz"):
            time.sleep(self.latency_s)
        super().do_GET()

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass


def outputs(page) -> str:
    return "\n".join(page.locator(".jp-OutputArea-output").all_inner_texts())


def run_in_new_cell(page, code: str, marker: str, timeout_s: float = 60) -> str:
    """Append a code cell with the keyboard (as a user would) and run it."""

    page.locator(".jp-Notebook .jp-Cell .jp-InputPrompt").last.click()
    page.keyboard.press("Escape")
    page.keyboard.press("b")
    page.keyboard.press("Enter")
    page.keyboard.insert_text(code)  # typing would trigger auto-indent
    page.keyboard.press("Shift+Enter")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        match = re.search(marker + r"[^\n]*", outputs(page))
        if match:
            return match.group(0)
        page.wait_for_timeout(500)
    raise TimeoutError(f"no {marker!r} output:\n{outputs(page)[-2000:]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--site", type=Path, help="local built site to serve")
    source.add_argument("--url", help="already hosted site, e.g. the GitHub Pages URL")
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--chrome", default=shutil.which("google-chrome"))
    parser.add_argument(
        "--latency-ms", type=float, default=0,
        help="delay each tile response when serving --site (simulates a CDN)",
    )
    args = parser.parse_args()
    args.shots.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        server = None
        if args.url:
            base = args.url.rstrip("/")
        else:
            QuietHandler.latency_s = args.latency_ms / 1000
            os.symlink(args.site.resolve(), Path(tmp) / PREFIX)
            handler = functools.partial(QuietHandler, directory=tmp)
            server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            base = f"http://127.0.0.1:{server.server_address[1]}/{PREFIX}"
        url = f"{base}/lab/index.html?path=nyc-taxi-tiles.ipynb"

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=args.chrome, headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 1000})
            errors: list[str] = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            started = time.perf_counter()
            page.goto(url)
            page.wait_for_selector(".jp-Notebook .jp-Cell", timeout=120_000)
            page.wait_for_timeout(3_000)
            page.locator(".lm-MenuBar-itemLabel", has_text="Run").first.click()
            page.locator(".lm-Menu-itemLabel", has_text="Run All Cells").first.click()
            try:
                page.wait_for_function(
                    "[...document.querySelectorAll('.jp-OutputArea-output')]"
                    ".some(e => /pickups|Error/.test(e.innerText))",
                    timeout=300_000,
                )
            except Exception:
                page.screenshot(path=str(args.shots / "failed.png"), full_page=True)
                raise SystemExit(f"notebook produced no output:\n{outputs(page)[-3000:]}\n{errors[:5]}")
            try:
                page.wait_for_selector(".jp-OutputArea canvas", timeout=120_000, state="attached")
            except Exception:
                page.screenshot(path=str(args.shots / "failed.png"), full_page=True)
                raise SystemExit(f"no plot rendered:\n{outputs(page)[-3000:]}\n{errors[:5]}")
            page.wait_for_timeout(3_000)
            first_render = time.perf_counter() - started
            text = outputs(page)
            if "Error" in text:
                raise SystemExit(f"notebook failed:\n{text[-3000:]}")
            plot = page.locator(".jp-OutputArea canvas").first
            plot.scroll_into_view_if_needed()
            page.screenshot(path=str(args.shots / "1-overview.png"))
            before = run_in_new_cell(
                page,
                "import time\n"
                "CALLS = []\n"
                "_aggregate = pyramid.aggregate\n"
                "def _timed(*a, **k):\n"
                "    t = time.time(); r = _aggregate(*a, **k)\n"
                "    CALLS.append((round(t, 2), round(time.time() - t, 3), r.attrs['pyramid_level']))\n"
                "    return r\n"
                "pyramid.aggregate = _timed\n"
                "print('BEFORE', pyramid.tiles_read)",
                "BEFORE",
            )

            box = plot.bounding_box()
            # Midtown Manhattan (x ~ -8.236e6, y ~ 4.976e6) in the default extent.
            page.mouse.move(box["x"] + box["width"] * 0.43, box["y"] + box["height"] * 0.54)
            zoom_started = time.perf_counter()
            zoom_wall = time.time()
            # A few wheel clicks, as a person would; each click zooms ~10%.
            for _ in range(6):
                page.mouse.wheel(0, -200)
                page.wait_for_timeout(300)
            page.wait_for_timeout(6_000)
            plot.scroll_into_view_if_needed()
            page.screenshot(path=str(args.shots / "2-zoomed.png"))
            calls_cell_started = time.time()
            after = run_in_new_cell(page, "print('AFTER', pyramid.tiles_read, 'CALLS', CALLS)", "AFTER")
            level = run_in_new_cell(
                page,
                "print('LEVEL', max(k[0] for k in pyramid._cache))",
                "LEVEL",
            )
            browser.close()
        if server is not None:
            server.shutdown()

    print(f"first render after page load: {first_render:.1f}s")
    print(f"before zoom: {before}; after zoom: {after}; deepest {level}")
    print(f"zoom interaction window: {time.perf_counter() - zoom_started:.1f}s")
    print(f"zoom began at {zoom_wall:.2f}; inspection cell sent at {calls_cell_started:.2f}")
    print("page errors:", errors[:5] or "none")
    tiles_before = int(before.split()[1])
    tiles_after = int(after.split()[1])
    if tiles_after <= tiles_before:
        raise SystemExit("FAIL: zooming did not fetch new tiles (no kernel re-render)")
    print("PASS: zoom triggered kernel re-renders from deeper pyramid levels")


if __name__ == "__main__":
    main()
