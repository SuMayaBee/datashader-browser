"""End-to-end check of the NYC tile demo in JupyterLab (headless Chrome).

Serves a built site under ``/datashader-browser/`` (the GitHub Pages prefix),
opens ``nyc-taxi-tiles.ipynb``, runs it, zooms with the mouse wheel and checks
that the kernel re-rendered from deeper pyramid levels.  The site must contain
``tiles/nyc-taxi`` (see the Pages workflow).

    uv run --with playwright python examples/browser_demo_check.py --site dist --shots .cache/shots
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
    def log_message(self, *args):
        pass

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
    page.keyboard.type(code)
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
    parser.add_argument("--site", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--chrome", default=shutil.which("google-chrome"))
    args = parser.parse_args()
    args.shots.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        os.symlink(args.site.resolve(), Path(tmp) / PREFIX)
        handler = functools.partial(QuietHandler, directory=tmp)
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        url = (
            f"http://127.0.0.1:{server.server_address[1]}/{PREFIX}/lab/index.html"
            "?path=nyc-taxi-tiles.ipynb"
        )

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
            page.wait_for_function(
                "[...document.querySelectorAll('.jp-OutputArea-output')]"
                ".some(e => /pickups|Error/.test(e.innerText))",
                timeout=300_000,
            )
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
            before = run_in_new_cell(page, "print('TILES', pyramid.tiles_read)", "TILES")

            box = plot.bounding_box()
            # Midtown Manhattan sits left of centre in the default extent.
            page.mouse.move(box["x"] + box["width"] * 0.36, box["y"] + box["height"] * 0.58)
            zoom_started = time.perf_counter()
            for _ in range(10):
                page.mouse.wheel(0, -300)
                page.wait_for_timeout(250)
            page.wait_for_timeout(4_000)
            plot.scroll_into_view_if_needed()
            page.screenshot(path=str(args.shots / "2-zoomed.png"))
            after = run_in_new_cell(page, "print('TILES', pyramid.tiles_read)", "TILES")
            level = run_in_new_cell(
                page,
                "print('LEVEL', max(k[0] for k in pyramid._cache))",
                "LEVEL",
            )
            browser.close()
        server.shutdown()

    print(f"first render after page load: {first_render:.1f}s")
    print(f"before zoom: {before}; after zoom: {after}; deepest {level}")
    print(f"zoom interaction window: {time.perf_counter() - zoom_started:.1f}s")
    print("page errors:", errors[:5] or "none")
    tiles_before = int(before.split()[1])
    tiles_after = int(after.split()[1])
    if tiles_after <= tiles_before:
        raise SystemExit("FAIL: zooming did not fetch new tiles (no kernel re-render)")
    print("PASS: zoom triggered kernel re-renders from deeper pyramid levels")


if __name__ == "__main__":
    main()
