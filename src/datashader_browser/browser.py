"""Helpers for running inside a JupyterLite / Xeus-Python WebAssembly kernel."""

from __future__ import annotations

import sys

IN_BROWSER = sys.platform == "emscripten"


def site_url(path: str = "") -> str:
    """Return an absolute URL for ``path`` relative to the hosting JupyterLite site.

    Static files deployed next to the site (for example a tile pyramid in
    ``<site>/tiles/nyc-taxi``) can then be opened without hard-coding the host:
    ``TilePyramid(site_url("tiles/nyc-taxi"))``.  Outside the browser this
    raises ``RuntimeError`` because there is no hosting page.
    """

    if not IN_BROWSER:
        raise RuntimeError("site_url() is only available in a browser kernel")
    import pyjs

    location = pyjs.js.self.location
    href = str(location.href)
    # The Xeus kernel worker is served from <site>/extensions/...; everything
    # before that segment is the site root (which may include a path prefix
    # such as /datashader-browser on GitHub Pages).
    marker = "/extensions/"
    root = href.split(marker)[0] if marker in href else str(location.origin)
    return f"{root.rstrip('/')}/{path.lstrip('/')}" if path else root


def repair_tornado_stub() -> None:
    """Replace Xeus-Python's placeholder ``tornado`` module with the real package.

    Xeus-Python registers an empty ``tornado`` module at startup.  When the
    real tornado is installed, the placeholder hides it, and Bokeh's server
    module (imported by Panel and therefore by HoloViews' Bokeh backend) fails
    with ``cannot import name 'netutil' from 'tornado' (unknown location)``.
    """

    module = sys.modules.get("tornado")
    if module is None or getattr(module, "__file__", None) is not None:
        return
    for name in [n for n in sys.modules if n == "tornado" or n.startswith("tornado.")]:
        del sys.modules[name]
    try:
        import tornado  # noqa: F401
    except ImportError:
        # No real tornado installed: restore the placeholder.
        sys.modules["tornado"] = module
