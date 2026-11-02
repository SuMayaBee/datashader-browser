"""Helpers for running inside a JupyterLite / Xeus-Python WebAssembly kernel."""

from __future__ import annotations

import sys
from typing import Any

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


# --- Kernel-opened client comms -------------------------------------------------
#
# HoloViews sends zoom/pan events from the page to Python over a comm that the
# *page* opens (pyviz_comms ``JupyterCommJS``).  Xeus-Python up to at least
# 0.19.0 cannot accept page-opened comms: its ``register_target`` callback
# passes a non-movable ``xcomm`` by value through pybind11 and every
# ``comm_open`` from the page fails with "return_value_policy = move, but type
# is neither movable nor copyable", so the events are dropped.
#
# Kernel-opened comms work (they carry plot updates and ipywidgets), and a comm
# is bidirectional once open.  The workaround therefore reverses who opens the
# event channel: the page only registers a comm target and queues messages,
# and Python opens a comm to that target right after the cell that displayed
# the plot has finished (IPython's ``post_run_cell`` event), by which time the
# plot output, and so the target, has reached the page.

_CLIENT_COMMS: dict[str, Any] = {}

_KERNEL_OPENED_COMMS_JS = r"""
(function () {
  var PyViz = window.PyViz;
  if (!PyViz || !PyViz.comm_manager || PyViz.comm_manager.__kernelOpenedClientComms) {
    return;
  }
  var manager = PyViz.comm_manager;
  manager.__kernelOpenedClientComms = true;
  manager.get_client_comm = function (plot_id, comm_id, msg_handler) {
    if (comm_id in PyViz.comms) {
      return PyViz.comms[comm_id];
    }
    var kernel = PyViz.kernels[plot_id];
    var queue = [];
    var proxy = {
      comm: null,
      onMsg: msg_handler,
      on_msg: function (handler) { this.onMsg = handler; },
      send: function (data, metadata, buffers, disposeOnDone) {
        if (this.comm) {
          return this.comm.send(data, metadata, buffers, disposeOnDone);
        }
        queue.push([data, metadata, buffers, disposeOnDone]);
      },
      close: function () { if (this.comm) { this.comm.close(); } }
    };
    PyViz.comms[comm_id] = proxy;
    if (!kernel) {
      console.warn("datashader-browser: no kernel registered for plot", plot_id);
      return proxy;
    }
    kernel.registerCommTarget(comm_id, function (comm) {
      proxy.comm = comm;
      comm.onMsg = function (msg) { if (proxy.onMsg) { proxy.onMsg(msg); } };
      queue.splice(0).forEach(function (args) { comm.send.apply(comm, args); });
    });
    return proxy;
  };
})();
"""


def open_pending_client_comms(*_: Any) -> None:
    """Open, from the kernel, every client comm a displayed plot is waiting for."""

    from comm import create_comm

    while _CLIENT_COMMS:
        comm_id, client = _CLIENT_COMMS.popitem()
        if getattr(client, "_comm", None) is None:
            client._handle_open(create_comm(target_name=comm_id, data={}), {})


def install_kernel_opened_comms() -> None:
    """Route HoloViews' page-to-kernel events over kernel-opened comms.

    Call after ``hv.extension(...)`` and before displaying a plot.  Safe to
    call repeatedly; it is a no-op outside the browser.
    """

    if not IN_BROWSER:
        return
    import pyviz_comms

    cls = pyviz_comms.JupyterCommJS
    if not getattr(cls, "_datashader_browser_patched", False):
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            _CLIENT_COMMS[self.id] = self

        cls.__init__ = __init__
        cls._datashader_browser_patched = True

        from IPython import get_ipython

        get_ipython().events.register("post_run_cell", open_pending_client_comms)

    from IPython.display import display

    display({"application/javascript": _KERNEL_OPENED_COMMS_JS}, raw=True)
