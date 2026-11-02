from __future__ import annotations

import sys
import types

import pytest

from datashader_browser import site_url
from datashader_browser.browser import repair_tornado_stub


def test_site_url_requires_a_browser_kernel():
    with pytest.raises(RuntimeError, match="browser"):
        site_url("tiles")


def test_placeholder_tornado_is_replaced_by_the_real_package(monkeypatch):
    pytest.importorskip("tornado")
    placeholder = types.ModuleType("tornado")  # no __file__, like Xeus-Python's stub
    for name in [n for n in sys.modules if n == "tornado" or n.startswith("tornado.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "tornado", placeholder)

    repair_tornado_stub()

    import tornado.netutil  # noqa: F401  - fails against the placeholder

    assert sys.modules["tornado"] is not placeholder
    assert sys.modules["tornado"].__file__


def test_placeholder_is_kept_when_tornado_is_not_installed(monkeypatch):
    placeholder = types.ModuleType("tornado")
    monkeypatch.setitem(sys.modules, "tornado", placeholder)
    monkeypatch.setattr(sys, "path", [])
    for name in [n for n in sys.modules if n.startswith("tornado.")]:
        monkeypatch.delitem(sys.modules, name)

    repair_tornado_stub()

    assert sys.modules["tornado"] is placeholder


def test_pending_client_comms_are_opened_from_the_kernel(monkeypatch):
    from datashader_browser import browser

    comm = types.ModuleType("comm")  # the Xeus-Python kernel provides the real one
    monkeypatch.setitem(sys.modules, "comm", comm)

    opened = []

    class FakeKernelComm:
        def __init__(self, target_name, data):
            self.target_name = target_name

    class FakeClient:
        _comm = None

        def _handle_open(self, kernel_comm, msg):
            self._comm = kernel_comm
            opened.append(kernel_comm.target_name)

    comm.create_comm = FakeKernelComm
    already_open = FakeClient()
    already_open._comm = object()
    monkeypatch.setattr(
        browser, "_CLIENT_COMMS", {"plot-a": FakeClient(), "plot-b": already_open}
    )

    browser.open_pending_client_comms(None)

    assert opened == ["plot-a"]
    assert browser._CLIENT_COMMS == {}


def test_kernel_opened_comms_are_a_no_op_outside_the_browser():
    from datashader_browser import browser

    browser.install_kernel_opened_comms()  # must not patch or display anything natively
    pyviz_comms = pytest.importorskip("pyviz_comms")
    assert not getattr(pyviz_comms.JupyterCommJS, "_datashader_browser_patched", False)
