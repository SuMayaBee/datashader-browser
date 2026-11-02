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
