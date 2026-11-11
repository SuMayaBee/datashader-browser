"""Render tile pyramids as image overlays for standalone web maps."""

from __future__ import annotations

import asyncio
import base64
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence, TypedDict

from .pyramid import TilePyramid


class MapImage(TypedDict):
    """A JSON-compatible PNG overlay and its bounds in pyramid coordinates."""

    url: str
    bounds: tuple[float, float, float, float]
    width: int
    height: int
    level: int
    under_resolved: bool


class TileMap:
    """Render viewports from a static tile pyramid, reusing its tile cache."""

    def __init__(
        self,
        location: str | Path | TilePyramid,
        *,
        agg: Any = None,
        cmap: Sequence[str] = ("lightblue", "darkblue"),
        how: str = "eq_hist",
        span: tuple[float, float] | None = None,
    ) -> None:
        self.pyramid = location if isinstance(location, TilePyramid) else TilePyramid(location)
        self._agg = agg
        self._cmap = list(cmap)
        self._how = how
        self._span = span

    async def render(
        self, bounds: Sequence[float], width: int, height: int
    ) -> MapImage | None:
        """Render (xmin, ymin, xmax, ymax) at the requested screen resolution.

        Bounds use the pyramid's coordinate system. The PNG is north-up and
        must be placed using the returned bin-aligned bounds. Returns None
        for a viewport entirely outside the pyramid. Requires Pillow.
        """
        xmin, ymin, xmax, ymax = map(float, bounds)
        if not all(map(math.isfinite, (xmin, ymin, xmax, ymax))) or xmin >= xmax or ymin >= ymax:
            raise ValueError("bounds must be finite and ordered as (xmin, ymin, xmax, ymax)")
        if type(width) is not int or type(height) is not int or min(width, height) < 1:
            raise ValueError("width and height must be positive integers")
        left, right = self.pyramid.x_range
        bottom, top = self.pyramid.y_range
        if xmax <= left or xmin >= right or ymax <= bottom or ymin >= top:
            return None

        from datashader.transfer_functions import shade

        aggregate = await self.pyramid.aggregate_async(
            (xmin, xmax), (ymin, ymax), width=width, height=height, agg=self._agg
        )
        image = shade(aggregate, cmap=self._cmap, how=self._how, span=self._span)
        png = base64.b64encode(image.to_bytesio().getvalue()).decode("ascii")
        left, right = aggregate.attrs["x_range"]
        bottom, top = aggregate.attrs["y_range"]
        return MapImage(
            url=f"data:image/png;base64,{png}",
            bounds=(left, bottom, right, top),
            width=image.shape[1],
            height=image.shape[0],
            level=aggregate.attrs["pyramid_level"],
            under_resolved=aggregate.attrs["under_resolved"],
        )


def serve_tile_map(tile_map: TileMap, *, port: Any = None) -> Callable[[], None]:
    """Serve map render requests over a pyjs Web Worker or MessagePort.

    Send an object with type "datashader:render", id, bounds, width and
    height. Replies carry type "datashader:result", the matching id, and
    either an image (possibly null) or an error message.
    """
    import pyjs

    target = pyjs.js.self if port is None else port
    loop = asyncio.get_event_loop()
    tasks: set[asyncio.Task] = set()

    async def respond(request):
        reply = {"type": "datashader:result", "id": request.get("id")}
        try:
            reply["image"] = await tile_map.render(
                request["bounds"], request["width"], request["height"]
            )
        except Exception as exc:
            reply["error"] = str(exc)
        target.postMessage(pyjs.js.JSON.parse(json.dumps(reply)))

    def on_request(payload):
        request = json.loads(str(payload))
        task = loop.create_task(respond(request))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    callback, handle = pyjs.create_callable(on_request)
    listener = pyjs.js.Function(
        "callback",
        """
        return ({ data }) => {
            if (data?.type === "datashader:render") callback(JSON.stringify(data));
        };
        """,
    )(callback)
    target.addEventListener("message", listener)
    if port is not None:
        target.start()

    def disconnect() -> None:
        target.removeEventListener("message", listener)
        handle.delete()

    return disconnect
