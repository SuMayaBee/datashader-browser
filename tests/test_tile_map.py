import asyncio
import base64
import io
import json

import pandas as pd
import pytest
from PIL import Image

from datashader_browser import TileMap, build_pyramid


def test_overlay_bounds_orientation_and_cache(tmp_path):
    points = pd.DataFrame({"x": [0.5], "y": [3.5]})
    build_pyramid(
        [points], tmp_path, "x", "y", x_range=(0, 4), y_range=(0, 4), tile_size=4,
        max_level=0,
    )
    tile_map = TileMap(tmp_path)

    async def render():
        bounds = (-2, 2.1, 1.1, 7)
        overlay = await tile_map.render(bounds, 4, 4)
        assert overlay is not None
        assert tile_map.pyramid.tiles_read == 1
        assert await tile_map.render(bounds, 4, 4) == overlay
        assert tile_map.pyramid.tiles_read == 1
        assert await tile_map.render((4, 0, 8, 4), 4, 4) is None
        return overlay

    overlay = json.loads(json.dumps(asyncio.run(render())))
    assert overlay["bounds"] == [0, 2, 2, 4]
    assert overlay["level"] == 0
    assert (overlay["width"], overlay["height"]) == (2, 2)
    png = base64.b64decode(overlay["url"].split(",", 1)[1])
    with Image.open(io.BytesIO(png)) as image:
        assert image.size == (2, 2)
        assert image.getpixel((0, 0))[3] == 255
        assert image.getpixel((0, 1))[3] == 0


def test_overlay_rejects_invalid_viewport(tmp_path):
    points = pd.DataFrame({"x": [0.5], "y": [0.5]})
    pyramid = build_pyramid([points], tmp_path, "x", "y", x_range=(0, 1), y_range=(0, 1))
    tile_map = TileMap(pyramid)
    with pytest.raises(ValueError, match="bounds"):
        asyncio.run(tile_map.render((0, 0, 1, float("inf")), 100, 100))
    with pytest.raises(ValueError, match="positive integers"):
        asyncio.run(tile_map.render((0, 0, 1, 1), 0, 100))
