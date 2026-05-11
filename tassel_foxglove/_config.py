"""Panel type mapping and Foxglove layout JSON generation.

This is the shared core imported by both the layout generator and the launcher.
"""

# panel type → Foxglove panel component name
FOXGLOVE_PANEL = {
    "image":            "Image",
    "compressed_image": "Image",
    "trajectory":       "ThreeDee",
    "path":             "ThreeDee",
    "marker":           "ThreeDee",
    "landmark":         "ThreeDee",
    "pointcloud":       "ThreeDee",
    "error":            "Plot",
}

# panel types that get double width in the grid
WIDE_PANELS = {"trajectory", "path"}


def _fix_topic(t: str) -> str:
    """Normalise a ROS topic name for Foxglove layout JSON."""
    return "/" + t.lstrip("/")


def _image_config(p: dict) -> dict:
    """Build Image panel config with full Foxglove Image panel support."""
    cfg: dict = {}

    # ── image source ──────────────────────────────────────────────────
    if p["type"] == "compressed_image":
        cfg["imageMode"] = {
            "mode": "topic",
            "compressedImageTopic": _fix_topic(p["topic"]),
        }
    else:
        cfg["imageMode"] = {
            "mode": "topic",
            "imageTopic": _fix_topic(p["topic"]),
        }

    # ── calibration ──────────────────────────────────────────────────
    if "calibration_topic" in p:
        ct = p["calibration_topic"]
        cfg["calibrationTopic"] = _fix_topic(ct) if ct else ""

    # ── annotations (2-D ImageMarker overlay) ────────────────────────
    annotations = p.get("annotations", [])
    if annotations:
        cfg["annotationTopics"] = [_fix_topic(a) for a in annotations]

    # ── sync timestamps ──────────────────────────────────────────────
    if "sync_timestamps" in p:
        cfg["syncTimestamps"] = p["sync_timestamps"]

    # ── display transforms ───────────────────────────────────────────
    if p.get("flip_horizontal"):
        cfg["flipHorizontal"] = True
    if p.get("flip_vertical"):
        cfg["flipVertical"] = True
    if p.get("rotation") in (0, 90, 180, 270):
        cfg["rotation"] = p["rotation"]

    # ── colour mode (single-channel images) ──────────────────────────
    if p.get("color_mode"):
        cfg["colorMode"] = p["color_mode"]
    if p.get("color_map"):
        cfg["colorMap"] = p["color_map"]
    if p.get("gradient"):
        cfg["gradient"] = p["gradient"]
    if "value_min" in p:
        cfg["valueMin"] = p["value_min"]
    if "value_max" in p:
        cfg["valueMax"] = p["value_max"]

    # ── image overlays ───────────────────────────────────────────────
    overlays = p.get("overlays", [])
    if overlays:
        cfg["imageOverlays"] = []
        for o in overlays:
            entry = {
                "topic": _fix_topic(o["topic"]),
                "opacity": o.get("opacity", 0.5),
                "blendMode": o.get("blend_mode", "Alpha"),
            }
            if "pixel_alpha_white_transparent" in o:
                entry["pixelAlphaWhiteIsTransparent"] = o["pixel_alpha_white_transparent"]
            if "schema" in o:
                entry["schema"] = o["schema"]
            cfg["imageOverlays"].append(entry)

    # ── scene / render settings ──────────────────────────────────────
    if "render_stats" in p:
        cfg["renderStats"] = p["render_stats"]
    if p.get("background"):
        cfg["background"] = p["background"]
    if p.get("label_scale"):
        cfg["labelScale"] = p["label_scale"]
    if p.get("ignore_collada_up_axis") is not None:
        cfg["ignoreColladaUpAxis"] = p["ignore_collada_up_axis"]
    if p.get("mesh_up_axis"):
        cfg["meshUpAxis"] = p["mesh_up_axis"]

    # ── 3-D topics superposed on the image ───────────────────────────
    scene_topics = p.get("scene_topics", [])
    if scene_topics:
        cfg["topics"] = {_fix_topic(t): {"visible": True} for t in scene_topics}

    # ── publish (cursor click / hover) ───────────────────────────────
    if p.get("click_topic"):
        cfg["clickTopic"] = _fix_topic(p["click_topic"])
    if p.get("hover_topic"):
        cfg["hoverTopic"] = _fix_topic(p["hover_topic"])

    return cfg


def panel_config(p: dict) -> dict:
    """Build the Foxglove per-panel config dict from a YAML panel entry."""
    ptype = p["type"]
    cfg: dict = {}

    if ptype in ("image", "compressed_image"):
        return _image_config(p)

    elif ptype == "trajectory":
        topics: dict[str, dict] = {}
        odom = _fix_topic(p.get("odom_topic", ""))
        if odom != "/":
            topics[odom] = {"visible": True}
        path_topic = p.get("path_topic", "")
        if path_topic:
            entry: dict = {"visible": True}
            if p.get("line_width"):
                entry["lineWidth"] = p["line_width"]
            if p.get("color"):
                entry["color"] = p["color"]
            topics[_fix_topic(path_topic)] = entry
        cfg["topics"] = topics
        cfg["displayTF"] = p.get("display_tf", True)

    elif ptype == "path":
        entry = {"visible": True}
        if p.get("line_width"):
            entry["lineWidth"] = p["line_width"]
        if p.get("color"):
            entry["color"] = p["color"]
        topics = {_fix_topic(p["topic"]): entry}
        cfg["topics"] = topics
        cfg["displayTF"] = p.get("display_tf", True)

    elif ptype in ("marker", "landmark", "pointcloud"):
        entry = {"visible": True}
        if p.get("point_size"):
            entry["pointSize"] = p["point_size"]
        if p.get("color"):
            entry["color"] = p["color"]
        topics = {_fix_topic(p["topic"]): entry}
        cfg["topics"] = topics
        cfg["displayTF"] = True

    elif ptype == "error":
        cfg["paths"] = [{
            "topic": _fix_topic(p["topic"]),
            "lineSize": p.get("line_size", 2),
            "lineColor": p.get("line_color", "#ff6347"),
        }]

    return cfg


def generate_layout(panels: list, grid_cols: int = 2) -> dict:
    """Return a Foxglove-compatible layout dict."""

    config_by_id: dict = {}
    for p in panels:
        pid = p["name"]
        config_by_id[pid] = {
            "type": FOXGLOVE_PANEL[p["type"]],
            "config": panel_config(p),
        }

    items: list = []
    x, y = 0, 0
    for p in panels:
        w = 2 if p["type"] in WIDE_PANELS else 1
        if x + w > grid_cols:
            x = 0
            y += 1
        items.append({"id": p["name"], "x": x, "y": y, "width": w, "height": 1})
        x += w
        if x >= grid_cols:
            x = 0
            y += 1

    return {
        "version": 1,
        "layout": {
            "configById": config_by_id,
            "layout": {
                "type": "Grid",
                "items": items,
            },
        },
    }
