"""Panel type mapping and Foxglove layout JSON generation.

Generates layout JSON in Foxglove SDK format (SplitContainer tree). This is the
format Foxglove Studio internally stores and recognises for layout import.
"""

# panel type → Foxglove SDK panel component name
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

# panel types that get double proportion in a row
WIDE_PANELS = {"trajectory", "path"}


def _fix_topic(t: str) -> str:
    """Normalise a ROS topic name for Foxglove layout JSON."""
    return "/" + t.lstrip("/")


def _image_config(p: dict) -> dict:
    image_mode: dict = {
        "imageTopic": _fix_topic(p["topic"]),
    }

    calib = p.get("calibration_topic", "")
    if calib:
        image_mode["calibrationTopic"] = _fix_topic(calib)

    if "sync_timestamps" in p:
        image_mode["synchronize"] = p["sync_timestamps"]

    if p.get("rotation") in (0, 90, 180, 270):
        image_mode["rotation"] = p["rotation"]
    if p.get("flip_horizontal"):
        image_mode["flipHorizontal"] = True
    if p.get("flip_vertical"):
        image_mode["flipVertical"] = True
    if p.get("color_mode"):
        image_mode["colorMode"] = p["color_mode"]
    if p.get("color_map"):
        image_mode["colorMap"] = p["color_map"]
    if p.get("gradient"):
        image_mode["gradient"] = p["gradient"]
    if "value_min" in p:
        image_mode["minValue"] = p["value_min"]
    if "value_max" in p:
        image_mode["maxValue"] = p["value_max"]

    return {"imageMode": image_mode}


def _3d_config(p: dict) -> dict:
    ptype = p["type"]
    topics: dict[str, dict] = {}

    if ptype in ("marker", "landmark", "pointcloud"):
        entry: dict = {"visible": True}
        if p.get("point_size"):
            entry["pointSize"] = p["point_size"]
        if p.get("color"):
            entry["color"] = p["color"]
        topics[_fix_topic(p["topic"])] = entry

    elif ptype == "path":
        entry: dict = {"visible": True}
        if p.get("line_width"):
            entry["lineWidth"] = p["line_width"]
        if p.get("color"):
            entry["color"] = p["color"]
        topics[_fix_topic(p["topic"])] = entry

    elif ptype == "trajectory":
        topics = {}
        odom = p.get("odom_topic", "")
        if odom:
            topics[_fix_topic(odom)] = {"visible": True}
        path_topic = p.get("path_topic", "")
        if path_topic:
            entry: dict = {"visible": True}
            if p.get("line_width"):
                entry["lineWidth"] = p["line_width"]
            if p.get("color"):
                entry["color"] = p["color"]
            topics[_fix_topic(path_topic)] = entry

    return {"topics": topics}


def _plot_config(p: dict) -> dict:
    return {
        "paths": [{
            "value": _fix_topic(p["topic"]),
            "lineSize": p.get("line_size", 2),
            "color": p.get("line_color", "#ff6347"),
        }]
    }


def _panel_node(p: dict) -> dict:
    """Build a Foxglove SDK panel node from a YAML panel entry."""
    ptype = p["type"]
    if ptype in ("image", "compressed_image"):
        config = _image_config(p)
    elif ptype in ("trajectory", "path", "marker", "landmark", "pointcloud"):
        config = _3d_config(p)
    elif ptype == "error":
        config = _plot_config(p)
    else:
        raise ValueError(f"Unknown panel type: {ptype}")

    return {
        "type": "panel",
        "panelType": FOXGLOVE_PANEL[ptype],
        "config": config,
        "version": 1,
    }


def _build_split_tree(panels: list, grid_cols: int) -> dict:
    """Arrange panels into a SplitContainer tree mimicking the grid layout.

    Rows become items of a column-direction split. Within each row, if there
    are multiple panels they are wrapped in a row-direction split, otherwise
    the panel sits directly in the column split.
    """
    rows: list[list[dict]] = []
    current_row: list[tuple[dict, int]] = []
    x = 0

    for p in panels:
        w = 2 if p["type"] in WIDE_PANELS else 1
        if x + w > grid_cols and current_row:
            rows.append(current_row)
            current_row = []
            x = 0
        current_row.append((p, w))
        x += w
    if current_row:
        rows.append(current_row)

    if not rows:
        return {}

    row_contents: list[dict] = []
    for row_panels in rows:
        if len(row_panels) == 1:
            row_contents.append(_panel_node(row_panels[0][0]))
        else:
            row_items = []
            for p, w in row_panels:
                row_items.append({
                    "proportion": w,
                    "content": _panel_node(p),
                })
            row_contents.append({
                "type": "split",
                "direction": "row",
                "items": row_items,
            })

    if len(row_contents) == 1:
        return row_contents[0]

    return {
        "type": "split",
        "direction": "column",
        "items": [{"proportion": 1, "content": r} for r in row_contents],
    }


def panel_config(p: dict) -> dict:
    """Build the Foxglove per-panel config dict from a YAML panel entry.

    Deprecated: kept for backwards compatibility. Use generate_layout() instead.
    """
    ptype = p["type"]
    if ptype in ("image", "compressed_image"):
        return _image_config(p)
    elif ptype in ("trajectory", "path", "marker", "landmark", "pointcloud"):
        return _3d_config(p)
    elif ptype == "error":
        return _plot_config(p)
    raise ValueError(f"Unknown panel type: {ptype}")


def generate_layout(panels: list, grid_cols: int = 2) -> dict:
    """Return a Foxglove-compatible layout dict in SDK SplitContainer format.

    The returned dict can be serialised to JSON and written to Foxglove's
    layout storage directory, or saved as a .foxglove.json file for import.
    """
    content = _build_split_tree(panels, grid_cols)
    return {
        "version": 1,
        "content": content,
    }
