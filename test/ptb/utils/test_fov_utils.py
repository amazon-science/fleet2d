import f2d.utils.fov_utils


def _build_poly_segments(obstacles):
    """
    Original implementation of returning unique segments given a set of obstacles
    """
    unique_segments = []
    for poly in obstacles:
        poly_segments = poly.get_segments()
        for s in poly_segments:
            if s not in unique_segments:
                unique_segments.append(s)
    return unique_segments


def test_visibility_polygon_unique_segments():
    """
    Tests the unique-segments components of VisibilityPolygonComputation
    """
    robot_position = f2d.utils.fov_utils.Point(5, 15)

    box = f2d.utils.fov_utils.Polygon(
        [
            f2d.utils.fov_utils.Point(0, 0),
            f2d.utils.fov_utils.Point(0, 100),
            f2d.utils.fov_utils.Point(100, 100),
            f2d.utils.fov_utils.Point(100, 0),
        ]
    )

    # We're making two identical obstacles at the exact same location, thus generating duplicate segments
    obstacle1 = f2d.utils.fov_utils.Polygon.get_rectangle(20, 30, 40, 50)
    obstacle2 = f2d.utils.fov_utils.Polygon.get_rectangle(20, 30, 40, 50)
    map_obstacles1 = [box, obstacle1]
    map_obstacles2 = [box, obstacle1, obstacle2]

    orig_poly_segments1 = _build_poly_segments(map_obstacles1)
    orig_poly_segments2 = _build_poly_segments(map_obstacles2)

    vis_polygon1 = f2d.utils.fov_utils.VisibilityPolygonComputation(robot_position, map_obstacles1)
    vis_polygon2 = f2d.utils.fov_utils.VisibilityPolygonComputation(robot_position, map_obstacles2)

    # We expect the segments to be in sorted order and unique
    assert vis_polygon1.segments == sorted(orig_poly_segments1)
    assert vis_polygon2.segments == sorted(orig_poly_segments2)
    assert vis_polygon1.segments == vis_polygon2.segments
