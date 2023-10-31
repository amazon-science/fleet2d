import collections
import functools
import itertools
import math

import numpy as np
import sortedcontainers
from matplotlib import pyplot as plt

from f2d.utils.fov_utils_cpp import get_closest_point as get_closest_point_cpp

_TOL = 1e-6  # Tolerance for what is treated as approximately zero.

"""
Geometric classes
"""

# Define basic geometric objects (e.g. point, segment, polygon).
_Point = collections.namedtuple("_Point", "x y")


class Point(_Point):
    """A point located at (x, y)."""

    def draw(self, ax, mpl_str="rx"):
        return draw_pts([self], ax, loop=False, mpl_str=mpl_str, arrows=False)

    # Support arithmetic operations on points. Add here as needed.
    def __add__(self, other_pt):
        return Point(self.x + other_pt.x, self.y + other_pt.y)

    def __sub__(self, other_pt):
        return Point(self.x - other_pt.x, self.y - other_pt.y)

    def __mul__(self, val):
        return Point(val * self.x, val * self.y)

    def __truediv__(self, val):
        return Point(self.x / val, self.y / val)


def is_same_point(point_a, point_b):
    return math.isclose(point_a.x, point_b.x, abs_tol=0.1) and math.isclose(point_a.y, point_b.y, abs_tol=0.1)


_Segment = collections.namedtuple("Segment", "a b")


class Segment(_Segment):
    """A line segment from Point a to Point b."""

    def draw(self, ax, mpl_str="k", arrows=True, arrow_width=0.01):
        return draw_pts(self, ax, loop=False, mpl_str=mpl_str, arrows=arrows, arrow_width=arrow_width)

    def get_points(self):
        return [self.a, self.b]

    def get_length(self):
        return get_distance(self.a, self.b)


class Polygon(list):
    """Polygon represented as a list of vertices (each is a Point instance).

    - The polygon is assumed closed and the edge from the last to the first vertex
    is implicitly constructed (i.e. it should not be provided).
    - The order of the vertices provided in the list is important. No shape
    assumptions on the polygon are made (e.g. convexity), as shape is obtained by
    connecting the segments of consecutive vertices.
    """

    def get_segments(self):
        """Returns segments forming the polygon, including the closing segment."""
        return [Segment(p_a, p_b) for p_a, p_b in zip(self, self[1:] + [self[0]])]

    def draw(self, ax, mpl_str="k", arrows=True, arrow_width=0.01):
        return draw_pts(self, ax, loop=True, mpl_str=mpl_str, arrows=arrows, arrow_width=arrow_width)

    @staticmethod
    def get_rectangle(x_min, x_max, y_min, y_max):
        """Convenience method to construct a simple rectangle Polygon instance."""
        return Polygon([Point(x_min, y_min), Point(x_min, y_max), Point(x_max, y_max), Point(x_max, y_min)])


"""
Helper functions
"""


def draw_pts(pts, ax, loop=False, mpl_str="k", arrows=False, arrow_width=0.01):
    """Draws the list of Point instances on a Matplotlib axis and returns obj."""
    # TODO: mpl_str does not work for arrows, fix this.
    if ax is None:
        unused_fig, ax = plt.subplots()
    x = [pt[0] for pt in pts]
    y = [pt[1] for pt in pts]
    if loop:
        x += [pts[0][0]]
        y += [pts[0][1]]
    x = np.array(x)
    y = np.array(y)
    if arrows:
        quiver = ax.quiver(
            x[:-1], y[:-1], x[1:] - x[:-1], y[1:] - y[:-1], scale_units="xy", angles="xy", scale=1, width=arrow_width
        )
        return quiver
    lines = ax.plot(x, y, mpl_str)
    return lines


def is_almost_zero(x):
    return abs(x) < _TOL


def angle_in_range(a):
    return (a + 180) % 360 - 180  # Range [-180, 180)


def angle_delta(a1, a2):
    return angle_in_range(a1 - a2)  # Range [-180, 180)


def get_theta(vector):
    """Returns theta (in [-180, 180) degrees) for the slope of a vector."""
    degrees = np.rad2deg(math.atan2(vector.y, vector.x))
    return angle_in_range(degrees)


def sort_ccw(points, origin=None, clockwise=False, ret_theta=False):
    """Sorts list of Point instances in counter-clockwise order w.r.t. an origin.

    Args:
      points: list of Point instances.
      origin: Point where origin is located. Defaults to (0, 0). Note that the
        order in which points appear depends on where you are looking from, i.e.
        where the origin is located.
      clockwise: whether to return in clockwise order.
      ret_theta: whether to also return the slope of the vector from origin to
        each point (in degrees).

    Returns:
      list of points in desired order (CCW or CW), and if ret_theta=True, then
      list of (theta, point).
    """
    if origin is None:
        origin = Point(0, 0)
    theta_points = sorted([(get_theta(pt - origin), pt) for pt in points])
    if clockwise:
        theta_points = theta_points[::-1]
    if ret_theta:
        return theta_points
    return [x[1] for x in theta_points]  # Discard theta before returning


def get_nearest_point(segment, other_point):
    """
      Returns nearest point of segment from other_point.

    Args:
      segment: a line segment, instance of Segment.
      other_point: a Point instance.

    Returns:
      nearest_point: to the other_point located on the segment.
    """
    seg_vector = np.array(segment.b - segment.a, dtype=float)
    seg_length = np.linalg.norm(seg_vector)
    if is_almost_zero(seg_length):
        return segment.a
    seg_unit = seg_vector / seg_length

    other_vector = np.array(other_point - segment.a, dtype=float)

    dot_pdt = np.dot(other_vector, seg_unit)
    if dot_pdt >= seg_length:
        nearest_point = segment.b
    elif dot_pdt <= 0:
        nearest_point = segment.a
    else:  # In strict interior of the segment
        delta = Point(float(dot_pdt * seg_unit[0]), float(dot_pdt * seg_unit[1]))
        nearest_point = segment.a + delta

    return nearest_point


def get_distance(pt_a, pt_b):
    return float(np.linalg.norm(np.array(pt_a) - np.array(pt_b)))


def get_distance_vec(pts_array, pt_or_pts):
    return np.linalg.norm(pts_array - np.array(pt_or_pts), axis=-1)


def get_ray(origin, theta, length):
    """Returns segment starting at origin with slope theta and of given length."""
    dest = Point(origin.x + length * np.cos(np.deg2rad(theta)), origin.y + length * np.sin(np.deg2rad(theta)))
    return Segment(origin, dest)


def get_bbox(seg):
    """Returns bbox containing the segment."""
    return [Point(min(seg.a.x, seg.b.x), min(seg.a.y, seg.b.y)), Point(max(seg.a.x, seg.b.x), max(seg.a.y, seg.b.y))]


def bbox_overlap_py(seg_12, seg_34):
    """Returns whether the bboxes of the two segments overlap."""
    min_pt_12, max_pt_12 = get_bbox(seg_12)
    min_pt_34, max_pt_34 = get_bbox(seg_34)
    if max_pt_34.x < min_pt_12.x or min_pt_34.x > max_pt_12.x:  # No overlap on x
        return False
    if max_pt_34.y < min_pt_12.y or min_pt_34.y > max_pt_12.y:  # No overlap on y
        return False
    return True


def on_same_side_py(seg_12, seg_34):
    """Checks pts 1 and 2 are on same side of seg_34 and vice-versa."""

    # If this function returns True, then the segments don't intersect.
    # If this function returns False, the segments might intersect (eg. collinear)
    def get_score(d_ba, d_ca):
        return d_ba.y * d_ca.x - d_ba.x * d_ca.y  # Eq of line

    p1, p2, p3, p4 = seg_12.a, seg_12.b, seg_34.a, seg_34.b
    # Check pts 3, 4 against seg_12
    delta_21, delta_31, delta_41 = p2 - p1, p3 - p1, p4 - p1
    score3, score4 = get_score(delta_21, delta_31), get_score(delta_21, delta_41)
    if score3 * score4 > 0:
        return True
    # Check pts 1, 2 against seg_34
    delta_43, delta_13, delta_23 = p4 - p3, p1 - p3, p2 - p3
    score1, score2 = get_score(delta_43, delta_13), get_score(delta_43, delta_23)
    if score1 * score2 > 0:
        return True
    return False


def get_visibility_polygon(source_x, source_y, obstacles):
    """Convenience method to get vis polygon directly."""
    vp = VisibilityPolygonComputation(Point(source_x, source_y), obstacles)
    vp.build_visibility_polygon()
    return vp.vis_polygon


def limit_id_range(min_angle, max_angle, interval_dict):
    """Limits the range of an interval dict with keys theta in [-180,180)."""
    interval = (min_angle, max_angle)
    return interval_dict[interval]  # dict restricted to given range


"""
FOV classes
"""


# TODO: Consider refactoring this so an instance does not have a fixed source.
# This will allow using a single instance across all points on the canvas.
class VisibilityPolygonComputation(object):
    """Polygon of visible area given the obstacles and the light source/robot."""

    _EPS_ANGLE = 1e-3  # Epsilon for angle computations.

    def __init__(self, source, obstacles):
        """Initializes the VisibilityPolygonComputation instance.

        Args:
          source: Point representing the light source (or the robot).
          obstacles: list of Polygon instances, each of whose walls represents an
            obstacle. If obstacle contains the source, the obstacle's complement
            forms an obstacle. The source is not allowed to lie on the boundary
            of any obstacle. It is assumed that there's at least one polygon which
            contains the source, so the visibility polygon is finite.
        """
        self.source = source
        self.obstacles = obstacles

        # Vertices
        #  Get all vertices in CCW order with their thetas.
        vertices = list(functools.reduce(lambda x, y: x + y, obstacles))
        theta_vertices = sort_ccw(vertices, source, ret_theta=True)
        self.vertices = [av[1] for av in theta_vertices]
        self.vertex_to_theta = {av[1]: av[0] for av in theta_vertices}

        # Segments
        self._build_segment_list()
        self._build_segment_attrs()  # Pre-compute some attrs to avoid recomputing.

        # Validate problem is well-defined
        self._validate_problem_instance()

        self.vis_polygon = None  # This will be computed when explicitly asked for.

    def _validate_problem_instance(self):
        if self.source in self.vertices:  # Ensure vertex and source don't overlap
            raise ValueError("source found in the polygon vertices")
        for s in self.segments:
            if is_almost_zero(self.segment_to_attrs[s]["np_distance"]):
                raise ValueError("source {} too close to {}".format(self.source, s))

    def _build_segment_list(self):
        """Collect all unique segments."""
        segments = sortedcontainers.SortedSet()
        for poly in self.obstacles:
            poly_segments = poly.get_segments()
            segments.update(poly_segments)
        self.segments = list(segments)

    def _build_segment_attrs(self):
        """Builds a cache of attributes for each segment."""
        source, segments = self.source, self.segments
        #  Get closest point and dist of each segment to the source
        segment_to_attrs = {
            s: {"nearest_point": get_nearest_point(s, source), "idx": idx} for idx, s in enumerate(segments)
        }
        for s, attrs in segment_to_attrs.items():
            # Compute distance to the nearest point
            segment_to_attrs[s]["np_distance"] = get_distance(source, attrs["nearest_point"])
        self.segment_to_attrs = segment_to_attrs

    def build_visibility_polygon(self):
        """Builds the visibility polygon."""
        polygon_vertices = []
        polygon_thetas = []
        max_dist = self._get_longest_ray_upper_bound()
        prev_seg_indices = [None, None]
        for v in self.vertices:
            theta = self.vertex_to_theta[v]
            # We throw out 3 rays from the source approximately towards the vertex -
            # one towards and one each on either side of the vertex in CCW fashion.
            angles = [theta - self._EPS_ANGLE, theta, theta + self._EPS_ANGLE]
            angles = map(angle_in_range, angles)
            for angle in angles:
                ray = get_ray(self.source, angle, max_dist)
                closest_point, cur_seg_idx = self._get_closest_pt(ray, prev_seg_indices[-1])
                if cur_seg_idx == prev_seg_indices[-1] == prev_seg_indices[-2]:
                    polygon_vertices.pop()
                    polygon_thetas.pop()
                else:
                    prev_seg_indices = [prev_seg_indices[-1], cur_seg_idx]
                polygon_vertices.append(closest_point)
                polygon_thetas.append(angle)
        self.vis_polygon = VisibilityPolygon(self.source, polygon_vertices, thetas=polygon_thetas)

    def _get_longest_ray_upper_bound(self):
        """Returns upper bound on the distance between source and the vertices."""
        max_dist = max([get_distance(self.source, v) for v in self.vertices])
        return max_dist + 1  # Strict upper bound

    def _get_closest_pt(self, ray_from_src, prev_seg_idx):
        # To validate C++ version matches Python:
        # return self._validate_get_closest_pt(ray_from_src, prev_seg_idx)

        # To toggle which version (Python or C++), comment/uncomment the lines below
        # return self._get_closest_pt_py(ray_from_src, prev_seg_idx)
        return self._get_closest_pt_cpp(ray_from_src, prev_seg_idx)

    def _validate_get_closest_pt(self, ray_from_src, prev_seg_idx):
        closest_pt_py, best_seg_idx_py = self._get_closest_pt_py(ray_from_src, prev_seg_idx)
        closest_pt_cpp, best_seg_idx_cpp = self._get_closest_pt_cpp(ray_from_src, prev_seg_idx)
        if best_seg_idx_py != best_seg_idx_cpp:
            print("Index {} vs {}".format(best_seg_idx_py, best_seg_idx_cpp))
        assert best_seg_idx_py == best_seg_idx_cpp
        if not is_same_point(closest_pt_py, closest_pt_cpp):
            print("Point {} vs {}".format(closest_pt_py, closest_pt_cpp))
        assert is_same_point(closest_pt_py, closest_pt_cpp)

        return closest_pt_cpp, best_seg_idx_cpp

    # TODO: This uses segment_to_attrs currently, fix that.
    def _get_closest_pt_py(self, ray_from_src, prev_seg_idx):
        """Returns the closest intersection pt along ray_from_src and polygon segment"""
        # We prioritize prev_seg in searching through the segments for intersection.
        # For clarity, one may ignore this optimization altogether.
        if prev_seg_idx is None:
            segments = self.segments
        else:
            segments = itertools.chain(
                [self.segments[prev_seg_idx]], self.segments[:prev_seg_idx], self.segments[prev_seg_idx + 1 :]
            )

        # We search through all segments (no particular order required) to find the
        # closest intersection point.
        closest_point = ray_from_src.b
        max_dist = get_distance(self.source, ray_from_src.b)
        best_seg = None
        for seg in segments:
            # We wish to compute the intersection of the ray with each segment. For
            # efficiency, we short-circuit this computation, performing computation
            # only when certain of intersection.
            if self.segment_to_attrs[seg]["np_distance"] >= max_dist:
                continue
            if not bbox_overlap_py(ray_from_src, seg):
                continue
            if on_same_side_py(ray_from_src, seg):
                continue
            int_pt = self._get_intersection(ray_from_src, seg)
            if int_pt is None:
                continue
            dist_from_source = get_distance(int_pt, self.source)
            if dist_from_source >= max_dist:
                continue
            closest_point, max_dist, best_seg = int_pt, dist_from_source, seg
        return closest_point, self.segment_to_attrs[best_seg]["idx"]

    def _get_closest_pt_cpp(self, ray_from_src, prev_seg_idx):
        """Returns the closest intersection pt along ray_from_src and polygon segment"""
        # Total segments is around 700 elements
        max_dist = get_distance(self.source, ray_from_src.b)
        closest_pt_cpp, best_seg_idx_cpp = get_closest_point_cpp(
            self.source,
            ray_from_src,
            self.segments,
            max_dist,
            self.segment_to_attrs,
            0 if prev_seg_idx is None else prev_seg_idx,
        )
        if closest_pt_cpp is None:
            print("No best segment could be found for {}".format(self.source))
            closest_pt = ray_from_src.b
        else:
            closest_pt = Point(closest_pt_cpp.x, closest_pt_cpp.y)

        return closest_pt, best_seg_idx_cpp

    def _get_intersection(self, source_seg, other_seg):
        """Returns intersection of two segments (None if they don't intersect)."""

        # We get to this function once we know the segments do not lie on the same
        # side of each other. Thus, they are collinear or they intersect.
        def get_seg_vars(seg):
            return (seg.a.y - seg.b.y), (seg.b.x - seg.a.x), (seg.b.x * seg.a.y - seg.a.x * seg.b.y)

        source_v = get_seg_vars(source_seg)
        other_v = get_seg_vars(other_seg)
        determinant = source_v[0] * other_v[1] - source_v[1] * other_v[0]
        if determinant == 0:  # They are collinear (possibly non-overlapping)
            dist_other_a = get_distance(self.source, other_seg.a)
            dist_other_b = get_distance(self.source, other_seg.b)
            other_pt = other_seg.a if dist_other_a < dist_other_b else other_seg.b
            other_dist = min(dist_other_a, dist_other_b)
            if other_dist > get_distance(self.source, source_seg.b):
                return None  # No intersection (non-overlapping collinear segments)
            return other_pt
        dx = source_v[2] * other_v[1] - source_v[1] * other_v[2]
        dy = source_v[0] * other_v[2] - source_v[2] * other_v[0]
        intersection_pt = Point(dx / determinant, dy / determinant)
        return intersection_pt


def get_sign(c, a, b):
    """Function that return sign indicating which side of the line the pt lies."""
    return np.sign((b.y - a.y) * (c.x - a.x) - (b.x - a.x) * (c.y - a.y))


def get_sign_from_attrs(segment_attrs, user_pt):
    """Return the sign from the segment attributes, accounting for different file formats"""
    if "get_sign" in segment_attrs:
        return segment_attrs["get_sign"](user_pt)
    else:
        return get_sign(user_pt, segment_attrs["a"], segment_attrs["b"])


class VisibilityPolygon(Polygon):
    def __init__(self, source, vertices, thetas=None):
        """Builds a VisibilityPolygon instance from vertices and angles."""
        # TODO: Assert that vertices and thetas are sorted CCW starting from -180
        # to 180.
        self.source = source
        # All thetas are [-180, 180) but sometimes they may start at, e.g. 179.999
        # after standardizing -180.001 to that range. We re-sort all vertices and
        # thetas by thetas. This guarantees thetas start at >=-180.
        thetas = thetas or self._compute_thetas(vertices)
        thetas, vertices = list(zip(*sorted(zip(thetas, vertices))))

        super().__init__(vertices)
        self.thetas = thetas
        self._build_cache()

    def draw(self, ax, mpl_str="k", **unused_kwargs):
        if ax is None:
            unused_fig, ax = plt.subplots()
        lines = super().draw(ax, mpl_str=mpl_str, arrows=False)
        lines += self.source.draw(ax)
        return lines

    def _compute_thetas(self, vertices):
        return [get_theta(pt - self.source) for pt in vertices]

    def _build_cache(self):
        # Dict mapping from theta intervals to segment attributes.
        cache = {}
        # The segment crossing our range boundary at 180 can cause issues when
        # trying to fetch the segment for a theta. We explicitly take care of this.
        thetas_start = itertools.chain([-180], self.thetas)
        thetas_end = itertools.chain(self.thetas, [180])
        vertices_start = itertools.chain([self[-1]], self)
        vertices_end = itertools.chain(self, [self[0]])

        # Populate the cache with segment attributes.
        iterable = zip(thetas_start, thetas_end, vertices_start, vertices_end)
        for theta_min, theta_max, vertex_min, vertex_max in iterable:
            theta_interval = (theta_min, theta_max)  # [x, y)
            if theta_interval.empty:
                # Skip segments directed at the source.
                # TODO: Re-evaluate if this is reasonable as we will miss some edges.
                continue
            cache[theta_interval] = self._get_segment_attrs(vertex_min, vertex_max)

        # Validate that the cache keys cover [-180, 180)
        cache_support = functools.reduce(lambda x, y: x | y, cache.keys())
        assert cache_support == (-180, 180)

        self._cache = cache

    def _get_segment_attrs(self, a, b):
        """Returns segment attrs to cache for efficient FoV queries."""
        # The segment attributes are used in the FoV algorithm for efficiency. They
        # are utilized after it is established that a point satisfies the FoV dist
        # and angle criteria (i.e. before we use info from the visibility polygon).
        dist_a, dist_b = get_distance(self.source, a), get_distance(self.source, b)
        segment_attrs = {
            # Points <= this distance are automatically accepted.
            "accept_distance": min(dist_a, dist_b),
            # Points > this distance are automatically rejected.
            "reject_distance": max(dist_a, dist_b),
            # Leftover points must be on same side of the line segment as source.
            # We pre-compute the value for the source and check if the point's value
            # has the same sign.
            "source_sign": get_sign(self.source, a, b),
            "a": a,
            "b": b,
        }
        return segment_attrs

    # TODO: Check carefully if we should check that robot_pose.x/y is near source.
    # This implementation assumes robot is located exactly at the source.
    def point_in_fov(self, user_pose, robot_pose, fov_dist, fov_angle):
        """Checks if user is in the FoV of the robot, given their poses."""
        user_pt = Point(user_pose.x, user_pose.y)
        dist_from_source = get_distance(user_pt, self.source)
        if fov_dist < dist_from_source:
            return False
        pt_theta = get_theta(user_pt - self.source)
        if abs(angle_delta(pt_theta, robot_pose.theta)) > fov_angle / 2:
            return False
        # Check with appropriate segment
        segment_attrs = self._cache[pt_theta]
        if dist_from_source <= segment_attrs["accept_distance"]:
            return True
        if dist_from_source > segment_attrs["reject_distance"]:
            return False
        # We check if both the source and the point lie on the same side of the line
        return segment_attrs["source_sign"] == get_sign_from_attrs(segment_attrs, user_pt)

    def points_in_vp(self, points_arr, fov_dist):
        """Returns indices of points in VP and within 360-FoV in interval dict.

        Args:
          points_arr: array of shape nx2 containing the (x,y) coords of n points.
          fov_dist: how far the robot can see, i.e. the radius of the FoV.

        Returns:
          theta_to_point_indices: dict mapping from theta to list
            containing indices of points that are within the VP and within fov_dist
            of the source. Here theta is the angle of the vector going from source
            to the point.
        """
        theta_to_point_indices = {}
        indices = np.arange(len(points_arr))

        # Distance check
        dist = get_distance_vec(points_arr, self.source)
        indices = indices[dist <= fov_dist]
        if len(indices) == 0:
            return theta_to_point_indices

        # VP check
        # TODO: Consider vectorizing this for speed.
        for pt_idx in indices.tolist():
            pt = Point(*points_arr[pt_idx])
            theta = get_theta(pt - self.source)
            segment = self._cache[theta]
            if dist[pt_idx] > segment["reject_distance"]:
                continue
            if dist[pt_idx] <= segment["accept_distance"] or get_sign_from_attrs(segment, pt):
                # Add index of point to the list corresponding to theta.
                theta_to_point_indices.setdefault(theta, [])
                theta_to_point_indices[theta].append(pt_idx)

        return theta_to_point_indices


# Basic visualization of FoV
if __name__ == "__main__":
    # Problem inputs
    # source = Point(100, 45)
    source = Point(110, 35)

    box = Polygon([Point(0, 0), Point(0, 100), Point(200, 100), Point(200, 0)])

    rect1 = Polygon.get_rectangle(60, 100, 60, 80)
    rect2 = Polygon.get_rectangle(25, 75, 30, 50)
    rect3 = Polygon.get_rectangle(25, 75, 10, 20)
    obstacles = [box, rect1, rect2, rect3]  # These are the obstacles

    vp = VisibilityPolygonComputation(source, obstacles)

    # Plot
    _, axs = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    ax = axs[0]
    ax.set_aspect("equal")
    draw_pts(vp.vertices, ax, mpl_str=".")
    source.draw(ax)
    draw_pts(vp.vertices, ax, arrows=True, arrow_width=0.005)

    ax = axs[1]
    for p in obstacles:
        p.draw(ax, arrow_width=0.002)
    source.draw(ax)
    vp.build_visibility_polygon()
    vp.vis_polygon.draw(ax, arrows=False, mpl_str="--")

    plt.show()
