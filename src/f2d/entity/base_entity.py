import collections
import dataclasses
import os
import pickle
import time

import numpy as np
import tqdm
from matplotlib import patches

from f2d.utils import cv_utils, fov_utils
from f2d.utils import tqdm_file_reader as tqdm_fr
from f2d.utils import utils

# For visualization
_RING_WIDTH_FRAC = 0.1  # As fraction of the entity radius


@dataclasses.dataclass()
class Pose(object):
    x: float
    y: float
    theta: float


class Entity(object):
    """Presence entity such as robot, human, pet."""

    def __init__(self, fp_obj, radius=None, pose_config=None, name=None, **kwargs):
        assert radius is not None and radius >= 0
        assert pose_config is not None

        self.fp_obj = fp_obj
        self.radius = radius  # In meters.
        self.pose_config = pose_config
        # Optional args
        self.color = kwargs.get("color") or "red"
        self.seed = kwargs.get("seed", None)
        self.name = name or "entity"

        # Get randomizer for reproducibility.
        self.randomizer = utils.get_randomizer(self.seed)

        # Diameter in pixels (upper bound)
        diameter_p = int(np.ceil(2 * self.radius * self.fp_obj.ppm))
        self.diameter_p = max(diameter_p, 1)
        self.viz_arrow_length = self.diameter_p

        # Create mask for the areas which the entity can explore.
        self.explorable_mask = cv_utils.erode_mask(self.fp_obj.fp_mask, self.diameter_p)

    def reset(self, sim_ts):
        """Resets entity for a new episode in the simulation."""
        # Note that we do not use sim_ts in this default implementation.
        self.pose = self.get_init_pose(sim_ts)  # In mask coordinates

        meander_config = self.pose_config["meander_strategy"]
        self.lin_speed = meander_config["lin_speed"] * self.fp_obj.ppm
        self.ang_speed = meander_config["ang_speed"]

    def step(self, sim_delta, sim_ts_new):
        """Takes the next step within the episode."""
        # Note that we do not use the args in this default implementation.
        self.pose.theta += self.ang_speed * self.randomizer.randn()
        self.pose.theta = fov_utils.angle_in_range(self.pose.theta)
        new_x = self.pose.x + self.lin_speed * np.cos(np.deg2rad(self.pose.theta))
        new_y = self.pose.y + self.lin_speed * np.sin(np.deg2rad(self.pose.theta))
        if _is_explorable(new_x, new_y, self.explorable_mask):
            self.pose.x = new_x
            self.pose.y = new_y
        # TODO: Clip off to ensure we stay in the explorable mask.

    def get_init_pose(self, sim_ts):
        # Note that we do not use sim_ts in this default implementation.

        # Native pose is expressed in meters (as in the floorplan). We
        # wish to convert that to mask coordinates in which we operate
        # by default.
        native_pose = Pose(**self.pose_config["init_pose"])
        native_xy = (native_pose.x, native_pose.y)
        mask_xy = self.fp_obj.get_mask_xy(native_xy, dtype=np.float64)
        mask_pose = Pose(mask_xy[0], mask_xy[1], native_pose.theta)

        return mask_pose

    def draw(self, ax):
        """Draw entity as a ring with an arrow."""
        radius_p = self.diameter_p / 2
        ring = patches.Wedge(
            (self.pose.x, self.pose.y),
            radius_p,
            0,
            360,
            width=_RING_WIDTH_FRAC * radius_p,
            color=self.color,
            label=self.name,
        )

        arrow = _get_arrow(self.pose, self.viz_arrow_length, self.color)
        ax.add_patch(ring)
        ax.add_patch(arrow)
        self._viz_objects = {"ring": ring, "arrow": arrow}

    def draw_update(self, ax):
        # Update the ring
        center = (self.pose.x, self.pose.y)
        self._viz_objects["ring"].set(center=center)
        # Update the arrow (this requires deleting old arrow and
        # creating a new one).
        self._viz_objects["arrow"].remove()
        arrow = _get_arrow(self.pose, self.viz_arrow_length, self.color)
        ax.add_patch(arrow)
        self._viz_objects["arrow"] = arrow


class Robot(Entity):
    def __init__(
        self, fp_obj, radius=None, pose_config=None, fov_config=None, **kwargs
    ):
        super().__init__(fp_obj, radius=radius, pose_config=pose_config, **kwargs)
        # Longer arrow for robot to distinguish easily.
        self.viz_arrow_length = 3 * self.diameter_p
        # TODO: Load FOV params
        self.fov_config = fov_config
        self.fov_dist = self.fov_config["distance"] * self.fp_obj.ppm
        self.fov_angle = self.fov_config["angle"]
        self._build_fov_map(lazy=self.fov_config["lazy_compute"])

    def reset(self, sim_ts):
        super().reset(sim_ts)
        # Map from users' name to (estimated) pose.
        self.user_poses = None

    def step(self, sim_delta, sim_ts_new, users=None):
        super().step(sim_delta, sim_ts_new)

        if not users:
            self.user_poses = None
            return

        # Check if each users is in the robot's FoV, and if so, expose their actual
        # pose (ideal, no noise case).
        fov_poses = [u.pose for u in users.values()]
        poses_visibility = self.check_visibility(fov_poses)
        self.user_poses = collections.OrderedDict(
            [
                (un, pose if is_pt_vis else None)
                for un, pose, is_pt_vis in zip(
                    users.keys(), fov_poses, poses_visibility
                )
            ]
        )

    # TODO: Consider changing "visibility" to "identification" or such. Decide on
    # whether to operate on Point or on Pose objects.
    # TODO: Re-visit the split of duties between Robot and VisibilityPolygon
    # classes for computing FoV.
    def check_visibility(self, poses):
        """Returns list of bools indicating if poses are visible from the robot."""
        vp = self.get_vis_polygon()
        poses_visibility = [
            vp.point_in_fov(pose, self.pose, self.fov_dist, self.fov_angle)
            for pose in poses
        ]
        return poses_visibility

    def _build_fov_map(self, lazy=True):
        self.vp_map = {}  # Maps from (float_x, float_y) to Polygon instance.

        # Load FoV data optionally
        if self.fov_config["load_fov_file"] is not None:
            fpath = self.fov_config["load_fov_file"]
            print(f"Loading FOV data from {fpath}")
            start_time = time.time()
            if os.path.exists(fpath):
                with open(fpath, "rb") as handle:
                    total = os.path.getsize(fpath)
                    with tqdm_fr.TQDMBytesReader(handle, total=total) as tqdmfd:
                        up = pickle.Unpickler(tqdmfd)
                        self._vp_map = up.load()
                total_time = time.time() - start_time
                print("  Loading took {:.2f} seconds".format(total_time))
                loaded, total = len(self._vp_map), self.explorable_mask.sum()
                print("  Precomputed FOVs for {}/{} pixels.".format(loaded, total))
            else:
                if not lazy:
                    print(f"FOV data not found, precomputing and saving to {fpath}")
                    # TODO: Confirm that all required indices are present.
                    explorable_indices = np.where(self.explorable_mask)
                    for source in tqdm.tqdm(list(zip(*explorable_indices))):
                        x, y = float(source[1]), float(source[0])
                        # TODO: Do we need the bbox at all?
                        self._vp_map[(x, y)] = fov_utils.get_visibility_polygon(
                            x, y, [self.fp_polygon]
                        )
                    # Save vp map
                    with open(fpath, "wb") as handle:
                        pickle.dump(self._vp_map, handle)

        # TODO: Change floorplan class so fp_polygon already comes in this format,
        # obviating the transformation below.
        self.fp_polygon = fov_utils.Polygon(
            [fov_utils.Point(v[0], v[1]) for v in self.fp_obj.fp_polygon]
        )

        # Compute FoV for all indices
        if lazy:
            return
        # TODO: Confirm that all required indices are present.
        explorable_indices = np.where(self.explorable_mask)
        for source in zip(*explorable_indices):
            x, y = float(source[1]), float(source[0])
            # TODO: Do we need the bbox at all?
            self.vp_map[(x, y)] = fov_utils.get_visibility_polygon(
                x, y, [self.fp_polygon]
            )

    def get_vis_polygon(self, xy=None):
        """Returns the visibility polygon at the given (or the robot's) position."""
        if xy is None:
            xy = (float(int(self.pose.x)), float(int(self.pose.y)))
        if xy in self.vp_map:
            return self.vp_map[xy]
        # Lazy compute of FoV; it's not precomputed, do so now.
        vp = fov_utils.get_visibility_polygon(xy[0], xy[1], [self.fp_polygon])
        self.vp_map[xy] = vp
        return vp

    def draw(self, ax):
        super().draw(ax)
        self._draw_fov(ax)
        self._draw_visible_poses(ax)

    def _draw_fov(self, ax):
        "Draws FOV region on the plot."
        # x, y, fp = self.pose.x, self.pose.y, self.fp_obj
        # fov = fov_utils.get_visibility_polygon((x, y), fp.fp_mask.shape, fp.fp_polygon)
        fov = self.get_vis_polygon()
        fov_obj = _draw_pts(fov, ax, loop=True, mpl_str="--k")
        self._viz_objects["fov_obj"] = fov_obj

    def _draw_visible_poses(self, ax):
        "Draws visible points on the plot."
        if not self.user_poses:
            return
        pt_objs = []
        for pose in self.user_poses.values():
            if pose is None:
                continue
            pt = fov_utils.Point(pose.x, pose.y)
            pt_objs += pt.draw(ax)
        self._viz_objects["pt_objs"] = pt_objs

    def draw_update(self, ax):
        super().draw_update(ax)

        # Remove old FoV and draw new one
        fov_obj = self._viz_objects["fov_obj"].pop()
        fov_obj.remove()
        self._draw_fov(ax)

        # Remove old and add new points
        if self._viz_objects.get("pt_objs"):
            for pt_obj in self._viz_objects["pt_objs"]:
                pt_obj.remove()
            self._viz_objects["pt_objs"] = None
        self._draw_visible_poses(ax)


def _draw_pts(pts, ax, loop=False, mpl_str="k"):
    x_ = [pt[0] for pt in pts]
    y_ = [pt[1] for pt in pts]
    if loop:
        x_ += [pts[0][0]]
        y_ += [pts[0][1]]
    line = ax.plot(x_, y_, mpl_str)
    return line


def _get_arrow(pose, length, color):
    dx = length * np.cos(np.deg2rad(pose.theta))
    dy = length * np.sin(np.deg2rad(pose.theta))
    arrow = patches.Arrow(pose.x, pose.y, dx, dy, width=20.0, color=color)
    return arrow


def _is_explorable(x, y, explorable_mask):
    if x < 0 or y < 0:
        return False
    height, width = explorable_mask.shape
    if x >= width or y >= height:
        return False
    return explorable_mask[int(y), int(x)]
