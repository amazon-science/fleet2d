# HOW TO USE THIS?
# 1. Provide the path to the config and ensure that the robot has a radius of 0
# in the CONFIG below. That is, robot.base.radius=0
# 2. Provide the correct names under DATA_DIR for where the data is stored.
import os
import time

import dill as pickle  # To pickle lambdas
import numpy as np
import tqdm

from f2d.simulation import simulation
from f2d.utils import fov_utils

# Change path manually to the `f2d` package folder on your machine.
os.chdir(os.path.join(os.path.expanduser("~"), "Desktop/F2D/f2d"))
# CONFIG
config_path = "f2d/configs/config_simple.yaml"
# DATA_DIR
data_dir = "expt_data"
fname = "fov_data_658e5214673c7a4e25b458e56bdb6144_v2.pickle"  # v2


def get_fp_polygon(array_type_polygon):
    return fov_utils.Polygon([fov_utils.Point(v[0], v[1]) for v in array_type_polygon])


# TODO: Clean up and allow command line args.

# Check folder paths.
assert os.path.exists(data_dir)
assert os.path.exists(config_path)

# Config path
print(f"config_path: {config_path}")

# Create sim instance
sim = simulation.Simulation(config_path)
sim.reset()

# Write the visibility polygons
robot = sim.robot
explorable_indices = np.where(robot.explorable_mask)

start_time = time.time()
vp_map = {}
fp_polygon = get_fp_polygon(robot.fp_obj.fp_polygon)
for source in tqdm.tqdm(
    list(zip(*explorable_indices))
):  # 2677s (v0)  # 1635s (v1)  # s (v2)
    x, y = float(source[1]), float(source[0])
    vp_obj = fov_utils.VisibilityPolygonComputation(fov_utils.Point(x, y), [fp_polygon])
    vp_obj.build_visibility_polygon()
    vp_map[(x, y)] = vp_obj.vis_polygon
end_time = time.time()
print("\nTime elapsed: {:.2f}s".format(end_time - start_time))

# Save the polygon dictionary.
fpath = os.path.join(data_dir, fname)
with open(fpath, "wb") as handle:
    pickle.dump(vp_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
