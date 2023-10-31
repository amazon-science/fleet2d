# """Script to generate the cache of Visibility Polygons (VP)."""

# import tqdm
# import time
# import pickle
# import os

# import numpy as np


# start_time = time.time()

# vp_map = {}
# fp_polygon = get_fp_polygon(robot.fp_obj.fp_polygon)
# for source in tqdm.tqdm(list(zip(*explorable_indices))):  # 2677s (old)  # 1635s (new)
# # for source in tqdm.tqdm(list(zip(*explorable_indices))[:1000]):  # 30s (old)  # 23s (new)
# # for source in [[robot.pose.y, robot.pose.x]]:  # Checking for viz here
#   x, y = float(source[1]),  float(source[0])
#   # TODO: Do we need the bbox at all?
#   vp_obj = fov_utils.VisibilityPolygonComputation(fov_utils.Point(x, y), [fp_polygon])
#   vp_obj.build_visibility_polygon()
#   vp_map[(x, y)] = vp_obj.vis_polygon
#   # break

# end_time = time.time()
# print('\nTime elapsed: {:.2f}s'.format(end_time-start_time))


# example_vp = vp_map[(x, y)]
# print(type(example_vp))


# data_dir = '/home/user/Desktop/Fleet2D/f2d/expt_data'
# assert os.path.exists(data_dir)

# fname = 'fov_data_658e5214673c7a4e25b458e56bdb6144_v1.pickle'  # New
# # fname = 'fov_data_658e5214673c7a4e25b458e56bdb6144.pickle'  # Old

# fpath = os.path.join(data_dir, fname)
# with open(fpath, 'wb') as handle:
#     pickle.dump(vp_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # with open(fpath, 'rb') as handle:
# #     loaded = pickle.load(handle)
