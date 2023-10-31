import json
import os

import cv2
import numpy as np
from skimage import measure

from f2d.floorplan import base_floorplan


class HouseExpoFloorplan(base_floorplan.Floorplan):
    """Obtains a HouseExpo floorplan (https://github.com/TeaganLi/HouseExpo)."""

    def __init__(self, json_fpath=None, ppm=None, border_pad=None, **kwargs):
        assert json_fpath is not None and os.path.exists(json_fpath)
        assert ppm is not None and isinstance(ppm, int)
        assert border_pad is not None and isinstance(border_pad, int)

        self.json_fpath = json_fpath
        self.ppm = ppm
        self.border_pad = border_pad

        # Step 1: Read Json file
        with open(self.json_fpath) as json_file:
            self.json_data = json.load(json_file)

        # Step 2: Construct floorplan mask
        self._init_coord_change()
        # We do "+ 1" below in height/width computation since the min and max are
        # inclusive.
        # For example, if ppm=1,x_min=0,x_max=1, we need 2 pixels, where the center of
        # the two pixels are 0 and 1 (assuming no border padding).
        width = self._x_max - self._x_min + 1 + self.border_pad * 2
        height = self._y_max - self._y_min + 1 + self.border_pad * 2
        self.cnt_map = np.zeros((height, width), dtype=np.uint8)  # Note x goes second
        # verts[:, 0] = verts[:, 0] - x_min + self.border_pad
        # verts[:, 1] = verts[:, 1] - y_min + self.border_pad
        cv2.drawContours(self.cnt_map, [self._verts], 0, 255, -1)
        self.fp_mask = self.cnt_map > 0

        # Step 3: Construct room masks, including unknown mask.
        self.room_masks = {}

        # [3a]: Build masks for room based on the bbox, considering only the largest
        # connected component. This is heuristic used to combat that each room
        # is currently expressed as a single bbox.
        for tp in self.json_data["room_category"]:
            room_type = _get_room_type(tp)
            for bbox_tp in self.json_data["room_category"][tp]:
                bbox_xy_min, bbox_xy_max = bbox_tp[:2], bbox_tp[2:]  # Native coords
                bbox_xy_min = self.get_mask_xy(bbox_xy_min)  # Mask coords
                bbox_xy_max = self.get_mask_xy(bbox_xy_max)  # Mask coords
                bbox = [
                    np.max([bbox_xy_min[0], 0]),
                    np.max([bbox_xy_min[1], 0]),
                    np.min([bbox_xy_max[0], self.fp_mask.shape[1]]),
                    np.min([bbox_xy_max[1], self.fp_mask.shape[0]]),
                ]

                room_name = base_floorplan.Floorplan.get_next_room_name(room_type, self.room_masks.keys())
                room_mask = np.zeros_like(self.fp_mask)
                room_mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = True
                room_mask &= self.fp_mask
                room_mask = _get_largest_component(room_mask)  # If disjoint mask due to approximation
                self.room_masks[room_name] = room_mask

        # [3b]: Clean up the masks by removing overlapping regions, going from smallest
        # to biggest room.
        room_masks_ordered = sorted(list(self.room_masks.items()), key=lambda x: x[1].sum())
        room_masks_ordered = [list(e) for e in room_masks_ordered]
        # For each room i, remove its area from all larger rooms.
        for i, (room_name, room_mask) in enumerate(room_masks_ordered):
            for j in range(i + 1, len(room_masks_ordered)):
                room_masks_ordered[j][1] &= np.logical_not(room_mask)
        self.room_masks = dict(room_masks_ordered)

        # [3c]: Add a last unknown mask if needed.
        union_mask = sum([e.astype(np.int64) for e in self.room_masks.values()])
        assert union_mask.max() <= 1
        leftover_mask = (union_mask == 0) & self.fp_mask
        if leftover_mask.sum() > 0:
            unknown_room_name = base_floorplan.Floorplan.get_next_room_name(
                base_floorplan.UNKNOWN_ROOM_TYPE, self.room_masks.keys()
            )
            self.room_masks[unknown_room_name] = leftover_mask

        super().__init__(**kwargs)

    def _init_coord_change(self):
        self.json_data["verts"] = _smooth_json_verts(self.json_data["verts"])
        # We want to convert from native coordinates (in meters) to pixels.
        self._verts = (np.array(self.json_data["verts"]) * self.ppm).astype(np.int64)
        self._verts = _remove_duplicates(self._verts)  # Post-quantization step
        self._x_max, self._x_min = np.max(self._verts[:, 0]), np.min(self._verts[:, 0])
        self._y_max, self._y_min = np.max(self._verts[:, 1]), np.min(self._verts[:, 1])

        self._verts[:, 0] = self._verts[:, 0] - self._x_min + self.border_pad
        self._verts[:, 1] = self._verts[:, 1] - self._y_min + self.border_pad

        # Create and store the polygon.
        self.fp_polygon = [tuple(pt) for pt in self._verts.tolist()]

    def get_mask_xy(self, native_xy, dtype=np.int64):
        mask_xy = (np.array(native_xy) * self.ppm).astype(dtype)
        mask_xy[0] += self.border_pad - self._x_min
        mask_xy[1] += self.border_pad - self._y_min
        return mask_xy


def _remove_duplicates(arr_int):
    cur_set = set()
    arr_list = arr_int.tolist()
    new_arr = []
    for x in arr_list:
        tup_x = tuple(x)
        if tup_x in cur_set:
            continue
        cur_set.add(tup_x)
        new_arr.append(x)
    return np.array(new_arr, dtype=arr_int.dtype)


def _smooth_json_verts(json_verts, eps=0.03):
    def is_significant(abs_change):
        return abs_change > eps

    prev = json_verts[0]
    new_verts = [prev]
    for cur in json_verts[1:]:
        x_change, y_change = abs(prev[0] - cur[0]), abs(prev[1] - cur[1])
        if not is_significant(x_change) and not is_significant(y_change):
            continue
        # At least one change (or both) is significant
        v = [
            cur[0] if is_significant(x_change) else prev[0],
            cur[1] if is_significant(y_change) else prev[1],
        ]
        new_verts.append(v)
        prev = v
    return new_verts


def _get_room_type(he_room_type):
    room_type = he_room_type.lower()
    if room_type not in base_floorplan.KNOWN_ROOM_TYPES:
        if room_type == "toilet":
            assert "bathroom" == base_floorplan.KNOWN_ROOM_TYPES[2]
            room_type = base_floorplan.KNOWN_ROOM_TYPES[2]
        elif room_type == "guest_room":
            assert "bedroom" == base_floorplan.KNOWN_ROOM_TYPES[0]
            room_type = base_floorplan.KNOWN_ROOM_TYPES[0]
        else:
            room_type = base_floorplan.UNKNOWN_ROOM_TYPE
    return room_type


def _get_largest_component(mask):
    components = measure.label(mask)
    largest_component = components == np.argmax(np.bincount(components.flat, weights=mask.flat))
    return largest_component


if __name__ == "__main__":
    print("hello")
