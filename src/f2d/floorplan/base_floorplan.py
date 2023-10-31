# Supported room types.
ROOM_TYPES = ["bedroom", "kitchen", "bathroom", "living_room", "dining_room", "hallway", "unknown"]  # All-capture class
KNOWN_ROOM_TYPES, UNKNOWN_ROOM_TYPE = ROOM_TYPES[:-1], ROOM_TYPES[-1]


class Floorplan(object):
    def __init__(self, **kwargs):
        # Creates attributes:
        # - ppm: number of Pixels Per Meter (e.g. ppm=100 implies each pixel=0.01m)
        # - fp_mask: binary mask over the floorplan, with true for explorable pixels
        # - fp_polygon: list of points forming FP.
        # - room_masks: room_masks[room_name] = binary mask for the room
        #     Example: room_masks['bedroom_0'] = binary_mask_for_bedroom_0
        #   The rooms are mutually exclusive and collectively exhaustive (MECE).

        # TODO(apaar): Perform checks and create room_props optionally.

        self.kwargs = kwargs

    def draw(self, ax):
        # Use fp_mask_obj.set_data to update this afterwards if needed.
        fp_mask_obj = ax.imshow(self.fp_mask, origin="lower")
        ax.set_xlim(left=0, right=self.fp_mask.shape[1])
        ax.set_ylim(bottom=0, top=self.fp_mask.shape[0])
        self._viz_objects = {"fp_mask_obj": fp_mask_obj}

    def draw_update(self, ax):
        # Typically we won't update the floorplan. This should be re-visited
        # if there are moving obstacles, etc.
        pass

    def get_fp_mask(self):
        self.fp_mask.copy()

    def get_room_mask(self, room_name, copy=True):
        room_mask = self.room_masks[room_name]
        return room_mask.copy() if copy else room_mask

    # Coordinate-pixel conversion helpers.

    def get_mask_xy(self, native_xy):
        pass

    # Room name/type/idx related helpers

    @staticmethod
    def get_room_name(room_type, room_idx):
        return "{}_{}".format(room_type, room_idx)

    @staticmethod
    def get_room_type(room_name):
        return "_".join(room_name.split("_")[:-1])

    @staticmethod
    def get_room_idx(room_name):
        return int(room_name.split("_")[-1])

    @staticmethod
    def get_next_room_name(room_type, all_room_names):
        next_idx = 0
        while Floorplan.get_room_name(room_type, next_idx) in all_room_names:
            next_idx += 1
        return Floorplan.get_room_name(room_type, next_idx)
