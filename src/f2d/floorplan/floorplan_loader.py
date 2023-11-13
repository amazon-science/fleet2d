from f2d.floorplan import he_floorplan

# Supported floorplan classes for initialization. Add class here after defining below.
FLOORPLAN_CLASSES = {"he": he_floorplan.HouseExpoFloorplan}


def get_floorplan(fp_config_dict):
    fp_type = fp_config_dict["type"]
    if fp_type not in FLOORPLAN_CLASSES:
        raise ValueError(f"Undefined floorplan type: f{fp_type}")
    if fp_type not in fp_config_dict:
        raise ValueError(f'No data for "{fp_type}" in input config!')
    print(f"Loading {fp_type} floorplan: {fp_config_dict[fp_type]}")
    fp_config_dict = fp_config_dict[fp_type]
    return FLOORPLAN_CLASSES[fp_type](**fp_config_dict)
