from f2d.entity import base_entity
from f2d.utils import utils

# Supported robot classes for initialization. Add class here after defining below.
ROBOT_CLASSES = {
    "base": base_entity.Robot,
}


def get_robot(robot_config_dict, fp_obj, randomizer=None):
    robot_type = robot_config_dict["type"]
    if robot_type not in ROBOT_CLASSES:
        raise ValueError(f"Undefined robot type {robot_type}")
    if robot_type not in robot_config_dict:
        raise ValueError(f'No data for "{robot_type}" in config!')
    robot_config_dict = robot_config_dict[robot_type]
    robot_config_dict["name"] = "Robot"
    if robot_config_dict.get("seed") is None:
        seed = utils.get_seed(randomizer)
        robot_config_dict["seed"] = seed
    return ROBOT_CLASSES[robot_type](fp_obj, **robot_config_dict)


