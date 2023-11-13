import collections

from f2d.entity import base_entity
from f2d.utils import utils

# Supported user classes for initialization. Add class here after defining below.
USER_CLASSES = {
    "base": base_entity.Entity,
}


def get_users(users_config_dict, fp_obj, randomizer=None):
    """Returns multiple users in an ordered map."""
    users = collections.OrderedDict()
    cs_user_names = users_config_dict["names"]
    if not cs_user_names:  # Empty users
        return users
    user_names = cs_user_names.split(",")
    for un in user_names:
        if un not in users_config_dict:
            raise ValueError(f"Config for user {un} not found")
        # Insert seed into the user config based on global seed, if it
        # is not already present.
        seed = utils.get_seed(randomizer)
        users[un] = get_user(users_config_dict[un], fp_obj, user_name=un, seed=seed)
    return users


def get_user(user_config_dict, fp_obj, user_name="", seed=None):
    user_type = user_config_dict["type"]
    un = user_name  # Alias
    if user_type not in USER_CLASSES:
        raise ValueError(f"Undefined user type {user_type} for {un}")
    if user_type not in user_config_dict:
        raise ValueError(f'No data for "{user_type}" in config for {un}!')
    user_config_dict = user_config_dict[user_type]
    user_config_dict["name"] = user_name
    if user_config_dict.get("seed") is None:  # Use global seed if needed
        user_config_dict["seed"] = seed
    return USER_CLASSES[user_type](fp_obj, **user_config_dict)


# DELETE THIS !! Only for testing
if __name__ == "__main__":
    pass
