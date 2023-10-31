import os
import pathlib


def get_main_dir(calling_file_path, calling_file_level):
    """Returns the absolute path of the main top-level directory."""
    assert os.path.isabs(calling_file_path)
    calling_file_path = os.path.realpath(calling_file_path)
    return str(pathlib.Path(calling_file_path).parents[calling_file_level])


def get_abs_path(fpath, calling_file_path, calling_file_level):
    """Returns absolute path whether `fpath` is rel or abs.

    Example:
                   fpath = 'f2d/configs/config_simple.yaml'
       calling_file_path = '~/Desktop/Fleet2D/f2d/floorplan/floorplan_creation.py'
      calling_file_level = 2
           returned path = '~/Desktop/Fleet2D/f2d/configs/config_simple.yaml'
    """
    if os.path.isabs(fpath):
        return fpath
    # `fpath` is relative path. By convention, it is relative to the main dir.
    calling_file_path = os.path.realpath(calling_file_path)
    main_dir_abs_path = get_main_dir(calling_file_path, calling_file_level)
    return os.path.join(main_dir_abs_path, fpath)
