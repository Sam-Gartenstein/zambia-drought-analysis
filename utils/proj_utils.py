import pyproj

def set_proj_data_directory():
    """
    Sets the PROJ data directory for pyproj to a predefined path.
    """
    path = "/Users/samuelgartenstein/anaconda3/envs/my_project_env/share/proj"
    pyproj.datadir.set_data_dir(path)
    print("PROJ data directory has been successfully set.")

