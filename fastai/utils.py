import os

def current_dir(folder=None):
    if folder:
        return os.path.join(os.getcwd(), folder)
    return os.getcwd()