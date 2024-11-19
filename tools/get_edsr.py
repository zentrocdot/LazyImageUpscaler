#!/usr/bin/python3
#
# Helper script
# Version 0.0.0.3

# Import the Python modules
import fsspec
import pathlib
from pathlib import Path

# Set the repository data.
ORG = "Saafke"
REPO = "EDSR_Tensorflow"
FOLDER = "models/"

# Set the local path.
LOCAL_PATH = "/super-resolution"

# Set the paths.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
DESTINATION = Path(''.join([str(PARENT_PATH), LOCAL_PATH]))

# Make a flat copy.
fs = fsspec.filesystem("github", org=ORG, repo=REPO)
fs.get(fs.ls(FOLDER), DESTINATION.as_posix())

