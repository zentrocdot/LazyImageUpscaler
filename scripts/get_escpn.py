#!/usr/bin/python3
#
# Helper script
# Version 0.0.0.2

# Import the Python modules
import fsspec
import pathlib
from pathlib import Path

# Set the repository data.
ORG = "fannymonori"
REPO = "TF-ESPCN"
FOLDER = "export/"

# Set the path variables.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
DESTINATION = Path(''.join([str(PARENT_PATH), "/resources"]))

# Make a flat copy of the folder.
fs = fsspec.filesystem("github", org=ORG, repo=REPO)
fs.get(fs.ls(FOLDER), DESTINATION.as_posix())


