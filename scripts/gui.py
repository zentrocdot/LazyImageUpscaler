#!/usr/bin/python3
#
# LazyUpscalerGui
# Version 0.0.0.1

# Import the Python modules.
import os
import pathlib
import cv2
from cv2 import dnn_superres
import gradio as gr
import numpy as np
import torch
from PIL import Image
from datetime import datetime

# Set the paths.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
MODEL_PATH = ''.join([str(PARENT_PATH), "/resources"])
OUTPUTS_PATH = ''.join([str(PARENT_PATH), "/outputs"])

# Set a private variable.
_SortDir = False

# Create a private global dictionary.
_model_dict = {}

# Create a private global list.
_model_list = []

# *********************
# Function model_scan()
# *********************
def model_scan(model_dir: str, ext: list) -> (list, list):
    '''File scan for .pb models.'''
    global _model_dict
    subdirs, files = [], []
    for fn in os.scandir(model_dir):
        if fn.is_dir():
            subdirs.append(fn.path)
        if fn.is_file():
            if os.path.splitext(fn.name)[1].lower() in ext:
                _model_dict[fn.name] = fn.path
    for dirs in list(subdirs):
        sd, fn = model_scan(dirs, ext)
        subdirs.extend(sd)
        files.extend(fn)
    return subdirs, files

# *************************
# Function get_model_list()
# *************************
def get_model_list() -> list:
    '''Simple function for use with components.'''
    global _model_dict
    global _model_list
    _model_dict = {}
    _model_list = []
    model_scan(MODEL_PATH, [".pb"])
    _model_list = list(_model_dict.keys())
    _model_list.sort(reverse=_SortDir)
    return _model_list

# ************************
# Function upscale_image()
# ************************
def upscale_image(model_path, numpy_image):
    # Check if model_path is None.
    if model_path == None:
        model_path = _model_list[0]
    # Initialise the super resolution object.
    sr = dnn_superres.DnnSuperResImpl_create()
    # Read the selected model
    selected_model = _model_dict.get(model_path)
    sr.readModel(selected_model)
    # Set the model and scale the image up.
    model = model_path.split(".")[0]
    name = model.split("_")[0]
    name = name.lower()
    factor = model.split("_")[1]
    factor = int(factor.split("x")[1])
    sr.setModel(name, factor)
    # Upscale the image.
    upscaled = sr.upsample(numpy_image)
    # Return the upscaled image.
    return upscaled

# +++++++++++++
# Create web UI
# +++++++++++++
with gr.Blocks() as demo:
    # Define a loacal function.
    def download_image(image):
        date_time_jpg = datetime.now().strftime("/%Y-%m-%d_%H:%M:%S.%f.jpg")
        name = ''.join([str(OUTPUTS_PATH), date_time_jpg])
        cvImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(name, cvImg)
        return image
    # Create a row.
    with gr.Row():
        model_file = gr.Dropdown(choices=get_model_list(), value=None, label="Model File List")
        upscale_button = gr.Button(value="Upscale Image")
        download_button = gr.Button(value="Download Image")
    # Create a row.
    with gr.Row():
        im_input = gr.Image(width=512, height=512)
        im_output = gr.Image(width=512, height=512)
    upscale_button.click(
        fn=upscale_image,
        inputs=[model_file, im_input],
        outputs=[im_output]
    )
    download_button.click(
        fn=download_image,
        inputs=[im_output],
        outputs=[im_output]
    )

# Main script function.
def main():
    demo.launch()

# Execute as module as well as programme.
if __name__ == "__main__":
    demo.launch()
