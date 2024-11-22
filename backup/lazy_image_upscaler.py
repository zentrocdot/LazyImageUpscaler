#!/usr/bin/python3
#
# LazyImageUpscaler
# Version 0.0.0.3

# Import the Python modules.
import os
import sys
import pathlib
import cv2
from cv2 import dnn_superres
import gradio as gr
import numpy as np
import torch
from PIL import Image
from datetime import datetime

# Set the path variables.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
MODEL_PATH = ''.join([str(PARENT_PATH), "/super-resolution"])
OUTPUTS_PATH = ''.join([str(PARENT_PATH), "/outputs"])

# Initialize different interpolation methods
inter_methods_dict = {
    "cv2.INTER_NEAREST": cv2.INTER_NEAREST,
    "cv2.INTER_LINEAR": cv2.INTER_LINEAR,
    "cv2.INTER_AREA": cv2.INTER_AREA,
    "cv2.INTER_CUBIC": cv2.INTER_CUBIC,
    "cv2.INTER_LANCZOS4": cv2.INTER_LANCZOS4
}
inter_methods_list = list(inter_methods_dict.keys())

# Set the scale factors.
scale_factors = [2,3,4,5,6,7,8]

# Set a private variable.
_SortDir = False

# Create a private global dictionary.
_model_dict = {}

# Create a private global list.
_model_list = []

# ***********************
# Function flip_image_h()
# ***********************
def flip_image_h(x):
    return np.fliplr(x)

# ***********************
# Function flip_image_v()
# ***********************
def flip_image_v(x):
    return np.flipud(x)

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
    # Define the global variables.
    global _model_list, _model_dict
    # Initialise model dict and model list.
    _model_dict = {}
    _model_list = []
    # Scan for all models.
    model_scan(MODEL_PATH, [".pb"])
    # Prepare model list.
    _model_list = list(_model_dict.keys())
    _model_list.sort(reverse=_SortDir)
    # Return the model list.
    return _model_list

# ******************************
# Function upscale_image_model()
# ******************************
def upscale_image_model(model_name, numpy_image):
    # Check if the model_path is None.
    if model_name == None:
        # Use the first entry of the model list.
        model_name = _model_list[0]
    # Initialise the super resolution object.
    sr = dnn_superres.DnnSuperResImpl_create()
    # Get path to model.
    model_path = _model_dict.get(model_name)
    # Read the selected model
    sr.readModel(model_path)
    # Get upscale parameter from model name.
    model = model_name.split(".")[0]
    str_name = model.split("_")[0].lower()
    str_factor = model.split("_")[1]
    int_factor = int(str_factor.split("x")[1])
    # Set the model and scale the image up.
    sr.setModel(str_name, int_factor)
    # Upscale the numpy image.
    upscaled = sr.upsample(numpy_image)
    # Return the upscaled image.
    return upscaled

# *******************************
# Function upscale_image_opencv()
# *******************************
def upscale_image_opencv(input_name, factor, numpy_image):
    model_name = inter_methods_dict.get(input_name)
    # Check if the model_path is None.
    if model_name == None:
        # Use the first entry of the model list.
        model_name = 0
    if factor == None:
        factor = 2
    print(input_name, "->", model_name)
    # Upscale the numpy image.
    val = numpy_image.shape[1] * factor
    ratio = val / numpy_image.shape[1]
    dimension = (val, int(numpy_image.shape[0] * ratio))
    upscaled = cv2.resize(numpy_image, dimension, interpolation=model_name)
    # Return the upscaled image.
    return upscaled

def sharpen_image(old_img, kernel_name):
    '''Sharpen image'''
    if kernel_name == None:
        kernel = np.array([[-0.0023, -0.0432, -0.0023], [-0.0432, 1.182, -0.0432], [-0.0023, -0.0432, -0.0023]])
    if kernel_name == "Kernel 0":
        kernel = np.array([[-0.0023, -0.0432, -0.0023], [-0.0432, 1.182, -0.0432], [-0.0023, -0.0432, -0.0023]])
    elif kernel_name == "Kernel 1":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_name == "Kernel 2":
        kernel = np.array([[-0.11, -0.11, -0.11], [-0.11, 1.89, -0.11], [-0.11, -0.11, -0.11]])
    elif kernel_name == "Kernel 3":
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    elif kernel_name == "Kernel 4":
        kernel = np.array([[-1.1,-1.1,-1.1], [-1.1,10,-1.1], [-1.1,-1.1,-1.1]])
    elif kernel_name == "Kernel 5":
        kernel = np.array([[-0.5, -1, -0.5], [-1, 7, -1], [-0.5, -1, -0.5]])
    elif kernel_name == "Kernel 6":
        kernel = np.array([[-0.45, -1, -0.45], [-1.3, 7.45, -1.3], [-0.45, -1, -0.45]])
    new_img = cv2.filter2D(old_img, -1, kernel)
    # Return image.
    return new_img

# +++++++++++++
# Create web UI
# +++++++++++++

with gr.Blocks(css="footer{display:none !important}") as webui:
    # Define a local function.
    def download_image(image):
        date_time_jpg = datetime.now().strftime("/%Y-%m-%d_%H:%M:%S.%f.jpg")
        name = ''.join([str(OUTPUTS_PATH), date_time_jpg])
        cvImg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(name, cvImg)
        return image
    def inversion(image):
        inverted_image = cv2.bitwise_not(image)
        return inverted_image
    def grayscale(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray
    with gr.Row():
        gr.HTML("""<div style='display:table-cell;vertical-align:middle;font-size:24px;height:32px;width:auto;'>Lazy Image Upscaler</div>""")
    with gr.Tab("Super Resolution"):
        # Create a row.
        with gr.Row():
            model_file = gr.Dropdown(choices=get_model_list(), value=None, label="Model List", scale=2)
            kernel_number = gr.Dropdown(choices=["Kernel 0", "Kernel 1", "Kernel 2", "Kernel 3", "Kernel 4", "Kernel 5", "Kernel 6"], label="Sharpening Kernel")
            with gr.Column(scale=0, min_width=200):
                download_button = gr.Button(value="📥 Download Image", scale=1)
                refresh_button = gr.Button(value="🔁 Refresh Model List", min_width=20, scale=1)
            with gr.Column(scale=0, min_width=200):
                upscale_button_pm = gr.Button(value="📐 Upscale Original", scale=1)
                sharpen_button = gr.Button(value="📊 Sharpening")
            with gr.Column(scale=0, min_width=200):
                inversion_button = gr.Button(value="✏️  Inversion", min_width=20, scale=1)
                grayscale_button = gr.Button(value="✒️  Grayscale", scale=1)
            with gr.Column(scale=0, min_width=200):
                flip_button_h = gr.Button(value="🔄 Horizontal Flip")
                flip_button_v = gr.Button(value="🔃 Vertical Flip")
        def refresh_list(model_file):
            updated_choices = get_model_list()
            return gr.Dropdown(updated_choices)
        refresh_button.click(
            fn=refresh_list,
            inputs=[model_file],
            outputs=[model_file]
        )
        # Create a row.
        with gr.Row():
            im_input = gr.Image(width=512, height=512)
            im_output = gr.Image(width=512, height=512, interactive=False)
        with gr.Row():
            dimension_original = gr.TextArea(lines=1, label="Dimension Original")
            dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled")
        def check_dim(img):
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]
            text = str(width) + " x " + str(height) + " pixel, " + str(channels) + " color channels"
            return text
        im_input.change(check_dim, inputs=[im_input], outputs=[dimension_original])
        im_output.change(check_dim, inputs=[im_output], outputs=[dimension_upscaled])
        upscale_button_pm.click(
            fn=upscale_image_model,
            inputs=[model_file, im_input],
            outputs=[im_output]
        )
        flip_button_h.click(
            fn=flip_image_h,
            inputs=[im_output],
            outputs=[im_output]
        )
        flip_button_v.click(
            fn=flip_image_v,
            inputs=[im_output],
            outputs=[im_output]
        )
        download_button.click(
            fn=download_image,
            inputs=[im_output],
            outputs=[im_output]
        )
        sharpen_button.click(
             fn=sharpen_image,
            inputs=[im_output, kernel_number],
            outputs=[im_output]
        )
        inversion_button.click(
            fn=inversion,
            inputs=[im_output],
            outputs=[im_output]
        )
        grayscale_button.click(
            fn=grayscale,
            inputs=[im_output],
            outputs=[im_output]
        )
    with gr.Tab("Standard Methods"):
        # Create a row.
        with gr.Row():
            inter_method = gr.Dropdown(choices=inter_methods_list, value=None, label="Upscaling Methods", scale=2)
            scale_number = gr.Dropdown(choices=scale_factors, value=None, label="Scaling Factor", scale=0)
            kernel_number = gr.Dropdown(choices=["Kernel 0", "Kernel 1", "Kernel 2", "Kernel 3", "Kernel 4", "Kernel 5", "Kernel 6"], label="Sharpening Kernel")
            download_button = gr.Button(value="📥 Download Image", scale=1)
            with gr.Column(scale=0, min_width=200):
                upscale_button_cv = gr.Button(value="📐 Upscale Original", scale=1)
                sharpen_button = gr.Button(value="📊 Sharpening", scale=1)
            with gr.Column(scale=0, min_width=200):
                inversion_button = gr.Button(value="✏️  Inversion", min_width=20, scale=1)
                grayscale_button = gr.Button(value="✒️  Grayscale", scale=1)
            with gr.Column(scale=0, min_width=200):
                flip_button_h = gr.Button(value="🔄 Horizontal Flip")
                flip_button_v = gr.Button(value="🔃 Vertical Flip")
        # Create a row.
        with gr.Row():
            im_input = gr.Image(width=512, height=512)
            im_output = gr.Image(width=512, height=512, interactive=False)
        with gr.Row():
            dimension_original = gr.TextArea(lines=1, label="Dimension Original")
            dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled")
        def check_dim(img):
            height = img.shape[0]
            width = img.shape[1]
            channels = img.shape[2]
            text = str(width) + " x " + str(height) + " pixel, " + str(channels) + " color channels"
            return text
        im_input.change(check_dim, inputs=[im_input], outputs=[dimension_original])
        im_output.change(check_dim, inputs=[im_output], outputs=[dimension_upscaled])
        upscale_button_cv.click(
            fn=upscale_image_opencv,
            inputs=[inter_method, scale_number, im_input],
            outputs=[im_output]
        )
        flip_button_h.click(
            fn=flip_image_h,
            inputs=[im_output],
            outputs=[im_output]
        )
        flip_button_v.click(
            fn=flip_image_v,
            inputs=[im_output],
            outputs=[im_output]
        )
        download_button.click(
            fn=download_image,
            inputs=[im_output],
            outputs=[im_output]
        )
        sharpen_button.click(
            fn=sharpen_image,
            inputs=[im_output, kernel_number],
            outputs=[im_output]
        )
        inversion_button.click(
            fn=inversion,
            inputs=[im_output],
            outputs=[im_output]
        )
        grayscale_button.click(
            fn=grayscale,
            inputs=[im_output],
            outputs=[im_output]
        )
    with gr.Row():
        gr.HTML("""<div style='margin:auto;text-align:center;display:block;vertical-align:middle;font-size:14px;height:32px;width:auto;'>©️ copyright 2024, zentrocdot</div>""")

# --------------------
# Main script function
# --------------------
def main():
    demo.launch()

# Execute the script as module or as programme.
if __name__ == "__main__":
    # start web ui.
    webui.launch()
