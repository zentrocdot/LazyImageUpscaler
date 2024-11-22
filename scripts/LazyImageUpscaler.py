#!/usr/bin/python3
'''Lazy Image Upscaler'''
# pylint: disable=line-too-long
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=c-extension-no-member
# pylint: disable=global-variable-undefined
# pylint: disable=global-variable-not-assigned
# pylint: disable=global-statement
# pylint: disable=redefined-outer-name
# pylint: disable=unused-argument
# pylint: disable=unused-import
# pylint: disable=too-many-lines
# pylint: disable=useless-return
# pylint: disable=consider-using-f-string
# pylint: disable=too-many-locals
# pylint: disable=wrong-import-position
# pylint: disable=bare-except
# pylint: disable=broad-except
# pylint: disable=eval-used
# pylint: disable=no-name-in-module
# pylint: disable=multiple-statements
# pylint: disable=import-error
# pylint: disable=format-string-without-interpolation
# pylint: disable=unused-variable
#
# LazyImageUpscaler
# Version 0.0.1.2
#
# Check Image quality:
# identify -format %Q filename.jpg
# identify -verbose filename.jpg
#
# Tool for Exif metadata:
# exiftool
#
# CIVITAI and Prompt Extraction:
# CIVITAI reads only the string of UserComment as Prompt:
#geninfo = "mushroom in outer space. Steps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 285838015, Size: 512x512, Model hash: 463d6a9fe8, Model: absolutereality_v181, Version: v1.10.0"
# CIVITAI reads string of UserComment as Prompt and Resources:
#geninfo = "mushroom in outer space.\nSteps: 20, Sampler: DPM++ 2M, Schedule type: Karras, CFG scale: 7, Seed: 285838015, Size: 512x512, Model hash: 463d6a9fe8, Model: absolutereality_v181, Version: v1.10.0"

# Set version string.
__Version__ = "Version 0.0.1.2"

# Suppress warnings to reduce output on screen. Comment this out if it is not necessary.
import warnings
warnings.simplefilter("ignore", category=Warning)

# Set tensorflow log level to reduce output on screen. Comment this out if it is not necessary.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import the Python modules.
import sys
import pathlib
from pathlib import Path
from datetime import datetime
import torch
import gradio as gr
import cv2
from cv2 import dnn_superres
import numpy as np
from PIL import Image, ImageChops, ImageOps
import piexif
import piexif.helper
from SSIM_PIL import compare_ssim
import skimage as ski
from skimage import data, color
from skimage.transform import rescale, resize
from skimage import io
from super_image import A2nModel, AwsrnModel, CarnModel, DrlnModel, EdsrModel, HanModel, MdsrModel, MsrnModel, PanModel, RcanModel, ImageLoader

# Import Diffuser modules. Supress some warnings.
from diffusers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger("diffusers")
logger.setLevel(logging.FATAL)
from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionLatentUpscalePipeline, StableDiffusionUpscalePipeline, StableDiffusionPipeline

import contextlib
import io as io_stdout

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = io_stdout.StringIO()
    yield
    sys.stdout = save_stdout

# Set the path variables.
SCRIPT_PATH = pathlib.Path(__file__).parent.resolve()
PARENT_PATH = SCRIPT_PATH.parent.absolute()
MODEL_PATH = ''.join([str(PARENT_PATH), "/super-resolution"])
SUPER_IMAGE_PATH = ''.join([str(PARENT_PATH), "/super-image"])
SUPER_RESOLUTION_PATH = ''.join([str(PARENT_PATH), "/super-resolution"])
STABLE_DIFFUSION_PATH = ''.join([str(PARENT_PATH), "/diffusion-models"])
OUTPUTS_PATH = ''.join([str(PARENT_PATH), "/outputs"])
CONFIG_PATH = ''.join([str(PARENT_PATH), "/configs"])

# Special path variables.
SD15_PATH = ''.join([str(PARENT_PATH), "/diffusion-models/stable-diffusion-v1-5"])
SDx2_PATH = ''.join([str(PARENT_PATH), "/diffusion-models/sd-x2-latent-upscaler"])
SDx4_PATH = ''.join([str(PARENT_PATH), "/diffusion-models/stable-diffusion-x4-upscaler"])
SR_TMP_IMG_PATH = ''.join([str(PARENT_PATH), "/super-image/tmp.png"])

# For internal use while development only.
#DEBUG = True
DEBUG = False

# Tab list names.
config_list=["isRawImage", "isExifImage", "isChopsImage", "isScikitTab",
             "isPilTab", "isOpencvTab", "isSuperResolutionTab",
             "isSuperImageTab", "isStableDiffusionTab", "isRawImage",
             "isExifImage", "isChopsImage", "JpgQuality"]

# Predefined values.
JpgQuality = 100
isRawImage = True
isExifImage = True
isChopsImage = True
isOpencvTab = True
isPilTab = True
isSuperResolutionTab = True
isSuperImageTab = True
isStableDiffusionTab = True
SafeTensor = False

# Set config file location.
config_file = Path(CONFIG_PATH + "/LazyImageUpscaler.config")

# Read config file if existing.
if config_file.is_file():
    # Open the config file in read only mode.
    with open(config_file, 'r', encoding= "utf-8") as file:
        for line in file:
            if not line.startswith("#") and not line == '\n':
                cfg_line = line.strip()
                cfg_list = cfg_line.split("=")
                for cfg_ele in config_list:
                    if cfg_list[0].strip() == cfg_ele:
                        value = cfg_list[1].strip()
                        if value == "True":
                            vars()[cfg_ele] = True
                        elif value == "False":
                            vars()[cfg_ele] = False
                        if cfg_ele == "JpgQuality":
                            JpgQuality = int(cfg_list[1].strip())

# Variable fo the temporary filename (quasi constant).
ORIGINAL_IMGAGE = None

# Set Tab name.
TAB = None

# Set value of info & warning message constant.
WARN_MSG = "You have not upscaled an image yet!"
INFO_MSG = "Nothing to do yet! Select an image first!"

# Initialize different interpolation methods of OpenCV.
inter_methods_dict = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_CUBIC": cv2.INTER_CUBIC,
    "INTER_LANCZOS4": cv2.INTER_LANCZOS4
}
inter_methods_list = list(inter_methods_dict.keys())

# Initialize different interpolation methods of PIL.
pil_methods_dict = {
    "NEAREST": Image.NEAREST,
    "BOX": Image.BOX,
    "BILINEAR": Image.BILINEAR,
    "HAMMING": Image.HAMMING,
    "BICUBIC": Image.BICUBIC,
    "LANCZOS": Image.LANCZOS
}
pil_methods_list = list(pil_methods_dict.keys())

# Initialize different interpolation methods of scikit.
scikit_method_dict = {
    "NEAREST-NEIGHBOR": 0,
    "BI-LINEAR": 1,
    "BI-QUDRATIC": 2,
    "BI-CUBIC": 3,
    "BI-QUARTIC": 4,
    "BI-QUINTIC": 5
}
scikit_method_list = list(scikit_method_dict.keys())

si_list = ["a2n", "awsrn-bam", "carn", "carn-bam", "drln", "drln-bam", "edsr", "edsr-base", "han", "mdsr", "mdsr-bam", "msrn", "msnr-bam", "pan", "pan-bam", "rcan"]
si_scale = [2,3,4]

sd_list = ["x4", "x2", "sd 1.5"]

# Set the scale factors.
scale_factors = list(range(1, 257))

# Initialise kernel list.
kernel_dict = {
    "Kernel 0": np.array([[-0.0023, -0.0432, -0.0023], [-0.0432, 1.182, -0.0432], [-0.0023, -0.0432, -0.0023]]),
    "Kernel 1": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Kernel 2": np.array([[-0.11, -0.11, -0.11], [-0.11, 1.89, -0.11], [-0.11, -0.11, -0.11]]),
    "Kernel 3": np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]),
    "Kernel 4": np.array([[-1.1,-1.1,-1.1], [-1.1,10,-1.1], [-1.1,-1.1,-1.1]]),
    "Kernel 5": np.array([[-0.5, -1, -0.5], [-1, 7, -1], [-0.5, -1, -0.5]]),
    "Kernel 6": np.array([[-0.45, -1, -0.45], [-1.3, 7.45, -1.3], [-0.45, -1, -0.45]])
}
kernel_list = list(kernel_dict.keys())

# brightness & contrast list using range.
bl = list(range(-255, 256))
cl = list(range(-127, 128))

# Set a private variable.
_SortDir = False

# Create a private global dictionary.
_model_dict = {}

# Create a private global list.
_model_list = []

# Define the user comment.
geninfo = "42 is the answer to all questions of the universe!"

# *********************
# Function model_scan()
# *********************
def model_scan(model_dir: str, ext: list) -> (list, list):
    '''File scan.'''
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
    '''Get model list.'''
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

# ******************
# Read Exif metadata
# ******************
def read_exif_metadata(image: Image.Image) -> tuple[str | None, dict]:
    '''Read Exif metadata from image file.'''
    # Get the image info dictionary from image file.
    items = (image.info or {}).copy()
    geninfo = items.pop('parameters', None)
    # Check if exif data are in the items dictionary.
    if "exif" in items:
        # Extract the exif metadata as binary byte string.
        exif_data = items["exif"]
        # Try to load the exif binary byte string.
        try:
            exif = piexif.load(exif_data)
        except OSError:
            exif = None
        # Try to extract the meta tag UserComment.
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")
        # If UserComment is not None, use the UserComment.
        if exif_comment:
            geninfo = exif_comment
    # Return the AI generator info.
    return geninfo

# ***********************
# Function elapsed_time()
# ***********************
def elapsed_time(start_time, end_time):
    '''Calculate (and print) the elapsed time between start and end.'''
    time_difference = (end_time - start_time).total_seconds()
    return time_difference

# ***************************
# Function upscale_image_cv()
# ***************************
def upscale_image_opencv(method_string, factor, image):
    '''Upscale image OpenCV.'''
    # Declare and set the value of the global variable TAB.
    global TAB
    TAB = "Standard upscaler OpenCV"
    # Check if image is none.
    if image is None:
        gr.Info(INFO_MSG)
        return None
    # Set the start time.
    start_time = datetime.now()
    # Create a numpy image.
    numpy_image = filepath2numpy(image)
    if DEBUG: print("Upscaled image type ->", numpy_image.dtype)
    # On factor is 1, return original image.
    if factor == 1:
        end_time = datetime.now()
        elapsed_time_ = elapsed_time(start_time, end_time)
        ssim_val = ssim_calc(numpy_image)
        return image, elapsed_time_, ssim_val
    # Check if the factor is None.
    if factor is None:
        factor = 1
    # Check if interpolation_method is None.
    if method_string is None:
        # Use the first entry of the model list.
        method_string = "cv2.INTER_NEAREST"
        method_value = inter_methods_dict.get("cv2.INTER_NEAREST")
    # Get the method value.
    method_value = inter_methods_dict.get(method_string)
    # Print method name and value.
    if DEBUG: print(method_string, "->", method_value)
    # Upscale the numpy image.
    val0 = numpy_image.shape[1] * factor
    val1 = numpy_image.shape[0] * factor
    dimension = (val0, val1)
    upscaled = cv2.resize(numpy_image, dimension, interpolation=method_value)
    # Set the end time.
    end_time = datetime.now()
    # Elapsed time.
    elapsed_time_ = elapsed_time(start_time, end_time)
    ssim_val = ssim_calc(upscaled)
    # Return the upscaled image.
    return upscaled, elapsed_time_, ssim_val

# ****************************
# Function upscale_image_pil()
# ****************************
def upscale_image_pil(method_name, factor, imageFilePath):
    '''Upscale image PIL.'''
    # Declare and set the value of the global variable TAB.
    global TAB
    TAB = "Standard PIL"
    if imageFilePath is None:
        gr.Info(INFO_MSG)
        return None
    # Set the start time.
    start_time = datetime.now()
    # On factor is 1, return original image.
    if factor == 1:
        numpy_image = filepath2numpy(imageFilePath)
        end_time = datetime.now()
        elapsed_time_ = elapsed_time(start_time, end_time)
        ssim_val = ssim_calc(numpy_image)
        return numpy_image, elapsed_time_, ssim_val
    # Get the method.
    method = pil_methods_dict.get(method_name)
    # Check if the model_path is None.
    if method is None:
        # Use the first entry of the model list.
        method_name = pil_methods_list[0]
        method = 0
    if DEBUG: print(method_name, "->", method)
    if factor is None:
        factor = 1
    # Upscale the numpy image.
    with Image.open(imageFilePath) as image:
        val0 = image.width * factor
        val1 = image.height * factor
        (width, height) = (val0, val1)
        upscaled = image.resize((width, height), reducing_gap=3.0)
    upscaled = np.array(upscaled)
    # Set the end time.
    end_time = datetime.now()
    # Elapsed time.
    elapsed_time_ = elapsed_time(start_time, end_time)
    #im = PIL.Image.fromarray(numpy.uint8(I))
    ssim_val = ssim_calc(upscaled)
    # Return the upscaled image.
    return upscaled, elapsed_time_, ssim_val

# *******************************
# Function upscale_image_scikit()
# *******************************
def upscale_image_scikit(scikit_method, factor, image):
    '''Upscale an image using scikit-image.'''
    # Declare and set the value of the global variable TAB.
    global TAB
    TAB = "Standard method scikit-image"
    # Check if image is none.
    if image is None:
        # Show a warning on screen.
        gr.Info(INFO_MSG)
        # Return None.
        return None
    # Set the start time.
    start_time = datetime.now()
    # Check if factor is None.
    if factor is None:
        factor = 1
    # Check if interpolation method is None.
    if scikit_method is None:
        scikit_method = scikit_method_list[0]
    # Get interpolation method as integer value.
    scikit_method_num = scikit_method_dict.get(scikit_method)
    # Set anti aliasing.
    anti_alias = False
    # Create a numpy array from the image using scikit-image.
    numpyImage = io.imread(image)
    if DEBUG: print("Upscaled image type ->", numpyImage.dtype)
    # On factor is 1, return original image.
    if factor == 1:
        if DEBUG: print(scikit_method, "->", scikit_method_num)
        end_time = datetime.now()
        elapsed_time_ = elapsed_time(start_time, end_time)
        ssim_val = ssim_calc(numpyImage)
        return image, elapsed_time_, ssim_val
    # Print on debug.
    if DEBUG: print(scikit_method, "->", scikit_method_num)
    # Get numpy diomension data.
    width = numpyImage.shape[1]
    height = numpyImage.shape[0]
    dim = numpyImage.shape[2]
    # Calculate new dimension.
    cols = int(width * factor)
    rows = int(height * factor)
    # Resize the image.
    upscaled = resize(numpyImage, (rows, cols, dim), order=scikit_method_num, anti_aliasing=anti_alias)
    # Set the end time.
    end_time = datetime.now()
    # Elapsed time.
    elapsed_time_ = elapsed_time(start_time, end_time)
    # Calculate ssim. Workaround for the numpy problem.
    img_type = str(upscaled.dtype)
    tmp_arr = upscaled
    match img_type:
        case "uint8":
            if DEBUG: print("Upscaled image type ->", upscaled.dtype)
            pass
        case "float64":
            if DEBUG: print("Upscaled image type ->", upscaled.dtype)
            tmp_arr = (tmp_arr * 255).astype(np.uint8)
        case _:
            if DEBUG: print("Upscaled image type ->", upscaled.dtype)
            tmp_arr = (tmp_arr * 127).astype(np.uint8)
    ssim_val = ssim_calc(tmp_arr)
    # Return the upscaled image, elapsed time and the ssim.
    return upscaled, elapsed_time_, ssim_val

# ***************************
# Function upscale_image_sr()
# ***************************
def upscale_image_sr(model_name, image):
    '''Upscale image using super resolution.'''
    # Declare and set the value of the global variable TAB.
    global TAB
    TAB = "Super Resolution"
    # Check if image is None.
    if image is None:
        gr.Info(INFO_MSG)
        return None
    # Set the start time.
    start_time = datetime.now()
    numpy_image = filepath2numpy(image)
    # Check if the model_path is None.
    if model_name is None:
        # Use the first entry of the model list.
        model_name = _model_list[0]
    if DEBUG: print(model_name)
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
    try:
        upscaled = sr.upsample(numpy_image)
    except:
        gr.Warning("Could not upscale image!")
        return None
    # Set the end time.
    end_time = datetime.now()
    # Elapsed time.
    elapsed_time_ = elapsed_time(start_time, end_time)
    ssim_val = ssim_calc(upscaled)
    # Return the upscaled image.
    return upscaled, elapsed_time_, ssim_val

# ***************************
# Function upscale_image_si()
# ***************************
def upscale_image_si(imageFilePath, model_name, model_scale):
    '''Upscale image using Super Image.'''
    # Define helper function.
    def upcase_first_letter(s):
        '''Upcase first letter.'''
        return s[0].upper() + s[1:]
    # Set some variables.
    pre = "pytorch_model_"
    post = "x.pt"
    # Declare and set the value of the global variable TAB.
    global TAB
    TAB = "Super Image"
    # Check if imageFilePath is None.
    if imageFilePath is None:
        gr.Info(INFO_MSG)
        return None
    # Set the start time.
    start_time = datetime.now()
    # Initialise cuda device.
    torch.cuda.set_device('cuda:0')
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    # Check if model name is None.
    if model_name is None:
        model_name = si_list[0]
    # Check if sclae is None.
    if model_scale is None:
        model_scale = 2
    # Create a PIL image.
    image = Image.open(imageFilePath)
    # Create model file name.
    model = pre + str(model_scale) + post
    # Create module name.
    module = model_name + "Model"
    if "edsr" in module:
        module = module.replace("-base", "")
    if "-bam" in module:
        module = module.replace("-bam", "")
    module = upcase_first_letter(module)
    # Initialise the super resolution object.
    sd_upscaler = "/super-image/" + model_name
    SD_PATH = ''.join([str(PARENT_PATH), sd_upscaler])
    try:
        # Trap text output of eval.
        with nostdout():
            md = module + ".from_pretrained(SD_PATH, scale=model_scale)"
            model = eval(md)
    except:
        gr.Warning("Model not installed yet!")
        return None
    # Upload model and image to GPU.
    model = model.to(device)
    inputs = ImageLoader.load_image(image)
    inputs = inputs.to(device)
    # Try to upscale the image.
    try:
        preds = model(inputs)
    except:
        gr.Warning("Could not upscale image!")
        return None, "", ""
    #img_path = "/super-image/tmp.png"
    #TMP_IMG_PATH = ''.join([str(PARENT_PATH), img_path])
    # Save the created image.
    ImageLoader.save_image(preds, SR_TMP_IMG_PATH)
    # Load the upscaled image
    upscaled = filepath2numpy(SR_TMP_IMG_PATH)
    # Set the end time.
    end_time = datetime.now()
    # Get the elapsed time.
    elapsed_time_ = elapsed_time(start_time, end_time)
    # Calculate the ssim.
    ssim_val = ssim_calc(upscaled)
    # Return the upscaled image.
    return upscaled, elapsed_time_, ssim_val

# ***************************
# Function upscale_image_sd()
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/latent_upscale
# ***************************
def upscale_image_sd(image_filename, model_name):
    '''Upscale an image using Stable Diffusion.'''
    # Declare and set the value of the global variable TAB.
    global TAB
    TAB = "Stable Diffusion"
    # Check if image is None. If None return None.
    if image_filename is None:
        gr.Info(INFO_MSG)
        return None
    # Check if factor is None. If None preset value.
    if model_name is None:
        model_name = "x4"
    # Set the start time.
    start_time = datetime.now()
    # Free GPU cache.
    torch.cuda.empty_cache()
    # Open image.
    image = Image.open(image_filename)
    # Resize image.
    image = image.resize((512, 512))
    # Upscale image using sd model.
    with torch.cuda.device(0):
        if model_name == "sd 1.5":
            prompt = ""
            pipeline = StableDiffusionImg2ImgPipeline.from_pretrained(SD15_PATH,
                           torch_dtype=torch.float16, use_safetensors=False)
            pipeline = pipeline.to("cuda")
            pipeline.enable_vae_tiling()
            pipeline.enable_sequential_cpu_offload()
            lowres_latents = pipeline(prompt=prompt, image=image, strength=0.0, guidance_scale=1.0,
                            num_inference_steps=20).images[0]
            upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(SDx2_PATH, torch_dtype=torch.float16)
            upscaler.to("cuda")
            upscaler.enable_vae_tiling()
            upscaler.enable_sequential_cpu_offload()
            #prompt = "strong colors, vibrant colors, very high-quality, accurate photo, intricate photo, highly detailed, sharp focus, colorful, 8K"
            upscaled = upscaler(prompt=prompt, image=lowres_latents, num_inference_steps=20).images[0]
        else:
            if model_name == "x4":
                SD_PATH = SDx4_PATH
            elif model_name == "x2":
                SD_PATH = SDx2_PATH
            pipe = DiffusionPipeline.from_pretrained(SD_PATH, torch_dtype=torch.float16, use_safetensors=False)
            pipe = pipe.to("cuda")
            pipe.enable_vae_tiling()
            pipe.enable_sequential_cpu_offload()
            #prompt = "strong colors, vibrant colors, very high-quality, accurate photo, intricate photo, highly detailed, sharp focus, colorful, 8K"
            prompt = ""
            upscaled = pipe(prompt=prompt, image=image,
                            num_inference_steps=25).images[0]
    # Convert upscaled image.
    upscaled = np.array(upscaled)
    # Set the end time.
    end_time = datetime.now()
    # Elapsed time.
    elapsed_time_ = elapsed_time(start_time, end_time)
    ssim_val = ssim_calc(upscaled)
    # Return the upscaled image.
    return upscaled, elapsed_time_, ssim_val

# ****************************************************************************
# Function rotate_image_l()
#
# np.ndarray -> numpy.ndarray
# Image      -> PIL.Image.Image
#
# Gradio accepts per definition as output to a component a numpy array as
# well as a PIL image. The input from the output image component is so far
# all the time a numpy array.
#
# See: https://www.gradio.app/docs/gradio/image
# ****************************************************************************
def rotate_image_l(image: np.ndarray) -> Image:
    '''90° left rotation of an image.'''
    # Check if the input image is None.
    if image is None:
        # Show warning on screen.
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    # Rotate image left.
    rotated_image = Image.fromarray(np.rot90(image, k=1, axes=(0,1)))
    # Return flipped image.
    #rotated_image = np.array(rotated_image) <- numpy array equivalent
    # Return rotated PIL image.
    return rotated_image

# ****************************************************************************
# Function sepia_filter()
#
# np.ndarray -> numpy.ndarray
# Image      -> PIL.Image.Image
#
# Gradio accepts per definition as output to a component a numpy array as
# well as a PIL image. The input from the output image component is so far
# all the time a numpy array.
#
# See: https://www.gradio.app/docs/gradio/image
# See: https://yabirgb.com/sepia_filter/
# ****************************************************************************
def sepia_filter(image: np.ndarray, sepia_value: str) -> Image:
    '''Apply sepia filter on the  image.'''
    # Check if the input image is None.
    if image is None:
        # Show warning on screen.
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    # Check if sepia value is None.
    if sepia_value is None:
        sepia_value = 0.0
    # Define the sepia filter factor.
    k = float(sepia_value)
    scalar = (1 - k)
    # Define the sepia filter matrix. To make sure, that k=1 and k=0 will be
    # correct different cases are used. Otherwise there can be rounding errors.
    unity_matrix = [[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]]
    unity_array = np.matrix(unity_matrix)
    sepia_matrix = [[0.393, 0.769, 0.189],
                    [0.349, 0.686, 0.168],
                    [0.272, 0.534, 0.131]]
    sepia_array = np.matrix(sepia_matrix)
    correct_matrix = [[0.607, -0.769, -0.189],
                      [-0.349, 0.314, -0.168],
                      [-0.349, -0.534, 0.869]]
    correct_array = np.matrix(correct_matrix)
    # Do something on match.
    match k:
        case 0:
            matrix = unity_array
        case 1:
            matrix = sepia_array
        case _:
            arr0 = sepia_array
            arr1 = correct_array * scalar
            matrix = np.add(arr0, arr1)
    #lmap = np.matrix(matrix)
    lmap = matrix
    if DEBUG: print(lmap)
    # Calculate the filtered image.
    filtered_image = np.array([x * lmap.T for x in image])
    # Check wich entries have a value greather than 255 and set it to 255
    filtered_image[np.where(filtered_image>255)] = 255
    # Create an image from the array
    filtered_image = Image.fromarray(filtered_image.astype('uint8'))
    # Return rotated PIL image.
    return filtered_image

# ****************************************************************************
# Function rotate_image_r()
#
# np.ndarray -> numpy.ndarray
# Image      -> PIL.Image.Image
#
# Gradio accepts per definition as output to a component a numpy array as
# well as a PIL image. The input from the output image component is so far
# all the time a numpy array.
#
# See: https://www.gradio.app/docs/gradio/image
# ****************************************************************************
def rotate_image_r(image: np.ndarray) -> Image:
    '''90° right rotation of an image.'''
    # Check if the input image is None.
    if image is None:
        # Show warning on screen.
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    # Rotate image right.
    rotated_image = Image.fromarray(np.rot90(image, k=-1, axes=(0,1)))
    #rotated_image = np.array(rotated_image) <- numpy array equivalent
    # Return rotated PIL image.
    return rotated_image

# ***********************
# Function flip_image_h()
# ***********************
def flip_image_h(image):
    ''' Horizontal flip of image.'''
    # Check if the image is None.
    if image is None:
        # Show warning on screen.
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    # Flip image.
    flipped_image = np.fliplr(image)
    # Return flipped image.
    return flipped_image

# ***********************
# Function flip_image_v()
# ***********************
def flip_image_v(image):
    ''' Horizontal flip of image.'''
    # Check if the image is None.
    if image is None:
        # Show warning on screen.
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    # Flip image.
    flipped_image = np.flipud(image)
    # Return flipped image.
    return flipped_image

# ******************************
# Function sharpen_image_model()
# ******************************
def sharpen_image(old_img, kernel_name):
    '''Sharpen image.'''
    if old_img is None:
        gr.Warning(WARN_MSG)
        return None
    # Check if kernel is set.
    if kernel_name is None:
        kernel = np.array([[-0.0023, -0.0432, -0.0023], [-0.0432, 1.182, -0.0432], [-0.0023, -0.0432, -0.0023]])
    else:
        kernel = kernel_dict[kernel_name]
    # Sharpen image.
    new_img = cv2.filter2D(old_img, -1, kernel)
    # Return image.
    return new_img

# **************************
# Function smoothing_image()
# https://docs.opencv.org/4.10.0/d4/d13/tutorial_py_filtering.html
# https://www.tutorialspoint.com/how-to-change-the-contrast-and-brightness-of-an-image-using-opencv-in-python
# https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
# **************************
def smoothing_image(image):
    '''Smoothing image.'''
    if image is None:
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    kernel = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) * (1/9)
    image = cv2.filter2D(image, -1, kernel)
    # Return image.
    return image

# *******************
# Function brighten()
# *******************
def brighten(image, bn):
    '''Brighten image.'''
    if image is None:
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    if bn is None:
        bn = bl[513]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = bn
    hsv[:,:,2] = cv2.add(hsv[:,:,2], value)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # Return the modified image.
    return image

# *******************
# Function contrast()
# *******************
def contrast(image, alpha, beta):
    '''
    alpha = contrast
    beta = brightness
    '''
    # Check if image is None.
    if image is None:
        # Show a warning message.
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    # Check if alpha and/or beta are None.
    if alpha is None:
        alpha = cl[511]
    if beta is None:
        beta = bl[513]
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    #adjusted = cv2.addWeighted(image, alpha, image, 0, beta)
    # Return the modified image.
    return adjusted

# ********************
# Function inversion()
# ********************
def inversion(image):
    '''Inversion of image.'''
    if image is None:
        gr.Warning(WARN_MSG)
        # Do nothing. Return None.
        return None
    inverted_image = cv2.bitwise_not(image)
    return inverted_image

# ********************
# Function grayscale()
# ********************
def grayscale(image):
    '''Grayscale image.'''
    if image is None:
        gr.Warning(WARN_MSG)
        return None
        # Do nothing. Return None.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# ********************
# Function denoising()
# https://docs.opencv.org/3.4/d5/d69/tutorial_py_non_local_means.html
# https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_photo/py_non_local_means/py_non_local_means.html
# https://www.geeksforgeeks.org/python-denoising-of-colored-images-using-opencv/
# https://docs.opencv.org/4.x/d1/d79/group__photo__denoise.html
# ********************
def denoising(src, string):
    '''Denoising image.'''
    if src is None:
        gr.Warning(WARN_MSG)
        return None
    string = string.replace("(", "")
    string = string.replace(")", "")
    val = string.split(",")
    h = float(val[0])
    hcolor = float(val[1])
    templateWindowSize = int(val[2])
    searchWindowSize = int(val[3])
    #dst = cv2.fastNlMeansDenoisingColored(src,None,10,10,7,21)
    dst = cv2.fastNlMeansDenoisingColored(src,None,h,hcolor,templateWindowSize,searchWindowSize)
    return dst

# ****************************************************************************
# Function gamma()
# https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/
# https://stackoverflow.com/questions/61695773/how-to-set-the-best-value-for-gamma-correction
# https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# ****************************************************************************
def gamma(numpyImage, gamma):
    '''gamma correction.'''
    # Check if value of numpyImage is None.
    if numpyImage is None:
        # Show a warn message.
        gr.Warning(WARN_MSG)
        # Return None.
        return None
    # Check if gamma is None.
    if gamma is None:
        gamma = 1.0
    #inverseGamma = 1.0 / gamma
    #lookUpTable = np.array([((i / 255.0) ** inverseGamma) * 255
    #for i in np.arange(0, 256)]).astype("uint8")
    lookUpTable = np.empty((1,256), np.uint8)
    try:
        for i in range(256):
            lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    except ZeroDivisionError as err:
        # Print the error in the terminal window.
        print(err)
        # Show a warning on screen.
        gr.Warning("An error has occurred! The chosen value for gamma is possibly not allowed! The image is unchanged!")
        # Return the numpyImage unchanged.
        return numpyImage
    modified_image = cv2.LUT(numpyImage, lookUpTable)
    # Return the modified image.
    return modified_image

# *************************
# Function download_image()
# *************************
def download_image(numpyImage):
    '''Download image.'''
    # Check if numpy image is None.
    if numpyImage is None:
        gr.Warning(WARN_MSG)
        return None
    # Calculate ssim.
    ssim_val = ssim_calc(numpyImage)
    # Initialise date time string.
    date_time_string = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    # Method without Exif metadata.
    if isRawImage:
        date_time_jpg = "/" + date_time_string + ".jpg"
        filename = ''.join([str(OUTPUTS_PATH), date_time_jpg])
        cvImg = cv2.cvtColor(numpyImage, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, cvImg, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # Method with Exif metadata.
    if isExifImage:
        # Initialise geninfo.
        geninfo = None
        # Create full path filename.
        date_time_jpg = "/" + date_time_string + "_exif.jpg"
        filename = ''.join([str(OUTPUTS_PATH), date_time_jpg])
        # Set jpg parameter.
        extension = ".jpg"
        image_format = Image.registered_extensions()[extension]
        # Try to read Exif metadata
        try:
            # Open temporary image.
            tmp_image = Image.open(ORIGINAL_IMAGE)
            # Read the Exif data.
            geninfo = read_exif_metadata(tmp_image)
        except AttributeError as err:
            print("AttributeError:", err)
            warn_msg = '''\
                          The Exif metadata are no longer available. Next
                          time, do not remove the original image before
                          the new created image has been downloaded!\
                          '''.format()
            gr.Warning(warn_msg)
            return None
        # Create an image.
        image = Image.fromarray(numpyImage.astype('uint8'), 'RGB')
        # Save the jpp image file.
        image.save(filename, format=image_format, quality=JpgQuality, lossless=True)
        # Check geninfo.
        if geninfo is not None:
            exif_bytes = piexif.dump({
                "Exif": {
                    piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(geninfo or "", encoding="unicode"),
                }
            })
            # Insert the Exif metadata in the image file.
            piexif.insert(exif_bytes, filename)
        # Insert Exif metadata in the image file.
        # https://pillow.readthedocs.io/en/stable/_modules/PIL/ExifTags.html
        tmpimg = Image.open(filename)
        exif = tmpimg.getexif()
        if TAB is not None:
            exif[0xC6F8] = TAB
        exif[0x0131] = "Lazy Image Upscaler"
        exif[0x010E] = "42 is the answer to every question in the universe!"
        tmpimg.save(filename, quality=JpgQuality, lossless=True, exif=exif)
    if isChopsImage:
        # Return an empty list, to make gradio happy.
        im1 = Image.open(ORIGINAL_IMAGE)
        im1 = im1.resize((512, 512))
        im2 = Image.open(filename)
        im2 = im2.resize((512, 512))
        diff = ImageChops.difference(im2, im1)
        inverted_diff = ImageOps.invert(diff)
        date_time_jpg = "/" + date_time_string + "_chops.jpg"
        cfilename = ''.join([str(OUTPUTS_PATH), date_time_jpg])
        inverted_diff.save(cfilename)
    # Return value.
    return ssim_val

# ********************
# Function ssim_calc()
# ********************
def ssim_calc(numpyImage):
    '''Calculate the structural similarity as floating point value.'''
    try:
        # First image is of type filepath.
        img1 = Image.open(ORIGINAL_IMAGE)
        # Resize image to 512 x 512 pixel.
        img1 = img1.resize((512, 512))
        # Second image is of type numpy.
        img2 = Image.fromarray(numpyImage)
        # Resize image to 512 x 512 pixel.
        img2 = img2.resize((512, 512))
        # Calculate the ssim on CPU.
        ssim_value = compare_ssim(img1, img2, GPU=False)
        # Return the ssim value.
    except Exception as err:
        print(err)
        ssim_value = "n/a"
    return ssim_value

# ********************
# Function pil2numpy()
# ********************
def pil2numpy(pilImage):
    '''Create a numpy image from a PIL image.'''
    # Check if pilImage is None.
    if pilImage is None:
        return None
    # Create a numpy array.
    numpyImage = np.array(pilImage)
    # Return the numpy image.
    return numpyImage

# *************************
# Function filepath2numpy()
# *************************
def filepath2numpy(imageFilePath):
    '''Create a numpy image from a file path image.'''
    # Check if imageFilePath is None.
    if imageFilePath is None:
        return None
    # Create a pil image.
    pilImage = Image.open(imageFilePath)
    # Create a numpy array.
    numpyImage = np.array(pilImage)
    # Return the numpy image.
    return numpyImage

# ********************
# Function check_dim()
# ********************
def check_dim(image):
    '''Check the image dimensions.'''
    # Check if image is None.
    if image is None:
        return ""
    # Check type of image.
    if isinstance(image, str):
        pilImage = Image.open(image)
        numpyImage = np.array(pilImage)
    elif isinstance(image, Image.Image):
        numpyImage = np.array(image)
    elif isinstance(image, np.ndarray):
        numpyImage = image
    # Get width, height and channels from numpy image.
    width = numpyImage.shape[1]
    height = numpyImage.shape[0]
    channels = numpyImage.shape[2]
    # Create a string with the dimensions.
    dim_str = f"{width} x {height} pixel, {channels} color channels".format(str(width), str(height), str(channels))
    # Return the string with the dimensions.
    return dim_str

# ****************************
# Function postprocess_image()
# ****************************
def postprocess_image(image):
    '''Check the image dimensions.'''
    # Check if image is None.
    if image is None:
        return ""
    # Get the image dimensions.
    imageDimensions = check_dim(image)
    # Return image file path and image dimensions.
    return imageDimensions

# ***************************
# Function preprocess_image()
# ***************************
def preprocess_image(imageFilePath):
    '''Preprocess the selected image.'''
    # Declare the global variable.
    global ORIGINAL_IMAGE
    # Check if tmp_file_path is None.
    if imageFilePath is None:
        return "", ""
    # Set the global variable.
    ORIGINAL_IMAGE = imageFilePath
    # Get the image dimensions.
    imageDimensions = check_dim(imageFilePath)
    # Return image file path and image dimensions.
    return [imageFilePath, imageDimensions]

# ***********************
# Function check_dim_0()
# Obsolete in the future!
# ***********************
def check_dim_0(img):
    '''Check image dimensions.'''
    # Check if img is None.
    if img is None:
        return ""
    # Create a numpy image.
    img = filepath2numpy(img)
    # Get image dimensions.
    text = check_dim(img)
    # Return a text string.
    return text

# ***********************
# Function check_dim_1()
# Obsolete in the future!
# ***********************
def check_dim_1(img):
    '''Check image dimensions.'''
    # Check if img is None.
    if img is None:
        return ""
    # Get image dimensions.
    text = check_dim(img)
    # Return a text string.
    return text

# ************************
# Function get_file_path()
# Obsolete in the future!
# ************************
def get_file_path(tmp_file_path):
    '''Get the temporary image file path.'''
    # Check if tmp_file_path is None.
    if tmp_file_path is None:
        return ""
    # Return the temporary file path.
    return tmp_file_path

# ***********************
# Function refresh_list()
# ***********************
def refresh_list(model_file):
    '''Refresh the model list.'''
    # Update the choices list.
    updated_choices = get_model_list()
    # Update the drop-down menue.
    return gr.Dropdown(updated_choices)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Create the Lazy Image Upscaler web UI
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Set the string constants.
UPSCALE_ORIGINAL = "📐 Upscale Original"
SEPIA_FILTER = "📊 Sepia Filter"
DOWNLOAD_IMAGE = "📥 Download Image"
CALC_SSIM = "🔰 Calculate SSIM"
CLEAR_RESET = "♻️  Clear & Reset"
HFLIP = "🔄 Horizontal Flip"
VFLIP = "🔃 Vertical Flip"
LROT = "↪️  Rotate Left"
RROT = "↩️  Rotate Right"
REFRESH_MODELS = "🔁 Refresh Model List"
SHARP_BUTTON = "📈 Sharpening"
SMOOTH_BUTTON = "📉 Smoothing"
DENOISE_BUTTON = "📝 Denoising"
GAMMA_BUTTON = "📝 Gamma"
BRIGHT_BUTTON = "🌕 Brightness"
CONTRAST_BUTTON = "🌑 Contrast"
INVERT_BUTTON = "✏️  Inversion"
GRAYSCALE_BUTTON = "✒️  Grayscale"
# The Gardio footer is removed by use of css style!
with gr.Blocks(css="footer{display:none !important}", fill_width=True,
               fill_height=False) as webui:
    # Create the header line.
    with gr.Row():
        header_text = """<div style='font-size:22px;height:26px;width:auto;
                         display:table-cell;vertical-align:middle;'>
                         Lazy Image Upscaler</div>"""
        gr.HTML(header_text)
    # ************************************************************************
    # OpenCV Section
    # ************************************************************************
    # Check if Tab should be created.
    if isOpencvTab:
        # Create a Tab in the main block.
        with gr.Tab("Standard Methods (OpenCV)"):
            # ------------------
            # Components section
            # ------------------
            # Create a row in the tab.
            with gr.Row():
                upscale_button_cv = gr.Button(value=UPSCALE_ORIGINAL, scale=2)
                flip_button_h = gr.Button(value=HFLIP, scale=1)
                flip_button_v = gr.Button(value=VFLIP, scale=1)
                rotate_button_l = gr.Button(value=LROT, scale=1)
                rotate_button_r = gr.Button(value=RROT, scale=1)
                sepia_button = gr.Button(value=SEPIA_FILTER, scale=1)
                download_button = gr.Button(value=DOWNLOAD_IMAGE, scale=2)
            # Create a row in the tab.
            with gr.Row():
                inter_method = gr.Dropdown(choices=inter_methods_list, value=inter_methods_list[0], label="Upscaling Methods", scale=2, min_width=190)
                scale_number = gr.Dropdown(choices=scale_factors, value=scale_factors[0], label="Scaling", scale=1, min_width=100)
                kernel_number = gr.Dropdown(choices=kernel_list, label="Sharpening Kernel", scale=0, min_width=140)
                brightness_number = gr.Number(value=0, label="Brightness", scale=0, interactive=True, min_width=90, step=0.1)
                contrast_number = gr.Number(value=1, label="Contrast", scale=0, interactive=True, min_width=90, step=0.1)
                gamma_number = gr.Number(value=1, label="Gamma", scale=0, interactive=True, min_width=90, step=0.1)
                denoise_string = gr.Textbox(value="(10,10,7,21)", max_lines=1, label="Denoising", scale=1, interactive=True, min_width=100)
                sepia_number = gr.Number(value="0", label="Sepia", scale=0, min_width=90, step=0.1)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    sharpen_button = gr.Button(value=SHARP_BUTTON, scale=1, min_width=60)
                    smoothing_button = gr.Button(value=SMOOTH_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    denoising_button = gr.Button(value=DENOISE_BUTTON, scale=1, min_width=60)
                    gamma_button = gr.Button(value=GAMMA_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    brighten_button = gr.Button(value=BRIGHT_BUTTON, scale=1, min_width=60)
                    contrast_button = gr.Button(value=CONTRAST_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    inversion_button = gr.Button(value=INVERT_BUTTON, scale=1, min_width=60)
                    grayscale_button = gr.Button(value=GRAYSCALE_BUTTON, scale=1, min_width=60)
            # Create a row in the tab.
            with gr.Row():
                # Create a column in the row.
                with gr.Column():
                    im_input = gr.Image(type='filepath', sources=['upload', 'clipboard'], height=512)
                    dimension_original = gr.TextArea(lines=1, label="Dimension Original Image")
                    original_file = gr.TextArea(lines=1, label="Original File Location", interactive=True)
                # Create a column in the row.
                with gr.Column():
                    im_output = gr.Image(height=512, interactive=False)
                    dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled Image")
                    # Create a row in the column.
                    with gr.Row():
                        time_value = gr.TextArea(lines=1, label="Elapsed Time in Seconds", interactive=True, min_width=100, scale=2)
                        ssim_value = gr.TextArea(lines=1, label="SSIM Value as Float", interactive=True, min_width=100, scale=2)
                        with gr.Column(min_width=250, scale=1):
                            ssim_dummy = gr.HTML(" ")
                            ssim_button = gr.Button(value=CALC_SSIM)
            clear_list = [im_input, im_output, dimension_original,
                          dimension_upscaled, time_value, ssim_value,
                          inter_method, scale_number, kernel_number,
                          brightness_number, contrast_number, gamma_number,
                          denoise_string, sepia_number]
            # ----------------------
            # Event listener section
            # ----------------------
            # Image input and output event listener.
            im_input.change(
                preprocess_image,
                inputs=[im_input],
                outputs=[original_file, dimension_original]
            )
            im_output.change(
                postprocess_image,
                inputs=[im_output],
                outputs=[dimension_upscaled]
            )
            # Image upscale button listener. Changes from Tab to Tab!
            upscale_button_cv.click(
                fn=upscale_image_opencv,
                inputs=[inter_method, scale_number, im_input],
                outputs=[im_output, time_value, ssim_value]
            )
            # Image flip listener.
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
            # Image rotate button listener.
            rotate_button_l.click(
                fn=rotate_image_l,
                inputs=[im_output],
                outputs=[im_output]
            )
            rotate_button_r.click(
                fn=rotate_image_r,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image sepia button listener.
            sepia_button.click(
                fn=sepia_filter,
                inputs=[im_output, sepia_number],
                outputs=[im_output]
            )
            # Image download button listener.
            download_button.click(
                fn=download_image,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Image sharpen & smoothing button listener.
            sharpen_button.click(
                fn=sharpen_image,
                inputs=[im_output, kernel_number],
                outputs=[im_output]
            )
            smoothing_button.click(
                fn=smoothing_image,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image denoising & gamma button listener.
            denoising_button.click(
                fn=denoising,
                inputs=[im_output, denoise_string],
                outputs=[im_output]
            )
            gamma_button.click(
                fn=gamma,
                inputs=[im_output, gamma_number],
                outputs=[im_output]
            )
            # Image brighten & contrast button listener.
            brighten_button.click(
                fn=brighten,
                inputs=[im_output, brightness_number],
                outputs=[im_output]
            )
            contrast_button.click(
                fn=contrast,
                inputs=[im_output, contrast_number, brightness_number],
                outputs=[im_output]
            )
            # Image inversion & grayscale button listener.
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
            # Image ssim button listener. For all Tabs equal!
            ssim_button.click(
                ssim_calc,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Create a clear & reset button.
            def clear_reset_cv():
                '''Clear & reset of interface'''
                return [inter_methods_list[0], scale_factors[0], kernel_list[0], 0, 1, 1, "(10,10,7,21)", "0"]
            clear_button = gr.ClearButton(components=clear_list, value=CLEAR_RESET)
            clear_button.click(
                fn=clear_reset_cv,
                inputs=[],
                outputs=[inter_method, scale_number, kernel_number, brightness_number, contrast_number, gamma_number, denoise_string, sepia_number]
            )
    # ************************************************************************
    # PIL Section
    # ************************************************************************
    if isPilTab:
        # Create a tab in the main block.
        with gr.Tab("Standard Methods (PIL)"):
        # Create a row in the tab.
            # ------------------
            # Components section
            # ------------------
            with gr.Row():
                upscale_button_pil = gr.Button(value=UPSCALE_ORIGINAL, scale=2)
                flip_button_h = gr.Button(value=HFLIP, scale=1)
                flip_button_v = gr.Button(value=VFLIP, scale=1)
                rotate_button_l = gr.Button(value=LROT, scale=1)
                rotate_button_r = gr.Button(value=RROT, scale=1)
                sepia_button = gr.Button(value=SEPIA_FILTER, scale=1)
                download_button = gr.Button(value=DOWNLOAD_IMAGE, scale=2)
            # Create a row in the tab.
            with gr.Row():
                pil_method = gr.Dropdown(choices=pil_methods_list, value=pil_methods_list[0], label="Upscaling Methods", scale=2, min_width=190)
                scale_number = gr.Dropdown(choices=scale_factors, value=scale_factors[0], label="Scaling", scale=1, min_width=100)
                kernel_number = gr.Dropdown(choices=kernel_list, label="Sharpening Kernel", scale=0, min_width=140)
                brightness_number = gr.Number(value=0, label="Brightness", scale=1, interactive=True, min_width=100, step=0.1)
                contrast_number = gr.Number(value=1, label="Contrast", scale=1, interactive=True, min_width=100, step=0.1)
                gamma_number = gr.Number(value=1, label="Gamma", scale=1, interactive=True, min_width=100, step=0.1)
                denoise_string = gr.Textbox(value="(10,10,7,21)", max_lines=1, label="Denoising", scale=1, interactive=True, min_width=100)
                sepia_number = gr.Number(value="0", label="Sepia", scale=0, min_width=90, step=0.1)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    sharpen_button = gr.Button(value=SHARP_BUTTON, scale=1, min_width=60)
                    smoothing_button = gr.Button(value=SMOOTH_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    denoising_button = gr.Button(value=DENOISE_BUTTON, scale=1, min_width=60)
                    gamma_button = gr.Button(value=GAMMA_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    brighten_button = gr.Button(value=BRIGHT_BUTTON, scale=1, min_width=60)
                    contrast_button = gr.Button(value=CONTRAST_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    inversion_button = gr.Button(value=INVERT_BUTTON, scale=1, min_width=60)
                    grayscale_button = gr.Button(value=GRAYSCALE_BUTTON, scale=1, min_width=60)
            # Create a row in the tab.
            with gr.Row():
                # Create a column in the row.
                with gr.Column():
                    im_input = gr.Image(type='filepath', sources=['upload', 'clipboard'], height=512)
                    dimension_original = gr.TextArea(lines=1, label="Dimension Original Image")
                    original_file = gr.TextArea(lines=1, label="Original File Location", interactive=True)
                # Create a column in the row.
                with gr.Column():
                    im_output = gr.Image(height=512, interactive=False)
                    dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled Image")
                    # Create a row in the column.
                    with gr.Row():
                        time_value = gr.TextArea(lines=1, label="Elapsed Time in Seconds", interactive=True, min_width=100, scale=2)
                        ssim_value = gr.TextArea(lines=1, label="SSIM Value as Float", interactive=True, min_width=100, scale=2)
                        with gr.Column(min_width=250, scale=1):
                            ssim_dummy = gr.HTML(" ")
                            ssim_button = gr.Button(value=CALC_SSIM)
            clear_list = [im_input, im_output, dimension_original,
                          dimension_upscaled, time_value, ssim_value,
                          inter_method, scale_number, kernel_number,
                          brightness_number, contrast_number, gamma_number,
                          denoise_string, sepia_number]
            # ----------------------
            # Event listener section
            # ----------------------
            # Image input and output event listener.
            im_input.change(
                preprocess_image,
                inputs=[im_input],
                outputs=[original_file, dimension_original]
            )
            # Image output listener.
            im_output.change(
                postprocess_image,
                inputs=[im_output],
                outputs=[dimension_upscaled])
            # Image upscale button listener. Changes from Tab to Tab!
            upscale_button_pil.click(
                fn=upscale_image_pil,
                inputs=[pil_method, scale_number, im_input],
                outputs=[im_output, time_value, ssim_value]
            )
            # Image flip listener.
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
            # Image rotate button listener.
            rotate_button_l.click(
                fn=rotate_image_l,
                inputs=[im_output],
                outputs=[im_output]
            )
            rotate_button_r.click(
                fn=rotate_image_r,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image sepia button listener.
            sepia_button.click(
                fn=sepia_filter,
                inputs=[im_output, sepia_number],
                outputs=[im_output]
            )
            # Image download button listener.
            download_button.click(
                fn=download_image,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Image sharpen & smoothing button listener.
            sharpen_button.click(
                fn=sharpen_image,
                inputs=[im_output, kernel_number],
                outputs=[im_output]
            )
            smoothing_button.click(
                fn=smoothing_image,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image denoising & gamma button listener.
            denoising_button.click(
                fn=denoising,
                inputs=[im_output, denoise_string],
                outputs=[im_output]
            )
            gamma_button.click(
                fn=gamma,
                inputs=[im_output, gamma_number],
                outputs=[im_output]
            )
            # Image brighten & contrast button listener.
            brighten_button.click(
                fn=brighten,
                inputs=[im_output, brightness_number],
                outputs=[im_output]
            )
            contrast_button.click(
                fn=contrast,
                inputs=[im_output, contrast_number, brightness_number],
                outputs=[im_output]
            )
            # Image inversion & grayscale button listener.
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
            # Image ssim button listener. For all Tabs equal!
            ssim_button.click(
                ssim_calc,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Create a clear & reset button.
            def clear_reset_pil():
                '''Clear & reset of interface'''
                return [pil_methods_list[0], scale_factors[0], kernel_list[0], 0, 1, 1, "(10,10,7,21)", "0"]
            clear_button = gr.ClearButton(components=clear_list, value="♻️  Clear & Reset")
            clear_button.click(
                fn=clear_reset_pil,
                inputs=[],
                outputs=[pil_method, scale_number, kernel_number, brightness_number, contrast_number, gamma_number, denoise_string, sepia_number]
            )
        # ************************************************************************
        # Scikit Section
        # ************************************************************************
        # Create a tab in the main block.
        with gr.Tab("Standard Methods (scikit-image)"):
            # ------------------
            # Components section
            # ------------------
            # Create a row in the tab.
            with gr.Row():
                upscale_button_sk = gr.Button(value=UPSCALE_ORIGINAL, scale=2)
                flip_button_h = gr.Button(value=HFLIP, scale=1)
                flip_button_v = gr.Button(value=VFLIP, scale=1)
                rotate_button_l = gr.Button(value=LROT, scale=1)
                rotate_button_r = gr.Button(value=RROT, scale=1)
                sepia_button = gr.Button(value=SEPIA_FILTER, scale=1)
                download_button = gr.Button(value=DOWNLOAD_IMAGE, scale=2)
            # Create a row in the tab.
            with gr.Row():
                #anti_alias = ["True", "False"]
                #anti_aliasing = gr.Dropdown(choices=anti_alias, value=anti_alias[0], label="Anti Aliasing", scale=1, min_width=100)
                scikit_method = gr.Dropdown(choices=scikit_method_list, value=scikit_method_list[0], label="Upscaling Methods", scale=2, min_width=190)
                scale_number = gr.Dropdown(choices=scale_factors, value=scale_factors[0], label="Scaling", scale=1, min_width=100)
                kernel_number = gr.Dropdown(choices=kernel_list, label="Sharpening Kernel", scale=0, min_width=140)
                brightness_number = gr.Number(value=0, label="Brightness", scale=1, interactive=True, min_width=100, step=0.1)
                contrast_number = gr.Number(value=1, label="Contrast", scale=1, interactive=True, min_width=100, step=0.1)
                gamma_number = gr.Number(value=1, label="Gamma", scale=1, interactive=True, min_width=100, step=0.1)
                denoise_string = gr.Textbox(value="(10,10,7,21)", max_lines=1, label="Denoising", scale=1, interactive=True, min_width=100)
                sepia_number = gr.Number(value="0", label="Sepia", scale=0, min_width=90, step=0.1)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    sharpen_button = gr.Button(value=SHARP_BUTTON, scale=1, min_width=60)
                    smoothing_button = gr.Button(value=SMOOTH_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    denoising_button = gr.Button(value=DENOISE_BUTTON, scale=1, min_width=60)
                    gamma_button = gr.Button(value=GAMMA_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    brighten_button = gr.Button(value=BRIGHT_BUTTON, scale=1, min_width=60)
                    contrast_button = gr.Button(value=CONTRAST_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    inversion_button = gr.Button(value=INVERT_BUTTON, scale=1, min_width=60)
                    grayscale_button = gr.Button(value=GRAYSCALE_BUTTON, scale=1, min_width=60)
            # Create a row in the tab.
            with gr.Row():
                # Create a column in the row.
                with gr.Column():
                    im_input = gr.Image(type='filepath', sources=['upload', 'clipboard'], height=512)
                    dimension_original = gr.TextArea(lines=1, label="Dimension Original Image")
                    original_file = gr.TextArea(lines=1, label="Original File Location", interactive=True)
                # Create a column in the row.
                with gr.Column():
                    im_output = gr.Image(height=512, interactive=False)
                    dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled Image")
                    # Create a row in the column.
                    with gr.Row():
                        time_value = gr.TextArea(lines=1, label="Elapsed Time in Seconds", interactive=True, min_width=100, scale=2)
                        ssim_value = gr.TextArea(lines=1, label="SSIM Value as Float", interactive=True, min_width=100, scale=2)
                        with gr.Column(min_width=250, scale=1):
                            ssim_dummy = gr.HTML(" ")
                            ssim_button = gr.Button(value=CALC_SSIM)
            clear_list = [im_input, im_output, dimension_original,
                          dimension_upscaled, time_value, ssim_value,
                          inter_method, scale_number, kernel_number,
                          brightness_number, contrast_number, gamma_number,
                          denoise_string, sepia_number]
            # ----------------------
            # Event listener section
            # ----------------------
            # Image input and output event listener.
            im_input.change(
               preprocess_image,
                inputs=[im_input],
                outputs=[original_file, dimension_original]
            )
            # Image output listener.
            im_output.change(
                postprocess_image,
                inputs=[im_output],
                outputs=[dimension_upscaled]
            )
            # Image upscale listener.
            upscale_button_sk.click(
                fn=upscale_image_scikit,
                inputs=[scikit_method, scale_number, im_input],
                outputs=[im_output, time_value, ssim_value]
            )
            # Image flip listener.
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
            # Image rotate button listener.
            rotate_button_l.click(
                fn=rotate_image_l,
                inputs=[im_output],
                outputs=[im_output]
            )
            rotate_button_r.click(
                fn=rotate_image_r,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image sepia button listener.
            sepia_button.click(
                fn=sepia_filter,
                inputs=[im_output, sepia_number],
                outputs=[im_output]
            )
            # Image download button listener.
            download_button.click(
                fn=download_image,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Image sharpen & smoothing button listener.
            sharpen_button.click(
                fn=sharpen_image,
                inputs=[im_output, kernel_number],
                outputs=[im_output]
            )
            smoothing_button.click(
                fn=smoothing_image,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image denoising & gamma button listener.
            denoising_button.click(
                fn=denoising,
                inputs=[im_output, denoise_string],
                outputs=[im_output]
            )
            gamma_button.click(
                fn=gamma,
                inputs=[im_output, gamma_number],
                outputs=[im_output]
            )
            # Image brighten & contrast button listener.
            brighten_button.click(
                fn=brighten,
                inputs=[im_output, brightness_number],
                outputs=[im_output]
            )
            contrast_button.click(
                fn=contrast,
                inputs=[im_output, contrast_number, brightness_number],
                outputs=[im_output]
            )
            # Image inversion & grayscale button listener.
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
            # Image ssim button listener. For all Tabs equal!
            ssim_button.click(
                ssim_calc,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Create a clear & reset button.
            def clear_reset_scikit():
                '''Clear & reset of interface'''
                return [scale_factors[0], kernel_list[0], 0, 1, 1, "(10,10,7,21)", "0"]
            clear_button = gr.ClearButton(components=clear_list, value=CLEAR_RESET)
            clear_button.click(
                fn=clear_reset_scikit,
                inputs=[],
                outputs=[scale_number, kernel_number, brightness_number, contrast_number, gamma_number, denoise_string, sepia_number]
            )
    # ************************************************************************
    # OpenCV Super Resolution Section
    # ************************************************************************
    if isSuperResolutionTab:
        # Create a Tab in the main block.
        with gr.Tab("Super Resolution (Protocol Buffer Models)"):
            # ------------------
            # Components section
            # ------------------
            # Create a row.
            with gr.Row():
                refresh_button = gr.Button(value=REFRESH_MODELS, min_width=20, scale=1)
                upscale_button_sr = gr.Button(value=UPSCALE_ORIGINAL, scale=2)
                flip_button_h = gr.Button(value=HFLIP, scale=1)
                flip_button_v = gr.Button(value=VFLIP, scale=1)
                rotate_button_l = gr.Button(value=LROT, scale=1)
                rotate_button_r = gr.Button(value=RROT, scale=1)
                sepia_button = gr.Button(value=SEPIA_FILTER, scale=1)
                download_button = gr.Button(value=DOWNLOAD_IMAGE, scale=1)
            # Create a row.
            with gr.Row():
                model_file = gr.Dropdown(choices=get_model_list(), value=_model_list[0], label="Model File List", scale=2, min_width=190)
                kernel_number = gr.Dropdown(choices=kernel_list, label="Sharpening Kernel", scale=0, min_width=140)
                brightness_number = gr.Number(value=0, label="Brightness", scale=1, interactive=True, min_width=100, step=0.1)
                contrast_number = gr.Number(value=1, label="Contrast", scale=1, interactive=True, min_width=100, step=0.1)
                gamma_number = gr.Number(value=1, label="Gamma", scale=1, interactive=True, min_width=100, step=0.1)
                denoise_string = gr.Textbox(value="(10,10,7,21)", max_lines=1, label="Denoising", scale=1, interactive=True, min_width=100)
                sepia_number = gr.Number(value="0", label="Sepia", scale=0, min_width=90, step=0.1)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    sharpen_button = gr.Button(value=SHARP_BUTTON, scale=1, min_width=60)
                    smoothing_button = gr.Button(value=SMOOTH_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    denoising_button = gr.Button(value=DENOISE_BUTTON, scale=1, min_width=60)
                    gamma_button = gr.Button(value=GAMMA_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    brighten_button = gr.Button(value=BRIGHT_BUTTON, scale=1, min_width=60)
                    contrast_button = gr.Button(value=CONTRAST_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    inversion_button = gr.Button(value=INVERT_BUTTON, scale=1, min_width=60)
                    grayscale_button = gr.Button(value=GRAYSCALE_BUTTON, scale=1, min_width=60)
            # Create a row in the tab.
            with gr.Row():
                # Create a column in the row.
                with gr.Column():
                    im_input = gr.Image(type='filepath', sources=['upload', 'clipboard'], height=512)
                    dimension_original = gr.TextArea(lines=1, label="Dimension Original Image")
                    original_file = gr.TextArea(lines=1, label="Original File Location", interactive=True)
                # Create a column in the row.
                with gr.Column():
                    im_output = gr.Image(height=512, interactive=False)
                    dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled Image")
                    # Create a row in the column.
                    with gr.Row():
                        time_value = gr.TextArea(lines=1, label="Elapsed Time in Seconds", interactive=True, min_width=100, scale=2)
                        ssim_value = gr.TextArea(lines=1, label="SSIM Value as Float", interactive=True, min_width=100, scale=2)
                        with gr.Column(min_width=250, scale=1):
                            ssim_dummy = gr.HTML(" ")
                            ssim_button = gr.Button(value=CALC_SSIM)
            # Define the components which should be cleared on reset.
            clear_list_sr = [im_input, im_output, dimension_original,
                             dimension_upscaled, time_value, ssim_value,
                             model_file, kernel_number,
                             brightness_number, contrast_number, gamma_number,
                             denoise_string, sepia_number]
            # ----------------------
            # Event listener section
            # ----------------------
            # Image input and output event listener.
            im_input.change(
                preprocess_image,
                inputs=[im_input],
                outputs=[original_file, dimension_original]
            )
            # Image output listener.
            im_output.change(
                postprocess_image,
                inputs=[im_output],
                outputs=[dimension_upscaled]
            )
            # Image upscale listener. Changes from Tab to Tab!
            upscale_button_sr.click(
                fn=upscale_image_sr,
                inputs=[model_file, im_input],
                outputs=[im_output, time_value, ssim_value]
            )
            # Refresh button listener. Tab specific.
            refresh_button.click(
                fn=refresh_list,
                inputs=[model_file],
                outputs=[model_file]
            )
            # Image flip listener.
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
            # Image rotate button listener.
            rotate_button_l.click(
                fn=rotate_image_l,
                inputs=[im_output],
                outputs=[im_output]
            )
            rotate_button_r.click(
                fn=rotate_image_r,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image sepia button listener.
            sepia_button.click(
                fn=sepia_filter,
                inputs=[im_output, sepia_number],
                outputs=[im_output]
            )
            # Image download button listener.
            download_button.click(
                fn=download_image,
                inputs=[im_output],
                outputs=[]
            )
            # Image sharpen & smoothing button listener.
            sharpen_button.click(
                 fn=sharpen_image,
                inputs=[im_output, kernel_number],
                outputs=[im_output]
            )
            smoothing_button.click(
                fn=smoothing_image,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image denoising & gamma button listener.
            denoising_button.click(
                fn=denoising,
                inputs=[im_output, denoise_string],
                outputs=[im_output]
            )
            gamma_button.click(
                fn=gamma,
                inputs=[im_output, gamma_number],
                outputs=[im_output]
            )
            # Image brighten & contrast button listener.
            brighten_button.click(
                fn=brighten,
                inputs=[im_output, brightness_number],
                outputs=[im_output]
            )
            contrast_button.click(
                fn=contrast,
                inputs=[im_output, contrast_number, brightness_number],
                outputs=[im_output]
            )
            # Image inversion & grayscale button listener.
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
            # Image ssim button listener. For all Tabs equal!
            ssim_button.click(
                ssim_calc,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Create a clear & reset button.
            def clear_reset_sr():
                '''Clear & reset of interface'''
                return [_model_list[0], kernel_list[0], 0, 1, 1, "(10,10,7,21)", "0"]
            clear_button = gr.ClearButton(components=clear_list_sr, value=CLEAR_RESET)
            clear_button.click(
                fn=clear_reset_sr,
                inputs=[],
                outputs=[model_file, kernel_number, brightness_number, contrast_number, gamma_number, denoise_string, sepia_number]
            )
    # ************************************************************************
    # Super Image Section
    # ************************************************************************
    if isSuperImageTab:
        # Create a Tab in the main block.
        with gr.Tab("Super Image (Pickle Tensor Models)"):
            # ------------------
            # Components section
            # ------------------
            # Create a row in the tab.
            with gr.Row():
                upscale_button_si = gr.Button(value=UPSCALE_ORIGINAL, scale=2)
                flip_button_h = gr.Button(value=HFLIP, scale=1)
                flip_button_v = gr.Button(value=VFLIP, scale=1)
                rotate_button_l = gr.Button(value=LROT, scale=1)
                rotate_button_r = gr.Button(value=RROT, scale=1)
                sepia_button = gr.Button(value=SEPIA_FILTER, scale=1)
                download_button = gr.Button(value=DOWNLOAD_IMAGE, scale=2)
            # Create a row in the tab.
            with gr.Row():
                superimage_model = gr.Dropdown(choices=si_list, value=si_list[0], label="Upscaling Methods", scale=2, min_width=190)
                superimage_scale = gr.Dropdown(choices=si_scale, value=si_scale[0], label="Scaling", scale=1, min_width=100)
                kernel_number = gr.Dropdown(choices=kernel_list, label="Sharpening Kernel", scale=0, min_width=140)
                brightness_number = gr.Number(value=0, label="Brightness", scale=1, interactive=True, min_width=100, step=0.1)
                contrast_number = gr.Number(value=1, label="Contrast", scale=1, interactive=True, min_width=100, step=0.1)
                gamma_number = gr.Number(value=1, label="Gamma", scale=1, interactive=True, min_width=100, step=0.1)
                denoise_string = gr.Textbox(value="(10,10,7,21)", max_lines=1, label="Denoising", scale=1, interactive=True, min_width=100)
                sepia_number = gr.Number(value="0", label="Sepia", scale=0, min_width=90, step=0.1)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    sharpen_button = gr.Button(value=SHARP_BUTTON, scale=1, min_width=60)
                    smoothing_button = gr.Button(value=SMOOTH_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    denoising_button = gr.Button(value=DENOISE_BUTTON, scale=1, min_width=60)
                    gamma_button = gr.Button(value=GAMMA_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    brighten_button = gr.Button(value=BRIGHT_BUTTON, scale=1, min_width=60)
                    contrast_button = gr.Button(value=CONTRAST_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    inversion_button = gr.Button(value=INVERT_BUTTON, scale=1, min_width=60)
                    grayscale_button = gr.Button(value=GRAYSCALE_BUTTON, scale=1, min_width=60)
            # Create a row in the tab.
            with gr.Row():
                # Create a column in the row.
                with gr.Column():
                    im_input = gr.Image(type='filepath', sources=['upload', 'clipboard'], height=512)
                    dimension_original = gr.TextArea(lines=1, label="Dimension Original Image")
                    original_file = gr.TextArea(lines=1, label="Original File Location", interactive=True)
                # Create a column in the row.
                with gr.Column():
                    im_output = gr.Image(height=512, interactive=False)
                    dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled Image")
                    # Create a row in the column.
                    with gr.Row():
                        time_value = gr.TextArea(lines=1, label="Elapsed Time in Seconds", interactive=True, min_width=100, scale=2)
                        ssim_value = gr.TextArea(lines=1, label="SSIM Value as Float", interactive=True, min_width=100, scale=2)
                        with gr.Column(min_width=250, scale=1):
                            ssim_dummy = gr.HTML(" ")
                            ssim_button = gr.Button(value=CALC_SSIM)
            clear_list = [im_input, im_output, dimension_original,
                          dimension_upscaled, time_value, ssim_value,
                          inter_method, scale_number, kernel_number,
                          brightness_number, contrast_number, gamma_number,
                          denoise_string, sepia_number]
            # ----------------------
            # Event listener section
            # ----------------------
            # Image input and output event listener.
            # Consecutively running image input event listener.
            im_input.change(
                preprocess_image,
                inputs=[im_input],
                outputs=[original_file, dimension_original]
            )
            # Image output listener.
            im_output.change(
                postprocess_image,
                inputs=[im_output],
                outputs=[dimension_upscaled])
            # Image upscale listener.
            upscale_button_si.click(
                fn=upscale_image_si,
                inputs=[im_input, superimage_model, superimage_scale],
                outputs=[im_output, time_value, ssim_value]
            )
            # Image flip button listener.
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
            # Image rotate button listener.
            rotate_button_l.click(
                fn=rotate_image_l,
                inputs=[im_output],
                outputs=[im_output]
            )
            rotate_button_r.click(
                fn=rotate_image_r,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image sepia button listener.
            sepia_button.click(
                fn=sepia_filter,
                inputs=[im_output, sepia_number],
                outputs=[im_output]
            )
            # Image download button listener.
            download_button.click(
                fn=download_image,
                inputs=[im_output],
                outputs=[]
            )
            # Image sharpen & smoothing button listener.
            sharpen_button.click(
                fn=sharpen_image,
                inputs=[im_output, kernel_number],
                outputs=[im_output]
            )
            smoothing_button.click(
                fn=smoothing_image,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image denoising & gamma button listener.
            denoising_button.click(
                fn=denoising,
                inputs=[im_output, denoise_string],
                outputs=[im_output]
            )
            gamma_button.click(
                fn=gamma,
                inputs=[im_output, gamma_number],
                outputs=[im_output]
            )
            # Image brighten & contrast button listener.
            brighten_button.click(
                fn=brighten,
                inputs=[im_output, brightness_number],
                outputs=[im_output]
            )
            contrast_button.click(
                fn=contrast,
                inputs=[im_output, contrast_number, brightness_number],
                outputs=[im_output]
            )
            # Image inversion & grayscale button listener.
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
            # Image ssim button listener. For all Tabs equal!
            ssim_button.click(
                ssim_calc,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Create a clear & reset button.
            def clear_reset_si():
                '''Clear & reset of interface'''
                return [si_list[0], si_scale[0], kernel_list[0], 0, 1, 1, "(10,10,7,21)", "0"]
            clear_button = gr.ClearButton(components=clear_list, value=CLEAR_RESET)
            clear_button.click(
                fn=clear_reset_si,
                inputs=[],
                outputs=[superimage_model, superimage_scale, kernel_number, brightness_number, contrast_number, gamma_number, denoise_string, sepia_number]
            )
    # ************************************************************************
    # Stable Diffusion Section
    # ************************************************************************
    if isStableDiffusionTab:
        # Create a Tab in the main block.
        with gr.Tab("Stable Diffusion (Diffusion Models)"):
            # ------------------
            # Components section
            # ------------------
            # Create a row in the tab.
            with gr.Row():
                upscale_button_sd = gr.Button(value=UPSCALE_ORIGINAL, scale=2)
                flip_button_h = gr.Button(value=HFLIP, scale=1)
                flip_button_v = gr.Button(value=VFLIP, scale=1)
                rotate_button_l = gr.Button(value=LROT, scale=1)
                rotate_button_r = gr.Button(value=RROT, scale=1)
                sepia_button = gr.Button(value=SEPIA_FILTER, scale=1)
                download_button = gr.Button(value=DOWNLOAD_IMAGE, scale=2)
            # Create a row in the tab.
            with gr.Row():
                inter_method = gr.Dropdown(choices=sd_list, value=sd_list[0], label="Upscaling Methods", scale=2, min_width=190)
                kernel_number = gr.Dropdown(choices=kernel_list, label="Sharpening Kernel", scale=0, min_width=140)
                brightness_number = gr.Number(value=0, label="Brightness", scale=1, interactive=True, min_width=100, step=0.1)
                contrast_number = gr.Number(value=1, label="Contrast", scale=1, interactive=True, min_width=100, step=0.1)
                gamma_number = gr.Number(value=1, label="Gamma", scale=1, interactive=True, min_width=100, step=0.1)
                denoise_string = gr.Textbox(value="(10,10,7,21)", max_lines=1, label="Denoising", scale=1, interactive=True, min_width=100)
                sepia_number = gr.Number(value="0", label="Sepia", scale=0, min_width=90, step=0.1)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    sharpen_button = gr.Button(value=SHARP_BUTTON, scale=1, min_width=60)
                    smoothing_button = gr.Button(value=SMOOTH_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    denoising_button = gr.Button(value=DENOISE_BUTTON, scale=1, min_width=60)
                    gamma_button = gr.Button(value=GAMMA_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    brighten_button = gr.Button(value=BRIGHT_BUTTON, scale=1, min_width=60)
                    contrast_button = gr.Button(value=CONTRAST_BUTTON, scale=1, min_width=60)
                # Create a column in the row.
                with gr.Column(scale=0, min_width=170):
                    inversion_button = gr.Button(value=INVERT_BUTTON, scale=1, min_width=60)
                    grayscale_button = gr.Button(value=GRAYSCALE_BUTTON, scale=1, min_width=60)
            # Create a row in the tab.
            with gr.Row():
                # Create a column in the row.
                with gr.Column():
                    im_input = gr.Image(type='filepath', sources=['upload', 'clipboard'], height=512)
                    dimension_original = gr.TextArea(lines=1, label="Dimension Original Image")
                    original_file = gr.TextArea(lines=1, label="Original File Location", interactive=True)
                # Create a column in the row.
                with gr.Column():
                    im_output = gr.Image(height=512, interactive=False)
                    dimension_upscaled = gr.TextArea(lines=1, label="Dimension Upscaled Image")
                    # Create a row in the column.
                    with gr.Row():
                        time_value = gr.TextArea(lines=1, label="Elapsed Time in Seconds", interactive=True, min_width=100, scale=2)
                        ssim_value = gr.TextArea(lines=1, label="SSIM Value as Float", interactive=True, min_width=100, scale=2)
                        with gr.Column(min_width=250, scale=1):
                            ssim_dummy = gr.HTML(" ")
                            ssim_button = gr.Button(value=CALC_SSIM)
            clear_list = [im_input, im_output, dimension_original,
                          dimension_upscaled, time_value, ssim_value,
                          inter_method, scale_number, kernel_number,
                          brightness_number, contrast_number, gamma_number,
                          denoise_string, sepia_number]
            # ----------------------
            # Event listener section
            # ----------------------
            # Image input and output event listener.
            im_input.change(
                preprocess_image,
                inputs=[im_input],
                outputs=[original_file, dimension_original]
            )
            im_output.change(
                postprocess_image,
                inputs=[im_output],
                outputs=[dimension_upscaled])
            # Image upscale button listener. Adapt to Upscaler!
            upscale_button_sd.click(
                fn=upscale_image_sd,
                inputs=[im_input, inter_method],
                outputs=[im_output, time_value, ssim_value]
            )
            # Image flip button listener.
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
            # Image rotate button listener.
            rotate_button_l.click(
                fn=rotate_image_l,
                inputs=[im_output],
                outputs=[im_output]
            )
            rotate_button_r.click(
                fn=rotate_image_r,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image sepia button listener.
            sepia_button.click(
                fn=sepia_filter,
                inputs=[im_output, sepia_number],
                outputs=[im_output]
            )
            # Image download button listener.
            download_button.click(
                fn=download_image,
                inputs=[im_output],
                outputs=[]
            )
            # Image sharpen & smoothing button listener.
            sharpen_button.click(
                fn=sharpen_image,
                inputs=[im_output, kernel_number],
                outputs=[im_output]
            )
            smoothing_button.click(
                fn=smoothing_image,
                inputs=[im_output],
                outputs=[im_output]
            )
            # Image denoising & gamma button listener.
            denoising_button.click(
                fn=denoising,
                inputs=[im_output, denoise_string],
                outputs=[im_output]
            )
            gamma_button.click(
                fn=gamma,
                inputs=[im_output, gamma_number],
                outputs=[im_output]
            )
            # Image brighten & contrast button listener.
            brighten_button.click(
                fn=brighten,
                inputs=[im_output, brightness_number],
                outputs=[im_output]
            )
            contrast_button.click(
                fn=contrast,
                inputs=[im_output, contrast_number, brightness_number],
                outputs=[im_output]
            )
            # Image inversion & grayscale button listener.
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
            # Image ssim button listener. For all Tabs equal!
            ssim_button.click(
                ssim_calc,
                inputs=[im_output],
                outputs=[ssim_value]
            )
            # Create a clear & reset button.
            def clear_reset_sd():
                '''Clear & reset of interface'''
                return [inter_methods_list[0], scale_factors[0], kernel_list[0], 0, 1, 1, "(10,10,7,21)", "0"]
            clear_button = gr.ClearButton(components=clear_list, value=CLEAR_RESET)
            clear_button.click(
                fn=clear_reset_sd,
                inputs=[],
                outputs=[inter_method, scale_number, kernel_number, brightness_number, contrast_number, gamma_number, denoise_string, sepia_number]
            )
    # Create a footer line.
    with gr.Row():
        # Define a multiline text string.
        footer_text = '''\
                      <div style='line-height:1.25;margin:auto;text-align:center;
                      display:block;vertical-align:middle;font-size:14px;height:32px;
                      width:auto;'>Lazy Image Upscaler. {version}. Usage at your own risk!
                      </div><div style='line-height:1.25;margin:auto;text-align:center;
                      display:block;vertical-align:middle;font-size:14px;height:32px;width:auto;'>
                      ©️  Copyright 2024, zentrocdot. All rights reserved.</div></div>\
                      '''.format(version=__Version__)
        # Create a HTML component.
        gr.HTML(footer_text)

# --------------------
# Main script function
# --------------------
def main():
    '''Main script function.'''
    # Start the web ui.
    webui.launch(server_name="127.0.0.1", server_port=7865)

# Execute the script as module or as programme.
if __name__ == "__main__":
    # Call main function.
    main()
