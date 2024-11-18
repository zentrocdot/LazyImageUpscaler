# Lazy Image Upscaler

## Preface

<p align="justify">The Lazy Image Upscaler is a web user interface for the upscaling of images. At the moment the Lazy Image Upscaler offers five possibilities to upscale an image. One can use standard methods from OpenCV and PIL to upscale images. Some references states that this is not working well. My experience is different to this statement. From my point of view the results using these standard methods are sufficient for most cases. The third and fourth method are using Machine Learning approaches. The upscaling is done using machine learning methods together with pretrained models. To be able to work with the web user interface, minimum one pretrained .pb model is required for the third method. At OpenCV one can find the links for downloading such pretrained models [1]. These pretrained models can also be found in [2-5]. The fourth method is using pretrained model which can be found at Hugging Face. The last method is using Stable Diffusion.</p>

## Introduction

<p align="justify">The tabs are arranged according to Logic:</p>

* Standard methods (nuerical interpolation methods)
* Methods using pretrained model from Machine Learning
* Stable Diffusion Upscaler model

## Installation

<p align="justify">Clone this repository to a local location of of your choice. Then you need some pertrained models, which has to placed in the directory resources. After that yu are ready to work with the web UI.</p>

## Start

<p align="justify">Use <code>start_webui.bash</code> in the main directory to start the local server. If there is a problem one can move into the subdirectory 
scripts. From there <code>lazy_image_upscaler.py</code> can be started.</p>

<p align="justify">Open a webbrowser and open localhost on</p>

<pre>http://127.0.0.1:7860</pre>

<p align="justify">If everything was okay so far, the web UI starts in the browser windwow.<p align="justify">

## OpenCV

* INTER_NEAREST
* INTER_LINEAR
* INTER_AREA
* INTER_CUBIC
* INTER_CUBIC  

## Pretrained Models

<p align="justify">Pretrained models which can be used are:<p align="justify">

* EDSR
* ESPCN
* FSRCNN
* LAPSRN

## Web UI

<p align="justify">The web UI is simple to use. One can select the pretrained model from a dropdown list. Per drag and drop or per upload the image can be loaded. Using the Upscale Image button scales the image up. Download Image downloads the image to the local storage.</p>

<a target="_blank" href=""><img src="./images/lazyimageupscaler.png" alt="button panel"></a>

## Download Images

<p align="justify">When a download is done image names looks like:</p>

2024-11-13_16:32:48.931703.jpg

<p align="justify">To make sure that each image is unique I am using the date, the time and the remaining microseconds as filename. The formatstring is:</p>

<pre>"%Y-%m-%d_%H:%M:%S.%f"</pre>

<p align="justify">The downloaded images can be found in the folder <code>outputs</code>.</p>

## Repository Structure

The repository structure is as follows

```
LazyImageUpscaler
    ├── scripts
    ├── resources
    ├── outputs
    ├── test_images
    ├── upscaler_examples
    └── images
```

<p align="justify">In the folder <code>scripts</code> there are four Python scripts, which can be used to download the models into the <code>resources</code> folder directly.</p>

<p align="justify">Under the main branch there are four directories. In scripts are the Python scripts for the web user interface. In resources there can be the .pb models be stored. After the installation this directory is empty. Created images are saved in outputs. images is the directory where documentation related images are stored.</p>

## To_Do 

<p align="justify">Improvement of this documentation. The web UI has to checked that it is more fail safe. The current work was quick and dirty programming. I need to sanitize and optimize the code.</p>

<p align="justify">I need a third Tab in the interface, to use Stable Diffusion for upscaling. Then I am able to compare all these methods togeter in on web UI.</p>

## Test Environment

<p align="justify">I developed and tested the Python script with following specification:</p>

* Linux Mint 21.3 (Virginia)
* Python 3.10.14
* OpenCV 4.10.0
* PIL 10.4.0
* Gradio 5.0.1
* Torch 2.4.1+cu121
* Numpy  1.26.4
* Chromium Browser (and others)
* Monitor with a resolution of 1366 x 768 pixel

## Time Consumption

The numerical approaches are the fastest. The AI approach is the one which takes the most time.

## Power consumption

The numerical approaches use the CPU and not the GPU, so this approach saves energy. Machine Learning
and AI use the GPU extensivly and have a high power consumption.

## Limitations

<p align="justify">In the Machine Learning Tabs and in the AI Tab there are input images 
larger than 512 x 512 pixel problematic.</p>

## Known Problems

<p align="justify">The critical parts of the software are the parts that use the GPU. In this
sense, three upscaling approaches are critical. Two approaches are using pretrained models from
Machine Learning and one is the well know AI approach.</p>

<p align="justify">Common errors if on talks about the GPU usages are.</p>

* RuntimeError
* OutOfMemorError 

## Troubleshooting

### Super Resolution

The error message

<code>module cv2.cv2 has no attribute dnn_superres</code>

or similiar error messages can be handeled as desribed bwlow. This error occured appeared from one moment to the next 
without me being able to understand why the error message occurred.

Following solved this problem:

<pre>
pip uninstall opencv-python
pip uninstall opencv-contrib-python
</pre>

Then install latest version of OpenCV with pip3:

<pre>
pip3 install opencv-contrib-python
</pre>

### Web UI

<p align="justify">In the case of unexpected persistent problems, shut down the Gradio
server in the terminal window. After relaunch of the server, refresh the Browser window.</p>

<p align="justify">In the case of unexpected persistent problems, shut down the Gradio
server in the terminal window. After relaunch of the server, refresh the Browser window.</p>

<p align="justify">If there is a problem with the server and with the port, one can chnage both values in the source code e.g. from.</p>

<code>webui.launch(server_name="127.0.0.1", server_port=7865)</code>

to 

<code>webui.launch()</code>

### Super Image

Error:

ImportError: cannot import name 'cached_download' from 'huggingface_hub'

Possible Solution:

pip install huggingface_hub==0.25.00
pip3 install -U sentence-transformers

## Installations Prerequisites

<p align="justify">Following Python requirements have to be fulfilled, that the Lazy Image upscaler is working:</p>

* gradio
* pil
* opencv
* piexif
* diffuser
* super-image

I am assuming that PIP is installed. Installation:

<pre>pip3 install upgrade pip</pre>

<pre>pip3 install gradio</pre>
<pre>pip3 install pillow</pre>
<pre>pip3 install opencv-contrib-python</pre>    
<pre>pip3 install piexif</pre>    
<pre>pip3 install diffuser</pre>
<pre>pip3 install super_image</pre>
<pre>pip3 install transformer</pre>

Local installation can be found in hidden directory .local in the user's main directory. Changes here may result in problems while running the application.

## Stable Diffusion Upscaler Model

Move to directory LazyImageUpscaler/stabilityai

<code>
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install

git clone https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

# If you want to clone without large files - just their pointers
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler
</code>

## References

[1] https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres

[2] https://github.com/Saafke/EDSR_Tensorflow/tree/master/models

[3] https://github.com/fannymonori/TF-ESPCN/tree/master/export

[4] https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models

[5] https://github.com/fannymonori/TF-LapSRN/tree/master/export

[6] https://github.com/cyc0102/opencv_super_resolution/tree/master

[7] https://huggingface.co/

[8] https://pypi.org/project/super-image/

[9] https://huggingface.co/stabilityai/sd-x2-latent-upscaler

[10] https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler

[11] https://pillow.readthedocs.io/en/stable/reference/Image.html
