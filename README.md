# Lazy Image Upscaler

## Preface

<p align="justify">The Lazy Image Upscaler is a web user interface for the upscaling of images. The upscaling is done using machine learning methods using pretrained models. To be able to work with the web user interface, minimum one pretrained .pb model is required. At OpenCV one can find the links for downloading such pretrained models [1]. These pretrained models can also be found in [2-5]. The current version of the Lazy Image Upscaler is the inital version for the web user interface.</p>

## Installation

<p align="justify">Clone this repository to a local location of of your choice. Then you need some pertrained models, which has to placed in the directory resources. After that yu are ready to work with the web UI.</p>

## Start

<p align="justify">Open a webbrowser and open localhost on</p>

<pre>http://127.0.0.1:7860</pre>

<p align="justify">If everything was okay so far, the web UI starts in the browser windwow.</p>

## Pretrained Models

Pretrained models which can be used are:

* edsr
* espcn
* fsrcnn
* lapsrn

## Web UI

The web UI is simple to use. One can select the pretrained model from a dropdown list. Per drag and drop or per upload the image can be loaded. Using the Upscale Image button scales the image up. Download Image downloads the image to the local storage.

<a target="_blank" href=""><img src="./images/lazyimageupscaler.png" alt="button panel"></a>

## Repository Structure

The repository structure is as follows

```LazyImageUpscaler
    ├── scripts
    ├── resources
    ├── outputs
    ├── testimages
    └── images
```
In the folder <pre>scripts</pre> there are four Python scripts, which can be used to download the models into the <pre>resources</pre> folder directly.

Under the main branch there are four directories. In scripts are the Python scripts for the web user interface. In resources there can be the .pb models be stored. After the installation this directory is empty. Created images are saved in outputs. images is the directory where documentation related images are stored.

## References

[1] https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres

[2] https://github.com/Saafke/EDSR_Tensorflow/tree/master/models

[3] https://github.com/fannymonori/TF-ESPCN/tree/master/export

[4] https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models

[5] https://github.com/fannymonori/TF-LapSRN/tree/master/export

[6] https://github.com/cyc0102/opencv_super_resolution/tree/master
