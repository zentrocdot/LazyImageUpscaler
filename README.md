# LazyUpscalerGui

## Preface

<p align="justify">LazyUpscalerGui is a web UI for the upscaling of images. To be able to work with the web UI, pretrained .pb models are required. At OpenCV you will find the links to download the models [1]. The model can also be found in [2-5]. The current version is the inital version for the web UI.</p>

## Installation

<p align="justify">Clone this repository to a local location of of your choice. Then you need some pertrained models, which has to placed in the directory resources. After that yu are ready to work with the web UI.</p>

## Start

<p align="justify">Open a webbrowser and open localhost on http://127.0.0.1:7860. If everything was okay so far, the web UI starts in the browser windwow.</p>

## Pretrained Models

Pretrained models which can be used are:

* edsr
* espcn
* fsrcnn
* lapsrn

## References

[1] https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres

[2] https://github.com/Saafke/EDSR_Tensorflow/tree/master/models

[3] https://github.com/fannymonori/TF-ESPCN/tree/master/export

[4] https://github.com/Saafke/FSRCNN_Tensorflow/tree/master/models

[5] https://github.com/fannymonori/TF-LapSRN/tree/master/export

[6] https://github.com/cyc0102/opencv_super_resolution/tree/master
