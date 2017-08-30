# Style-transfer
## Examples:
Base Image / Style Image:
<div align="center">
<img src="https://raw.githubusercontent.com/seuphor/vgg19-style/master/examples/base1.jpg" height="250px">
<img src="https://raw.githubusercontent.com/seuphor/vgg19-style/master/examples/style3.jpg" height="250px">

<img src="https://raw.githubusercontent.com/seuphor/vgg19-style/master/examples/base1.png" height="250px">
<img src="https://raw.githubusercontent.com/seuphor/vgg19-style/master/examples/shinkai.png" height="250px">
</div>
<hr>
Stylized Images:

<div align="center">
<img src="https://raw.githubusercontent.com/seuphor/vgg19-style/master/examples/stylized1.jpg" height="250px">
<img src="https://raw.githubusercontent.com/seuphor/vgg19-style/master/examples/stylized2.jpg" height="250px">
</div>

## Setup
#### Dependencies:
* [tensorflow](https://github.com/tensorflow/tensorflow)
* [opencv](http://opencv.org/downloads.html)

#### After installing the dependencies: 
* Download the [VGG-19 model weights](http://www.vlfeat.org/matconvnet/pretrained/) (see the "VGG-VD models from the *Very Deep Convolutional Networks for Large-Scale Visual Recognition* project" section). 
* Copy the weights file `imagenet-vgg-verydeep-19.mat` to the project directory.

## Acknowledgements

The implementation is based on the projects: 
* A Neural Algorithm of Artistic Style by [Leon A. Gatys et al.](https://arxiv.org/pdf/1508.06576.pdf)
* Python implementation 'neural-style' by [Siraj](https://github.com/llSourcell/How_to_do_style_transfer_in_tensorflow)
