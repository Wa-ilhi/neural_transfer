This repository contains a pytorch implementation of an algorithm for artistic neural style transfer. The algorithm can be used to mix the content of an image with the style of another image.

## Requirements

The program is written in Python, and uses [pytorch](http://pytorch.org/), [scipy](https://www.scipy.org). A GPU is not necessary, but can provide a significant speed up especially for training a new model. Regular sized images can be styled on a laptop or desktop using saved models.

The program runs in Streamlit, an open-source python-based library designed for machine learning. (https://docs.streamlit.io/)
### Required Libraries
* Pytorch
* Scipy
* Streamlit

## Installation

To install Streamlit, run this command
```
pip install streamlit
```
To install pytorch
```
pip install torch torchvision
```
To install scipy
```
pip install scikit-learn
```
## Features
* **Train a Style Transfer Model:** Train a model to learn the style of a given image using a dataset of content images.
* **Apply Style Transfer:** Use a pre-trained model to apply artistic style transfer to your content images.
* **Streamlit Interface:** An interactive web application to upload content images and apply style transfer using pre-trained models.
## Usage
### Training a model
To train a new style transfer model, you need to have a dataset of content images and a style image. Use the following command to start training:
```
python style_transfer.py train --dataset path/to/content_images --style-image path/to/style_image.jpg --save-model-dir path/to/save_model --cuda 1
```
### Training Parameters to consider
* **--epochs:** Number of training epochs (default: 2).
* **--batch-size:** Batch size for training (default: 4).
* **--dataset: **Path to the dataset of content images.
* **--style-image:** Path to the style image.
* **--save-model-dir:** Directory to save the trained model.
* **--checkpoint-model-dir:** Directory to save model checkpoints.
* **--image-size:** Size of training images (default: 256x256).
* **--style-size:** Size of the style image (default: original size).
* **--cuda:** Use GPU for training (1 for GPU, 0 for CPU).
* **--seed:** Random seed for training (default: 42).
* **--content-weight:** Weight for content loss (default: 1e5).
* **--style-weight:** Weight for style loss (default: 1e10).
* **--lr:** Learning rate (default: 1e-3).
* **--log-interval:** Interval for logging training loss (default: 500).
* **--checkpoint-interval:** Interval for saving model checkpoints (default: 2000).
### Applying an Style
To apply style transfer to an image using a pre-trained model, use the following command:
```
python style_transfer.py eval --content-image path/to/content_image.jpg --output-image path/to/output_image.jpg --model path/to/trained_model.pth --cuda 1

```
### Evaluation Parameters to consider
* **--content-image:** Path to the content image you want to stylize.
* **--content-scale:** Factor for scaling down the content image (optional).
* **--output-image:** Path for saving the output image.
* **--model:** Path to the saved model to be used for stylizing the image.
* **--cuda:** Use GPU for stylizing (1 for GPU, 0 for CPU).
* **--export-onnx:** Export model to ONNX format (optional).
## Running the Streamlit App
This will open a web interface where you can upload a content image and apply a style transfer using a pre-trained model.
```
streamlit run <yourfile.py>
```
