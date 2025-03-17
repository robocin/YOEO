# YOEO — You Only Encode Once
A CNN for Embedded Object Detection and Semantic Segmentation

# This project is based upon [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) and will continuously be modified to our needs.


<img src="https://user-images.githubusercontent.com/15075613/131497667-c4e3f35f-db4b-4a53-b816-32dac6e1d85d.png" alt="example_image" height="150"/><img src="https://user-images.githubusercontent.com/15075613/131497744-e142c4ed-d69b-419a-96c3-39d871796081.png" alt="example_image" height="150"/><img src="https://user-images.githubusercontent.com/15075613/131498064-fc6545d9-8a1d-4953-a80b-52a3d2293c83.jpg" alt="example_image" height="150"/><img src="https://user-images.githubusercontent.com/15075613/131499391-e14a968a-b403-4210-b5f7-eb9be90a61db.png" alt="example_image" height="150"/>

<img src="https://user-images.githubusercontent.com/15075613/131554376-1a0e5560-5aa6-462a-afb1-c0eeb0de5a4a.png" alt="example_image" height="150"/><img src="https://user-images.githubusercontent.com/15075613/131502742-bcc588b1-e766-4f0b-a2c4-897c14419971.png" alt="example_image" height="150"/><img src="https://user-images.githubusercontent.com/15075613/131503205-cbf47af6-8bfb-44f1-bbcb-37fdf54f139d.png" alt="example_image" height="150"/><img src="https://user-images.githubusercontent.com/15075613/131502830-e060e113-4abc-413a-bbdc-ffa6994b6a11.png" alt="example_image" height="150"/>


## Installation
### Installing from source

For normal training and evaluation we recommend installing the package from source using a poetry virtual environment.

```bash
git clone https://github.com/bit-bots/YOEO
cd YOEO/
pip3 install poetry --user
poetry install
```

You need to join the virtual environment by running `poetry shell` in this directory before running any of the following commands without the `poetry run` prefix.
Also have a look at the other installing method, if you want to use the commands everywhere without opening a poetry-shell.

#### Download pretrained weights

```bash
./weights/download_weights.sh
```

## Test
Evaluates the model on the test dataset.
See help page for more details.

```bash
poetry run yoeo-test -h
```

## Inference
Uses pretrained weights to make predictions on images.

```bash
poetry run yoeo-detect --images data/samples/
```

<p align="center"><img src="https://user-images.githubusercontent.com/15075613/131503350-3e232e91-016b-4034-8bda-15e6619b0f98.png" width="480"\></p>


## API

You are able to import the modules of this repo in your own project if you install this repo as a python package.

An example prediction call from a simple OpenCV python script would look like this:

```python
import cv2
from yoeo import detect, models

# Load the YOEO model
model = models.load_model(
  "<PATH_TO_YOUR_CONFIG_FOLDER>/yoeo.cfg",
  "<PATH_TO_YOUR_WEIGHTS_FOLDER>/yoeo.pth")

# Load the image as a numpy array
img = cv2.imread("<PATH_TO_YOUR_IMAGE>")

# Convert OpenCV bgr to rgb
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Runs the YOEO model on the image
boxes, segmentation = detect.detect_image(model, img)

print(boxes)
# Output will be a numpy array in the following format:
# [[x1, y1, x2, y2, confidence, class]]

print(segmentation)
# Output will be a 2d numpy array with the coresponding class id in each cell
```

For more advanced usage look at the method's doc strings.

## Convert your YOEO model

### Convert your YOEO model to an ONNX model

To convert your YOEO model to an ONNX model, you can use the following command:

```bash
poetry run yoeo-to-onnx config/yoeo.cfg config/yoeo.pth # Replace path with your .cfg and .pth file
```

For more information on ONNX, read the [ONNX runtime website](https://onnxruntime.ai/).

### Testing the ONNX Model 

```bash
python3 yoeo/run_onnx.py data/samples/frame6554.jpg config/yoeo.onnx
```
#### Parameters:
- `data/samples/frame6554.jpg`: This is the path to the input image you want to test the model with.
- `config/yoeo.onnx`: Make sure to replace this with the actual path to your ONNX model file.


### Convert ONNX model to TensorRT

After successful conversion of your YOEO model to an ONNX model using [this guide](#convert-your-yoeo-model-to-an-onnx-model), you can move on with the next conversion to an TensorRT model model using the following command.  However, before converting to TensorRT, you need to modify the ONNX model due to compatibility issues with certain operations, the a `Cast` operation from `float` to `int`, which TensorRT does not support.

Here is how you can work around this issue:

```bash
python3 onnx_int2float.py 
```

 The onnx2trt.sh script invokes the trtexec, which is a tool provided by TensorRT for converting ONNX models into optimized TensorRT engines. Use the following command to convert the yoeo_fixed.onnx model to TensorRT:
```bash
$ sh onnx2trt.sh
```

For more information on trtexec, read the [NVIDIA documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html).

## Publication

### YOEO — You Only Encode Once: A CNN for Embedded Object Detection and Semantic Segmentation

**Abstract** <br>
Fast and accurate visual perception utilizing a robot’s limited hardware resources is necessary for many mobile robot applications.
We are presenting YOEO, a novel hybrid CNN which unifies previous object detection and semantic segmentation approaches using one shared encoder backbone to increase performance and accuracy.
We show that it outperforms previous approaches on the TORSO-21 and Cityscapes datasets.

[[ResearchGate]](https://www.researchgate.net/publication/356873226_YOEO_-_You_Only_Encode_Once_A_CNN_for_Embedded_Object_Detection_and_Semantic_Segmentation)
 [[Download]](https://www.researchgate.net/profile/Marc-Bestmann/publication/356873226_YOEO_-_You_Only_Encode_Once_A_CNN_for_Embedded_Object_Detection_and_Semantic_Segmentation/links/61b0c82d1a5f480388c36100/YOEO-You-Only-Encode-Once-A-CNN-for-Embedded-Object-Detection-and-Semantic-Segmentation.pdf)

```
@inproceedings{vahlyoeo,
  title={YOEO — You Only Encode Once: A CNN for Embedded Object Detection and Semantic Segmentation},
  author={Vahl, Florian and Gutsche, Jan and Bestmann, Marc and Zhang, Jianwei},
  year={2021},
  organization={IEEE},
  booktitle={2021 IEEE International Conference on Robotics and Biomimetics (ROBIO)}
}
```
