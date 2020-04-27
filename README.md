# Person detection from live cctv video per second

The project developed using TensorFlow to detect the no of people every second entering building gate. `work in progress`

### Python Packages Needed

* <a href='https://github.com/tensorflow/tensorflow'>Tensorflow</a><br>
* <a href='https://github.com/skvark/opencv-python'>openCV</a><br>
* you must have pip and python pre installed if not install them first.

## Installation instructions
``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```
For Ubuntu
``` bash
sudo apt-get install protobuf-compiler
```
For OSX
```
brew install protobuf
```
Other Libraries
```
pip install opencv-python
```
Tensorflow object detection API
```
protoc utils/*.proto --python_out=.
```

## Running
If you want to test out the implementation then you can use the person_detection_every_sec.py which is working on pre recorded video<br/>
```
python person_detection_every_sec.py
```

### Instruction to plot bounding boxes
As per the original implementation of the tensorflow object detection API, the bounding boxes are normalised. To get the original dimensions you need to do the following.

```
(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                              ymin * im_height, ymax * im_height)
```
* Download and use, no license needed*