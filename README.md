# Optimal Stereo Camera Calibration via Genetic Algorithms

> Demonstration code accompanying our submission to the IJCAI 2021 AI4AD [workshop](https://www.ai4ad.net).

## Table of Contents

* [General Info](#general-information)
* [Packages](#packages)
* [Features](#features)
* [Usage](#usage)
* [Citation](#citation)
* [Contact](#contact)

## General Information

- Abstract: Stereo camera real world measurement accuracy is strongly dependent on the camera calibration process,
  which, in turn, relies on the set of input images and camera model. There is a strong relationship between the quality
  of the input, the calibration process and the accuracy of the 3D reconstruction. The total set of calibration images
  has a strong influence on the calibration outcome and selecting the optimal subset of images and parameters for
  calibration estimation is a problem of combinatorial complexity. In this paper we propose a genetic algorithm
  calibration image selection process, driven by multiple metrics, that eliminates the image and parameter selection
  bias and avoids exploratory search over the entire solution space. We numerically compare the contribution of our
  image and parameter selection algorithm to the overall reconstruction accuracy with the state of the art methods in
  stereo and mono calibration and show that we can bring substantial improvements to the 3D reconstruction accuracy or
  mono calibration parameter estimation.

## Packages

- Python - version 3.8
- NumPy - version 1.19
- OpenCV - version 4.5

## Features

- Provided a set of calibration images of a planar chessboard pattern, the genetic algorithm finds the optimal subset of
  images which yield the best stereo/mono calibration.
- For stereo, the metric used is the reconstructed 3D distance between the top-left corner and all the other corners, in
  millimeters.
- For mono, the metric used it the average root-mean-square error (RMS), in pixels.

## Usage

First you need to print a calibration pattern. Please generate one using [this tool](https://calib.io), by choosing a
chessboard pattern and preferably setting the width odd and the height even (or viceversa). Keep note of the square size
you chose. We used a 9x8 chessboard with square size 7mm. Try to lay the pattern perfectly flat on a hard surface and
take a couple of still images of the board under varying poses in front of the camera(s), making sure all the chessboard
squares are visible. After acquiring around 30 images, you can either follow our naming convention: "left (ID).jpg"
and "right (ID).jpg", or alter the method "init()" in the "cal_wiz_main.py" script, where the loading takes place.
Create a directory named "in" inside the root of the project and place the images there.

Next, modify the "cal_wiz_constants.py" script to match your setup:

```python
TOTAL = 30 # Number of acquisitions;
CHESS = (9 - 1, 8 - 1) # Width and height of the chessboard;
SQ_MM = 7.00 # Square size in your preferred unit; 3D measurements will be in the same unit.
```

The lower bound for the number of active pairs (selected images) is 15. Please modify it as per your needs. Alter the "
cal_wiz_chromosome.py" script on the following line:

```python
MIN_GENES = 15
```

Finally, create an "out" folder in the project root and run the script "cal_wiz_main.py". The solution will be found in
the "sol.yml" file, containing the intrinsics and extrinsics for the cameras, along some statistics. During the
execution, the ETA for the process is displayed in the console.

Of particular interest are a couple of flags located in the "cal_wiz_constants.py" script:

```python
DEBUG = False # Enable/disable debugging information;
PARALLEL = True # Use the batch computation of chromosome fitnesses; will use all available CPU cores;
FLIP = False # Enable/disable horizontal flipping of calibration images;
FLAGS = False # Enable/disable the optional 8 calibration flags;
MONO = False # Perform mono calibration instead of stereo.
```

## Citation

If you found our work useful, please cite it using this BibTeX entry...

## Contact

Created by [@amarinescu](https://amarinescu.ro)
