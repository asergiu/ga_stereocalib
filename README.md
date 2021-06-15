# Optimal Stereo Camera Calibration via Genetic Algorithms
> Demonstration code accompanying our submission to the IJCAI 2021 AI4AD workshop.
> Link to the workshop landing page [_here_](https://www.ai4ad.net).

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Usage](#usage)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)


## General Information
- Abstract: Stereo camera real world measurement accuracy is strongly dependent on the camera calibration process, which, in turn, relies on the set of input images and camera model. There is a strong relationship between the quality of the input, the calibration process and the accuracy of the 3D reconstruction. The total set of calibration images has a strong influence on the calibration outcome and selecting the optimal subset of images and parameters for calibration estimation is a problem of combinatorial complexity. In this paper we propose a genetic algorithm calibration image selection process, driven by multiple metrics, that eliminates the image and parameter selection bias and avoids exploratory search over the entire solution space. We numerically compare the contribution of our image and parameter selection algorithm to the overall reconstruction accuracy with the state of the art methods in stereo and mono calibration and show that we can bring substantial improvements to the 3D reconstruction accuracy or mono calibration parameter estimation.


## Technologies Used
- Python 3 - version 3.8
- NumPy - version 1.19
- OpenCV - version 4.5


## Features
- Provided a set of calibration images of a planar chessboard pattern, the genetic algorithm finds the optimal subset of images which yield the best stereo/mono calibration.
- For stereo, the metric used is the reconstructed 3D distance between the top-left corner and all the other corners, in millimeters.
- For mono, the metric used it the average root-mean-square error (RMS), in pixels.


## Usage
First you need to print a calibration pattern. Please generate one using [this tool](https://calib.io), by choosing a chessboard pattern and preferably setting the width odd and the height even (or viceversa). Keep note of the square size you chose. We used a 9x8 chessboard with square size 7mm. Try to lay the pattern perfectly flat on a hard surface and take a couple of still images of the board under varying poses in front of the camera(s), making sure all the chessboard squares are visible. After acquiring around 30 images, you can either follow our naming convention: "left (ID).jpg" and "right (ID).jpg", or alter the method "init()" in the "cal_wiz_main.py" script, where the loading takes place. Create a directory named "in" inside the root of the project and place the images there.

Next, modify the "cal_wiz_constants.py" script to match your setup:

```python
TOTAL = 30 # Number of acquisitions;
CHESS = (9 - 1, 8 - 1) # Width and height of the chessboard;
SQ_MM = 7.00 # Square size in your preferred unit; 3D measurements will be in the same unit.
```


## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


## Contact
Created by [@amarinescu](https://amarinescu.ro)
