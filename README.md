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
How does one go about using it?
Provide various use cases and code examples here.

`write-your-code-here`


## Acknowledgements
Give credit here.
- This project was inspired by...
- This project was based on [this tutorial](https://www.example.com).
- Many thanks to...


## Contact
Created by [@amarinescu](https://amarinescu.ro)
