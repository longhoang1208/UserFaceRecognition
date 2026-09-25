# [PYTHON] User Face Recognition
This is my source code for User Face Recognition project, a python program that scan user face and can be able to recognize whether the current user is labeled `admin` or not.

## How it works
It uses `opencv` to extract camera frame and `mediapipe` face detection solution to detect faces. When user press space key, it gets their face ROI in frame. Then, it collects ROI data from many frames and uses `numpy` to get the mean matrix among the ROI matrixes to get a sample matrix, the scanned usre face will be labeled `admin`. Finally, the program will calculate the mean difference between the current real-time ROI and the sample to define whether the current user is `admin` or `unknown`.

## Requirements
### What I've used in this project
- python 3.10
- mediapipe == 0.10.21
- opencv-python
- numpy


### Installation
```bash
pip install mediapipe==0.10.21
```
