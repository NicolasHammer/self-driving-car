# Simulated GTA V Self-Driving Car

The objective of this work is to create a self-driving car in the video game GTA V through end-to-end computer vision.  This is done by training a convolutional neural network using screen data while driving.  During testing -- when the car is actually driven -- a YOLOv3 model is used to detect objects in the view of the car and adjust the car's speed, accordingly.

This work is based on the code in the following repositories:
1. [Alzaib Karovalia - Autonomous-Self-Driving-Car-GTA-5](https://github.com/Alzaib/Autonomous-Self-Driving-Car-GTA-5)
2. [Sentdex - pygta5](https://github.com/Sentdex/pygta5)
