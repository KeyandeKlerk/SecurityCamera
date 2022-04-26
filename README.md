# Keyan's Smart Security Camera

## Description

A python program designed to access the devices camera and run facial recognition and gender classification on the video feed. 

## What I Learnt
- how to use tkinter to make a GUI
- how to use open-cv to do image manipulation, and use the integrated facial recognition. 
- how to incorporate DNN gender classification.
- how to manipulate files and their locations

## How to Install

Install the requirements.txt file by running

> pip install -r requirements.txt

Run main.py to run the program directly or run menu.py to run the GUI alternative

## Features

### Main.py

- Access Webcam 
- Train models 
- Gender classification on training images
- Facial recognition on every second frame of video feed
- If face detected that is not in training models then saves frame to detections folder

### Menu.py

- GUI interface 
- Upload images
- Find faces (redirects to main.py)
- Exit program

## TO DO Soon

:heavy_check_mark: Only insert face into detections folder if not already there
:white_large_square: Gender classification on detection folder images
:white_large_square: Age classification on detection folder
:white_large_square: Allow more training images /train/{name}/{number}.jpg

## TO DO Future

- ALPR