# Keyan's Smart Security Camera

## Description

A python program designed to access the devices camera and run facial recognition and gender classification on the video feed. 

- learnt how to use tkinter to make a GUI
- learnt how to use open-cv to do image manipulation, and use the integrated facial recognition. 
- learnt how to incorporate DNN gender classification.
- learnt how to manipulate files and their locations

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

- Gender classification on detection folder images
- Save image on unique face
- Age classification on detection folder
- Allow more training images /train/{name}/{number}.jpg

## TO DO Future

- ALPR