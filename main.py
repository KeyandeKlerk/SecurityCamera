import cv2
import os
import sys
import numpy as np
import face_recognition as fr
from charset_normalizer import detect


def init():
    global cap, face_cascade, body_cascade
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    body_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_fullbody.xml")

    if not os.path.exists('detections'):
        os.mkdir('detections')


def train_images():
    path = "./train/"

    global known_name_encodings, known_names
    known_names = []
    known_name_encodings = []

    images = os.listdir(path)
    for _ in images:
        try:
            image = fr.load_image_file(path + _)
            image_path = path + _
            encoding = fr.face_encodings(image)[0]

            known_name_encodings.append(encoding)
            known_names.append(os.path.splitext(
                os.path.basename(image_path))[0].capitalize())
        except Exception as e:
            print(e)
            sys.exit(1)


def display_video():
    while(True):

        # Capture the video frame
        ret, frame = cap.read()

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Exit using 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        recognize_faces(frame)


def recognize_faces(frame):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_name_encodings, face_encoding)
        name = ""

        face_distances = fr.face_distance(
            known_name_encodings, face_encoding)
        best_match = np.argmin(face_distances)

        if matches[best_match]:
            name = known_names[best_match]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 15),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, 1.0, (255, 255, 255), 1)
        if name == "":
            cv2.imwrite("./detections/gary.jpg", frame)


if __name__ == "__main__":
    init()
    print("Init Complete")
    train_images()
    print("Training Complete")
    display_video()
