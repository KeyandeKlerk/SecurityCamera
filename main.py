import face_recognition
import cv2
import numpy as np
import os
from os import path
from datetime import datetime

# Init Function


def init():
    global gender_net, gender_list, MODEL_MEAN_VALUES, webcam_video
    global known_face_encodings, known_face_names, known_gender
    global face_encodings, face_locations, face_names, gender_names, process_this_frame

    # Loading the net from gender classification
    gender_net = cv2.dnn.readNetFromCaffe(
        './src/deploy_gender.prototxt', './src/gender_net.caffemodel')
    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

    # Labels of Gender
    gender_list = ['Male', 'Female']

    # Get a reference to webcam video
    webcam_video = cv2.VideoCapture(0)

    # Create arrays of known face encodings, their names and their genders
    known_face_encodings = []
    known_face_names = []
    known_gender = []

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    gender_names = []

    print("Init complete")


# Function to check if face is already in detections folder


def checkUnique(frame):

    counter = 0
    date = datetime.now()
    s = date.strftime("%c")
    formatted_date = s.replace(" ", "_")
    formatted_date = formatted_date.replace(":", "-")
    filename = "./detections/%s.jpg" % formatted_date

    # Get directory path
    folder_dir = "C:\\Users\\Keyan\\Programming Projects\\Python\\SecurityCamera\\detections"
    current_image = frame
    highest_likeness = 0

    for images in os.listdir(folder_dir):
        if (images.endswith(".jpg")):
            base = cv2.imread(folder_dir + "\\" + images)

            # Image manipulation
            hsv_base = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
            hsv_test = cv2.cvtColor(current_image, cv2.COLOR_BGR2HSV)

            h_bins = 50
            s_bins = 60
            histSize = [h_bins, s_bins]
            h_ranges = [0, 180]
            s_ranges = [0, 256]
            ranges = h_ranges + s_ranges
            channels = [0, 1]

            hist_base = cv2.calcHist(
                [hsv_base], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist_base, hist_base, alpha=0,
                          beta=1, norm_type=cv2.NORM_MINMAX)
            hist_test = cv2.calcHist(
                [hsv_test], channels, None, histSize, ranges, accumulate=False)
            cv2.normalize(hist_test, hist_test, alpha=0,
                          beta=1, norm_type=cv2.NORM_MINMAX)

            # Compare hist of the 2 images
            compare_method = cv2.HISTCMP_CORREL
            base_test = cv2.compareHist(
                hist_base, hist_test, compare_method)

            if base_test > highest_likeness:
                highest_likeness = base_test

        # Only write image to detections folder if likeness to other images is less that 90%
    if highest_likeness < 0.85:

        # Create formatted filename for predictGender
        formatted_filename = filename[13:]
        formatted_filename = "".join(
            (folder_dir, "\\", formatted_filename))

        cv2.imwrite(
            filename, frame)

        predicted_gender = predictGender(formatted_filename)

        new_filename = "".join(
            (filename[:-4], "_", predicted_gender))

        # Checks if image exists and if so adds counter to file name
        if path.exists(new_filename + ".jpg"):
            os.rename(formatted_filename, "".join(
                (new_filename, "_", str(counter), ".jpg")))
            counter += 1
        else:
            os.rename(formatted_filename, "".join(
                (new_filename, ".jpg")))


# Function to predict the gender


def predictGender(image_path):
    face_img = cv2.imread(image_path)
    blob = cv2.dnn.blobFromImage(
        face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = gender_list[gender_preds[0].argmax()]
    return gender

# Function to predict age from image


def trainModel(training_folder):

    dirs = os.listdir(training_folder)
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue
        _label = int(dir_name.replace("s", ""))
        persons_path = training_folder + "/" + dir_name
        persons = os.listdir(persons_path)

        # Manipulates each image in the folder of persons
        for person in persons:
            image_path = persons_path + '/' + person
            filename = person[:-4]

            person_image = face_recognition.load_image_file(image_path)
            person_face_encoding = face_recognition.face_encodings(person_image)[
                0]
            person_name = filename
            person_gender = predictGender(image_path)

            known_face_encodings.append(person_face_encoding)
            known_face_names.append(person_name)
            known_gender.append(person_gender)

    print("Model training complete")

# Starting point of program


def main():
    init()
    trainModel(
        "C:\\Users\\Keyan\\Programming Projects\\Python\\SecurityCamera\\train")

    process_this_frame = True
    while True:
        # Grab a single frame of video
        ret, frame = webcam_video.read()
        font = cv2.FONT_HERSHEY_DUPLEX

        cv2.putText(frame, "Press Q to quit.", (0, 30),
                    font, 0.8, (255, 255, 255), 1)

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)

            face_names = []
            gender_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                global name, gender
                name = ""
                gender = ""

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]
                #     gender = known_gender[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    gender = known_gender[best_match_index]

                face_names.append(name)
                gender_names.append(gender)

                # If face is found that is not in training set
                if name == "":
                    checkUnique(frame)

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            if name != "":
                # Draw a label with a name and gender below the face
                cv2.rectangle(frame, (left, bottom),
                              (right, bottom + 56), (255, 0, 0), cv2.FILLED)

                cv2.putText(frame, name, (left + 6, bottom + 24),
                            font, 0.8, (255, 255, 255), 1)

                cv2.putText(frame, gender, (left + 6, bottom + 52),
                            font, 0.8, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Face Recognition', frame)

        # Hit 'q' on the keyboard to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    webcam_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

else:
    print("Running by module")
