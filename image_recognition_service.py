from __future__ import print_function

from flask import Flask, request, make_response
import json
import os
import sys

import cv2
import numpy as np


# Initialize Flask app instance
app = Flask(__name__)

# Initialize global variables
subjects = ["Unknown", "Stanley", "Justin", "Yasir", "Jo", "Nisarg"]
#cascadeClassifierPath = "C:/opencv/build/etc/lbpcascades/lbpcascade_frontalface_improved.xml"
cascadeClassifierPath = "C:/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(cascadeClassifierPath)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

@app.route('/train', methods=['POST'])
def train_model():
    # Store JSON Request
    req = request.get_json(silent=True, force=True)

    # Dump JSON Request to stdout
    print("Request:")
    print(json.dumps(req, indent=4))

    # Start Capture
    #start_capture_stuff()

    # Create Response Dictionary
    voice = "Hello World Speech"
    res = {
        "speech": voice,
        "displayText": voice,
        "source": "test"
    }

    # Make dictionary into response object and add headers
    r = make_response(res)
    r.headers['Content-Type'] = 'application/json'

    # Return response
    return r


def generate_dataset(capture, path, label, num_samples = 30):
    sample_idx = 0
    samples_captured = 0

    dir_path = path + "/user_" + label
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    while True:
        # Obtain next video frame
        retval, img = capture.read()

        # Convert frame into gray image space and detect faces
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        # Iterate through faces found
        for (x, y, w, h) in detected_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            while os.path.exists(dir_path + "/sample_" + str(sample_idx) + ".jpg"):
                sample_idx += 1

            # Save captured face image in the dataset folder
            resized = cv2.resize(img_gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dir_path + "/sample_" + str(sample_idx) + ".jpg", resized)
            samples_captured += 1

            # Display captured frame
            cv2.imshow('frame', img)

        # Wait for 1 second
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
        elif samples_captured >= num_samples:
            break


def getFaces(path):
    # Create lists to hold all subject faces and labels
    face_list = []
    label_list = []

    # Iterate through all directories (users) in dataSet folder
    for dir_name in os.listdir(path):
        # Ignore irrelevant directories
        if not dir_name.startswith("user_"):
            continue

        # Extract user id from directory name
        label = int(dir_name.replace("user_", ""))

        # Generate list of image paths
        dir_path = path + "/" + dir_name
        image_paths = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]

        # Iterate through images
        for image_path in image_paths:
            # Read Image
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_list.append(img_gray)
            label_list.append(label)

    return face_list, label_list


def predict_user(capture, num_samples = 300):
    samples_captured = 0
    results = {label : 0 for label in subjects}

    while True:
        # Obtain next video frame
        retval, img = capture.read()

        # Convert frame into gray image space and detect faces
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in detected_faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            label, confidence = face_recognizer.predict(img_gray[y:y + h, x:x + w])

            if confidence > 60:
                label = 0

            label_text = subjects[label]
            results[subjects[label]] += 1
            samples_captured += 1

            cv2.putText(img, label_text, (x, y + h), cv2.FONT_HERSHEY_PLAIN, 1.5, (225, 0, 0))

        cv2.imshow('frame', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif samples_captured >= num_samples:
            break

    capture.release()
    cv2.destroyAllWindows()

    return [k for k,v in results.items() if v == max(results.values())][0]

if __name__ == '__main__':
    device_id = 0
    port = 3000

    print("Starting VideoCapture object on Device %d" % device_id)
    cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        print("Unable to initialize VideoCapture object")

    new_face_input = raw_input("Would you like to add a new face to the dataset (y/n): ")
    if new_face_input.lower() == "y":
        user_id = raw_input("Enter User ID: ")
        generate_dataset(cap, "dataSet", user_id, 50)

    train_input = raw_input("Would you like to retrain the model (y/n): ")
    if train_input.lower() == "y":
        faces, labels = getFaces("dataSet")
        face_recognizer.train(faces, np.array(labels))

        if not os.path.exists("model"):
            os.makedirs("model")
        face_recognizer.write("model/model.yml")

    face_recognizer.read("model/model.yml")
    print(predict_user(cap))
    #print("Starting app on port %d" % port)
    #app.run(debug=False, port=port, host='0.0.0.0')