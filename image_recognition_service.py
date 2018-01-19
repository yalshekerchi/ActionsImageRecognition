from __future__ import print_function

from flask import Flask, request, make_response
import json
import os
import sys
import shelve
import cv2
import numpy as np

# Initialize Flask app instance
app = Flask(__name__)

# config strings
shelfPath = 'model/user_shelf.db'
cascadeClassifierPath = 'libs/haarcascade_frontalface_default.xml'

# global objects
users = {} # user_shelf
face_cascade = cv2.CascadeClassifier(cascadeClassifierPath)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

####
# Initialize global variables
subjects = ["Unknown", "Stanley", "Justin", "Yasir", "Jo", "Nisarg"]
#cascadeClassifierPath = "C:/opencv/build/etc/lbpcascades/lbpcascade_frontalface_improved.xml"
#cascadeClassifierPath = "C:/dev/opencv/build/etc/haarcascades/haarcascade_frontalface_default.xml"
####

@app.route('/train', methods=['POST'])
def train_model():
    # Store JSON Request
    req = request.get_json(silent=True, force=True)

    # Dump JSON Request to stdout
    print("Request:")
    print(json.dumps(req, indent=4))

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

def get_max_id(dict):
    return max([int(x) for x in dict.vals()])

def update_db(value):
    # check existing entry
    for key, val in users.iteritems():
        if val == value:
            return key
    # insert new entry
    user_id = str(get_max_id(users) + 1)
    users[user_id] = value
    return user_id

def generate_dataset(capture, path, name, num_samples = 30):
    sample_idx = 0
    samples_captured = 0

    face_list = []
    label_list = []

    user_id = update_db(name)

    dir_path = path + "/user_" + user_id
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
            resized_img = cv2.resize(img_gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_AREA)
            face_list.append(resized_img)
            label_list.append(int(user_id))

            cv2.imwrite(dir_path + "/sample_" + str(sample_idx) + ".jpg", resized_img)

            samples_captured += 1
            sample_idx += 1

            # Display captured frame
            cv2.imshow('frame', img)

        # Wait for 1 second
        if cv2.waitKey(500) & 0xFF == ord('q'):
            break
        elif samples_captured >= num_samples:
            break

    return face_list, label_list


def get_faces(path):
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


def predict_user(capture, num_samples = 50):
    samples_captured = 0
    results = [0]*len(users)

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

            name = users[str(label)]
            results[label] += 1
            samples_captured += 1

            cv2.putText(img, name, (x, y + h), cv2.FONT_HERSHEY_PLAIN, 1.5, (225, 0, 0))

        cv2.imshow('frame', img)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif samples_captured >= num_samples:
            break

    capture.release()
    cv2.destroyAllWindows()

    return users[str(results.index(max(results)))]

if __name__ == '__main__':
    device_id = 0
    port = 3000

    try:
        users = shelve.open(shelfPath)
        face_recognizer.read("model/model.yml")
        
        print("Starting VideoCapture object on Device %d" % device_id)
        cap = cv2.VideoCapture(0)
        if cap.isOpened() is False:
            print("Unable to initialize VideoCapture object")

        new_face_input = raw_input("Would you like to add a new face to the dataset (y/n): ")
        if new_face_input.lower() == "y":
            user_name = raw_input("Enter user name: ")
            faces, labels = generate_dataset(cap, "dataSet", user_name, 50)
            face_recognizer.update(faces, labels)

        train_input = raw_input("Would you like to retrain the model (y/n): ")
        if train_input.lower() == "y":
            faces, labels = get_faces("dataSet")
            face_recognizer.train(faces, np.array(labels))

            if not os.path.exists("model"):
                os.makedirs("model")
            face_recognizer.write("model/model.yml")

        print(predict_user(cap))
    finally:
        users.close()
