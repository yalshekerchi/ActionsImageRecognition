from __future__ import print_function

from flask import Flask, request, make_response
import json
import os
import sys
import shelve
import cv2
import numpy as np

# config strings
CAPTURE_DEVICE_ID = 0
FLASK_SERVER_HOST = '0.0.0.0'
FLASK_SERVER_PORT = 3000
MODEL_DIR = 'model'
DATASET_DIR = 'dataset'
SHELF_PATH = MODEL_DIR + '/user_shelf.db'
MODEL_PATH = MODEL_DIR + '/model.yml'
CASCADE_CLASSIFIER_PATH = 'libs/haarcascade_frontalface_default.xml'
MAX_CONFIDENCE = 60
DEFAULT_NUM_SAMPLES_TRAIN = 30
DEFAULT_NUM_SAMPLES_IDENTIFY = 50

# global objects
face_cascade = cv2.CascadeClassifier(CASCADE_CLASSIFIER_PATH)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# Initialize Flask flask_app instance
flask_app = Flask(__name__)

# start video capture
print("Starting VideoCapture object on Device %d" % CAPTURE_DEVICE_ID)
cap = cv2.VideoCapture(CAPTURE_DEVICE_ID)
if cap.isOpened() is False:
    print("Unable to initialize VideoCapture object. Exiting...")
    sys.exit()


@flask_app.route('/recognition_service', methods=['POST'])
def webhook_handler():
    # Store JSON Request
    req = request.get_json(silent=True, force=True)

    # Dump JSON Request to stdout
    print("Request:")
    print(json.dumps(req, indent=4))

    # Extract parameters and action from request body
    parameters = req.get("result").get("parameters")
    action = req.get("result").get("action")

    # Generate response body
    if action == "add_user" or action == "update_user":
        res = generate_add_user_response(parameters)
    elif action == "identify_user":
        res = generate_identify_user_response()
    else:
        res = {}

    res_string = json.dumps(res, indent=4)

    # Make dictionary into response object and add headers
    r = make_response(res_string)
    r.headers['Content-Type'] = 'application/json'

    # Return response
    return r


def get_max_id(obj):
    if len(obj.keys()) == 0:
        return 0    
    return max([int(key) for key in obj.keys()])


def update_db(value, key=None):
    # check existing entry
    for k, v in users.iteritems():
        if v == value:
            return k

    # insert new entry
    if key is None:
        key = str(get_max_id(users) + 1)
    users[key] = value
    return key


def generate_dataset(capture, dataset_path, user_name, num_samples=DEFAULT_NUM_SAMPLES_TRAIN, wait_time=500,
                     display_window=True):
    sample_idx = 0
    samples_captured = 0

    # Create lists to hold all subject faces and labels
    face_list = []
    label_list = []

    # Get user id if in db, else add user to db
    user_id = update_db(user_name)
    user_dataset_path = dataset_path + "/user_" + user_id

    # Create user directory if non-existent
    if not os.path.exists(user_dataset_path):
        os.makedirs(user_dataset_path)
        with open(user_dataset_path + '/user_name.txt', 'w') as f:
            f.write(user_name)

    while True:
        # Obtain next video frame
        retval, img = capture.read()

        # Convert frame into gray image space and detect faces
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        # Iterate through faces found
        for (x, y, w, h) in detected_faces:
            samples_captured += 1
            sample_idx += 1

            # Resize image to 200x200
            resized_img = cv2.resize(img_gray[y:y + h, x:x + w], (200, 200), interpolation=cv2.INTER_AREA)

            # Save captured face image in the user dataset directory, and append to lists
            while os.path.exists(user_dataset_path + "/sample_" + str(sample_idx) + ".jpg"):
                sample_idx += 1
            cv2.imwrite(user_dataset_path + "/sample_" + str(sample_idx) + ".jpg", resized_img)
            face_list.append(resized_img)
            label_list.append(int(user_id))

            # Display captured frame
            if display_window:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow('frame', img)

        # Wait for 500 milliseconds
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
        elif samples_captured >= num_samples:
            break

    cv2.destroyAllWindows()
    return face_list, label_list


def get_faces(dataset_path):
    # Create lists to hold all subject faces and labels
    face_list = []
    label_list = []

    # Iterate through all directories (users) in dataset path
    for user_dataset_name in os.listdir(dataset_path):
        # Ignore irrelevant directories
        if not user_dataset_name.startswith("user_"):
            continue

        # Extract user id and name from directory
        user_dataset_path = dataset_path + "/" + user_dataset_name
        user_id = int(user_dataset_name.replace("user_", ""))
        with open(user_dataset_path + '/user_name.txt', 'r') as f:
            user_name = f.read()

        # Add user to db
        update_db(user_name, str(user_id))

        # Generate list of image paths
        image_paths = [os.path.join(user_dataset_path, f) for f in os.listdir(user_dataset_path)]

        # Iterate through images
        for image_path in image_paths:
            # Skip non-jpg files
            if not image_path.endswith('.jpg'):
                continue

            # Read Image and append to lists
            img = cv2.imread(image_path)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_list.append(img_gray)
            label_list.append(user_id)

    return face_list, label_list


def predict_user(capture, num_samples=DEFAULT_NUM_SAMPLES_IDENTIFY, display_window=True):
    samples_captured = 0

    # Use list of length users to count number of detections for each user
    results = [0]*len(users)

    while True:
        # Obtain next video frame
        retval, img = capture.read()

        # Convert frame into gray image space and detect faces
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.3, minNeighbors=5)

        # Iterate through faces found
        for (x, y, w, h) in detected_faces:
            # Predict user_id, set to Unknown if confidence > MAX_CONFIDENCE
            user_id, confidence = face_recognizer.predict(img_gray[y:y + h, x:x + w])
            if confidence > MAX_CONFIDENCE:
                user_id = 0

            user_name = users[str(user_id)]
            results[user_id] += 1
            samples_captured += 1

            # Display captured frame
            if display_window:
                cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
                cv2.putText(img, user_name, (x, y + h), cv2.FONT_HERSHEY_DUPLEX, 1.5, (225, 0, 0))
                cv2.imshow('frame', img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        elif samples_captured >= num_samples:
            break

    cv2.destroyAllWindows()

    # Returns user_name with most detections, if equal, user with index is returned
    return users[str(results.index(max(results)))]


def generate_add_user_response(parameters):
    given_name = parameters.get("given-name")
    num_samples = parameters.get("num-samples")

    if num_samples == "":
        num_samples = DEFAULT_NUM_SAMPLES_IDENTIFY
    else:
        num_samples = int(num_samples)

    face_list, label_list = generate_dataset(cap, DATASET_DIR, given_name, num_samples)
    face_recognizer.update(face_list, np.array(label_list))
    face_recognizer.write(MODEL_PATH)
    users.sync()

    response_string = given_name + " has been successfully added."
    res = {
        "speech": response_string,
        "displayText": response_string,
    }
    return res


def generate_identify_user_response():
    identify_result = predict_user(cap)

    if identify_result == "Unknown":
        response_string = "Unknown person is at the door."
    else:
        response_string = identify_result + " is at the door."

    res = {
        "speech": response_string,
        "displayText": response_string,
    }
    return res

if __name__ == '__main__':
    # Create directories
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # get a db handle
    users = shelve.open(SHELF_PATH)

    try:
        # Check if model exits
        if os.path.exists(MODEL_PATH):
            face_recognizer.read(MODEL_PATH)
        else:
            # Reset db
            users.clear()
            users['0'] = 'Unknown'

            # Read existing images from dataset directory
            faces, labels = get_faces(DATASET_DIR)
            if len(faces) == 0:
                print("No faces found in dataset folder.")

            # Train model
            face_recognizer.train(faces, np.array(labels))

        # Prompt user to add new face to dataset
        new_face_input = raw_input("Would you like to add a new face to the dataset (y/n): ")
        if new_face_input.lower() == "y":
            name = raw_input("Enter user name: ")
            faces, labels = generate_dataset(cap, DATASET_DIR, name)
            face_recognizer.update(faces, np.array(labels))

        # Prompt user to add new face to dataset
        train_input = raw_input("Would you like to rescan the dataset folder? This will erase current model (y/n): ")
        if train_input.lower() == "y":
            # Reset db
            users.clear()
            users['0'] = 'Unknown'

            faces, labels = get_faces(DATASET_DIR)
            face_recognizer.train(faces, np.array(labels))

        # Write model to disk, sync db
        face_recognizer.write(MODEL_PATH)
        users.sync()

        # Start app
        print("Starting facial recognition service on " + FLASK_SERVER_HOST + ":" + str(FLASK_SERVER_PORT))
        flask_app.run(debug=False, port=FLASK_SERVER_PORT, host=FLASK_SERVER_HOST)

    finally:
        users.close()
