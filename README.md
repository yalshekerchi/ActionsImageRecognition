# Who's There?
A facial recognition service developed in *Python* leveraging *OpenCV* to be deployed onto a security camera and identify authorized visitors

 - Utilizes Dialogflow (API.AI) to process requests from Google Assistant enabled devices and interface with the facial recognition service backend
 - Used the Frontal Face Haar Cascade for facial detection to crop out the faces while training the model
 - Used OpenCV's Local Binary Patterns Histograms Algorithm for the Face Recognizer Class
 - Includes user registration feature that detects faces and generates a dataset from a video source, using it to train the LBPH facial recognition algorithm
 - Uses the Flask microframework web service to host the service
 
## Supported Intents
The following intents are supported from Dialogflow along with the parameters passed onto the web service
### Add User Intent
Resisters new user to the database and retrains model using new image dataset
 - given-name: @sys.given-name (Required Parameter)
 - num-samples: @sys.number-integer
 
### Update User Intent
Updates model using new image dataset
 - given-name: @sys.given-name (Required Parameter)
 - num-samples: @sys.number-integer
 
### Identify User Intent
Analyzes face from video feed to perform prediction on whether the user is authorized
