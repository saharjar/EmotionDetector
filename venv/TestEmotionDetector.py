import cv2
import os
import numpy as np
from keras.models import model_from_json

emotion_dict = {0:"Angry", 1:"Disgusted", 2:"Fearful",3:"Happy", 4:"Neutral", 5:"Sad", 6:"Surprised"}


script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, '..', 'models')
json_file_path = os.path.join(models_dir, 'model.json')

# Open the JSON file
json_file = open(json_file_path, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

h5_file_path = os.path.join(models_dir, 'model.h5')
model.load_weights(h5_file_path)
print("Loaded model from disk")

# start the webcam feed

cap = cv2.VideoCapture("C:\\Users\\User\\Downloads\\sampl2.mp4")

while True:
    # Drow bounding box around face
    ret, frame = cap.read()

    if not ret:
        break

    if not frame.size == 0:
        frame = cv2.resize(frame, (1280, 720))

        script_dir = os.path.dirname(os.path.abspath(__file__))
        haar_dir = os.path.join(script_dir, '..', 'haarcascade')
        haar_file_path = os.path.join(haar_dir, 'haarcascade_frontalface_default.xml')
        face_detector = cv2.CascadeClassifier(haar_file_path)
        gray_frame= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # Detect faces availabel on camera
        num_faces = face_detector.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

        # Take each face available on the camera and preprocess it
        for(x,y,w,h) in num_faces:
            cv2.rectangle(frame, (x,y-50),(x+w,y+h+10), (0,255,0),4)
            roi_gray_frame = gray_frame[y:y+h, x:x+w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48,48)),-1),0)

            #predict the emotions
            emotion_prediction = model.predict(cropped_img)
            maxindx = int(np.argmax(emotion_prediction))
            cv2.putText(frame,emotion_dict[maxindx],(x+5,y-20), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)

        cv2.imshow('Emotion Detection',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
