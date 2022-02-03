from flask import Flask, render_template, url_for, Response, request
import tensorflow as tf
import requests
import cv2
import numpy as np
import imutils
from keras.preprocessing import image
from pygame import mixer
import webbrowser
import cv2
import numpy as np
import datetime, time
import os, sys
from scipy import stats

import player2

app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)
face_haar_cascade = cv2.CascadeClassifier('../Model/haarcascade_frontalface_default.xml')
model = tf.keras.models.load_model('../Model/model_csv.h5')
label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
global capture
capture=0
url = 'https://www.youtube.com/results?search_query='
chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe %s"
playlist = ''
    
def gen_frames():
    global capture
    count = 0
    final_pred = []
    while True:
        success, frame = camera.read()
        cap_img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_haar_cascade.detectMultiScale(cap_img_gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(255,255,255),2)
            count +=1
            roi_gray = cap_img_gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48,48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            predictions = model.predict(img_pixels)
            emotion_label = np.argmax(predictions)
            emotion_prediction = label_dict[emotion_label]
            cv2.putText(frame, emotion_prediction, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,200,0), 1 )
            final_pred.append(emotion_prediction)
            if count>=30:
                break
            test_1 = np.array(final_pred)
            mode = stats.mode(test_1)
        if success: 
            if(capture):
                capture=0
                if mode[0][0] == 'Neutral':
                    playlist = 'neutral+songs'
                    webbrowser.get(chrome_path).open(url+playlist)
                elif mode[0][0] == 'Happiness':
                    playlist = 'happy+songs'
                    webbrowser.get(chrome_path).open(url+playlist)
                elif mode[0][0] == 'Angry':
                    playlist = 'angry+songs'
                    webbrowser.get(chrome_path).open(url+playlist)
                elif mode[0][0] == 'Sad':
                    playlist = 'sad+songs'
                    webbrowser.get(chrome_path).open(url+playlist)
                elif mode[0][0] == 'Disgust':
                    playlist = 'disgust+songs'
                    webbrowser.get(chrome_path).open(url+playlist)
                elif mode[0][0] == 'Fear':
                    playlist = 'fear+songs'
                    webbrowser.get(chrome_path).open(url+playlist)
                else:
                    playlist = 'surprise+songs'
                    webbrowser.get(chrome_path).open(url+playlist)
            try:
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
        else:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selection')
def selection():
    return render_template('selection.html')

@app.route('/emoji')
def emoji():
    return render_template('emoji.html')

@app.route('/face_detection')
def face_detection():
    return render_template('face_detection.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
            
    elif request.method=='GET':
        return render_template('face_detection.html')
    return render_template('face_detection.html')

if __name__ == '__main__':
    app.run()

cv2.destroyAllWindows()
