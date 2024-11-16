from flask import Flask, render_template, request
import cv2 as cv
import time
import sys
import os

from werkzeug.utils import secure_filename

upload_folder = os.path.join('static', 'uploads')

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"  #tensorflow pre trained model

model_mean_values = (78.4263377603, 87.7689143744, 114.895847746)  #eliminate the effect of illunination changes-normalization
genderList = ['Male', 'Female']

genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

padding = 20
   
def getFaceBox(faceNet, frame, confidence_percent=0.7):
    img_frame = frame.copy()
    frameHeight = img_frame.shape[0]
    frameWidth = img_frame.shape[1]
    blob = cv.dnn.blobFromImage(img_frame, 1.0, (300, 300))  #mean subtraction, scaling performed

    faceNet.setInput(blob)  #Setting the image as input
    detections = faceNet.forward() #Runs forward pass to calculate output
    #print(detections)
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_percent:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(img_frame, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 5)
    return img_frame, bboxes

def gender_classifier(frame):
    t = time.time()
    frameFace, bboxes = getFaceBox(faceNet, frame, confidence_percent=0.7)
    for bbox in bboxes:
        #print(bbox)
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        #print(face)
        blob = cv.dnn.blobFromImage(face, 1.0, (250, 250), model_mean_values, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        #print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

        result = "{}".format(gender)
        cv.putText(frameFace, result, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
    return frameFace, result

# WSGI Application
app=Flask(__name__)
app.config['upload'] = upload_folder

@app.route('/', methods = ['GET','POST'])
def welcome():
    return render_template("home.html")

@app.route('/predict', methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['imagefile']
        if file.filename == "":
            return "No file selected"
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['upload'], filename))
            input1 = cv.imread(os.path.join(app.config['upload'], filename))
            output1, result = gender_classifier(input1)
            img = os.path.join(app.config['upload'], filename)
            return render_template('output.html', result = result, img = img)
        return render_template('output.html')

if __name__=='__main__':
    app.run(debug=True) #Restarts server automatically
