import cv2
from flask import Flask, render_template, Response, jsonify
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

app = Flask(__name__)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

classifier = Classifier("Image_classifier_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "Hello", "I", "I Love You",
    "J", "K", "L", "M", "N", "No", "O", "P", "Q", "R", "S","Space", "T", "U",
    "V", "W", "X", "Y", "Yes", "Z",
]

predicted_label = ""
all_predicted_labels = []  # Store all predicted labels

def generate_frames():
    global predicted_label, all_predicted_labels
    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

            imgCropShape = imgCrop.shape
            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap : wCal + wGap] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    predicted_label = labels[index]
                    if(predicted_label=="Space"):
                        predicted_label = "_ _"

                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap : hCal + hGap, :] = imgResize
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    predicted_label = labels[index]
                    if(predicted_label=="Space"):
                        predicted_label = "_ _"

                cv2.putText(
                    imgOutput,
                    labels[index],
                    (x, y - 25),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 255),
                    2,
                )
                print(labels[index])
                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset),
                    (x + w + offset, y + h + offset),
                    (255, 0, 255),
                    4,
                )

                # Append the new predicted label to the list
                all_predicted_labels.append(labels[index])

            except Exception as e:
                print(f"Error in resizing: {e}")

            _, buffer = cv2.imencode('.jpg', imgOutput)
            img_output_encoded = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_output_encoded + b'\r\n')

        cv2.waitKey(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_label')
def get_label():
    global predicted_label, all_predicted_labels
    return jsonify({'label': predicted_label, 'all_labels': all_predicted_labels})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
