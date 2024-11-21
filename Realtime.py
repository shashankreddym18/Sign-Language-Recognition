import cv2
from flask import Flask, render_template, Response, jsonify, request
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time

app = Flask(__name__)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Image_classifier_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = [
    "A", "B", "C", "D", "E", "F", "G", "H", "Hello", "I", "I Love You",
    "J", "K", "L", "M", "N", "No", "O", "P", "Q", "R", "S", " ", "T", "U",
    "V", "W", "X", "Y", "Yes", "Z",
]

predicted_label = ""
all_predicted_labels = []  # Store all predicted labels
last_prediction_time = 0  # For delay between predictions
last_update_time = 0  # To control text update delay


def generate_frames():
    global predicted_label, all_predicted_labels, last_prediction_time, last_update_time
    prediction_delay = 0.75  # Delay in seconds for registering predictions
    text_update_delay = 2.5  # Delay in seconds between updating displayed text

    while True:
        success, img = cap.read()
        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand["bbox"]

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset: y + h + offset, x - offset: x + w + offset]

            aspectRatio = h / w

            try:
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize

                # Predict only after a delay
                current_time = time.time()
                if current_time - last_prediction_time > prediction_delay:
                    prediction, index = classifier.getPrediction(imgWhite, draw=False)
                    predicted_label = labels[index]

                    if predicted_label != "Space":  # Exclude spaces from prediction delay
                        last_prediction_time = current_time

                # Update displayed text only after a delay
                if current_time - last_update_time > text_update_delay:
                    all_predicted_labels.append(predicted_label)  # Allow duplicates
                    last_update_time = current_time

                cv2.putText(
                    imgOutput,
                    predicted_label,
                    (x, y - 25),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (255, 0, 255),
                    2,
                )
                cv2.rectangle(
                    imgOutput,
                    (x - offset, y - offset),
                    (x + w + offset, y + h + offset),
                    (255, 0, 255),
                    4,
                )

            except Exception as e:
                print(f"Error in resizing: {e}")

            _, buffer = cv2.imencode('.jpg', imgOutput)
            img_output_encoded = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img_output_encoded + b'\r\n')

        cv2.waitKey(1)


@app.route('/add_new_line', methods=['POST'])
def add_new_line():
    """Add a new line to the predicted text when the Enter key is pressed."""
    global all_predicted_labels
    all_predicted_labels.append("\n")  # Append a newline marker
    return jsonify({'success': True, 'all_labels': ''.join(all_predicted_labels).replace("\n", "<br>")})


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the prediction history."""
    global all_predicted_labels
    all_predicted_labels = []  # Clear the history
    return jsonify({'success': True, 'all_labels': ''.join(all_predicted_labels).replace("\n", "<br>")})


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/communicate')
def communicate():
    return render_template('index.html')


@app.route('/learn')
def learn():
    return render_template('learn.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_label')
def get_label():
    global predicted_label, all_predicted_labels
    # Replace newline markers (\n) with <br> tags for HTML display
    return jsonify({'label': predicted_label, 'all_labels': ''.join(all_predicted_labels).replace("\n", "<br>")})


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
