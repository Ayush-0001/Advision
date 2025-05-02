import cv2
import numpy as np
import pickle
from flask import Flask, render_template, Response
from collections import deque

# Load gender model (SVM + PCA)
with open("model_svm.pickle", "rb") as f:
    model_svm = pickle.load(f)

with open("pca_dict.pickle", "rb") as f:
    pca_data = pickle.load(f)

# Auto-detect correct mean key
if 'mean_face' in pca_data:
    mean_face_arr = pca_data['mean_face']
elif 'mean_face_arr' in pca_data:
    mean_face_arr = pca_data['mean_face_arr']
else:
    raise KeyError("Mean face key not found in pca_dict.pickle")

model_pca = pca_data['pca']


# Load age model (DNN)
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
AGE_BUCKETS = ['(0-2)', '(4-6)', '(8-13)', '(15-20)', 
               '(20-30)', '(35-45)', '(48-55)', '(60-100)']

# Load face detector
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Ad mapping
advertisement_mapping = {
    # Male
    ('male', '(0-2)'): 'static/ads/male_0_2.jpg',
    ('male', '(4-6)'): 'static/ads/male_4_6.jpg',
    ('male', '(8-13)'): 'static/ads/male_8_13.jpg',
    ('male', '(15-20)'): 'static/ads/male_15_20.jpg',
    ('male', '(20-30)'): 'static/ads/male_20_30.jpg',
    ('male', '(35-45)'): 'static/ads/male_35_45.jpg',
    ('male', '(48-55)'): 'static/ads/male_48_55.jpg',
    ('male', '(60-100)'): 'static/ads/male_60_100.jpg',

    # Female
    ('female', '(0-2)'): 'static/ads/female_0_2.jpg',
    ('female', '(4-6)'): 'static/ads/female_4_6.jpg',
    ('female', '(8-13)'): 'static/ads/female_8_13.jpg',
    ('female', '(15-20)'): 'static/ads/female_15_20.jpg',
    ('female', '(20-30)'): 'static/ads/female_20_30.jpg',
    ('female', '(35-45)'): 'static/ads/female_35_45.jpg',
    ('female', '(48-55)'): 'static/ads/female_48_55.jpg',
    ('female', '(60-100)'): 'static/ads/female_60_100.jpg',

    # Fallback
    ('unknown', 'unknown'): 'static/ads/default.jpg'
}

# Helpers
def get_advertisement(gender, age):
    return advertisement_mapping.get((gender, age), advertisement_mapping[('unknown', 'unknown')])

def predict_gender(face_gray):
    try:
        face_resized = cv2.resize(face_gray, (100, 100)).astype('float32')
        face_normalized = face_resized / 255.0
        face_flattened = face_normalized.reshape(1, -1)
        face_centered = face_flattened - mean_face_arr
        face_pca = model_pca.transform(face_centered)

        # Get prediction probabilities if available
        if hasattr(model_svm, "predict_proba"):
            probs = model_svm.predict_proba(face_pca)[0]
            max_idx = np.argmax(probs)
            prediction = model_svm.classes_[max_idx]
            confidence = float(probs[max_idx])
        else:
            prediction = model_svm.predict(face_pca)[0]
            confidence = 1.0  # No probability support

        if prediction in ['male', 'female']:
            gender_history.append(prediction)
        else:
            gender_history.append('unknown')

        return prediction, confidence

    except Exception as e:
        print("Gender prediction error:", e)
        return 'unknown', 0.0


def predict_age(face_rgb):
    blob = cv2.dnn.blobFromImage(face_rgb, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    i = age_preds[0].argmax()
    age = AGE_BUCKETS[i]
    return age

# Flask app
app = Flask(__name__)
current_ad_path = advertisement_mapping[('unknown', 'unknown')]
current_prediction = "Detecting..."

box_history = deque(maxlen=5)
gender_history = deque(maxlen=10)

def generate_frames():
    global current_ad_path, current_prediction
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            box_history.append((x, y, w, h))
            avg_x = int(np.mean([b[0] for b in box_history]))
            avg_y = int(np.mean([b[1] for b in box_history]))
            avg_w = int(np.mean([b[2] for b in box_history]))
            avg_h = int(np.mean([b[3] for b in box_history]))

            face_gray = gray[avg_y:avg_y+avg_h, avg_x:avg_x+avg_w]
            face_rgb = cv2.resize(frame[avg_y:avg_y+avg_h, avg_x:avg_x+avg_w], (227, 227))

            raw_gender = predict_gender(face_gray)
            gender_history.append(raw_gender)
            if len(gender_history) > 0:
                gender = max(set(gender_history), key=gender_history.count)
            else:
                gender = 'unknown'

            age = predict_age(face_rgb)

            label = f"{gender}, Age: {age}"
            current_prediction = label
            current_ad_path = get_advertisement(gender, age)

            if gender == "male":
                color = (255, 0, 0)
            elif gender == "female":
                color = (203, 192, 255)
            else:
                color = (200, 200, 200)

            cv2.rectangle(frame, (avg_x, avg_y), (avg_x+avg_w, avg_y+avg_h), color, 2)
            cv2.putText(frame, label, (avg_x, avg_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ad_path')
def ad_path():
    return current_ad_path

@app.route('/prediction')
def prediction():
    global current_prediction
    return current_prediction
 
if __name__ == '__main__':
    app.run(debug=False)
