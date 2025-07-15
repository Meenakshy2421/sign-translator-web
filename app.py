from flask import Flask, render_template, Response
import cv2
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image
import numpy as np
# from main import text_to_speech

app = Flask(__name__)
camera = cv2.VideoCapture(0)

d = {0: ' ', 1: 'What is your name?', 2: 'Hello', 3: 'I need To Take A Break',
     4: 'Where are you?', 5: 'Excuse Me', 6: 'Have Fun', 7: 'Good morning',
     8: 'How are You?', 9: 'I Love You', 10: 'I am Sorry', 11: "I don't know",
     12: 'See you later', 13: 'You are welcome', 14: 'Nice to meet you', 15: 'I am fine',
     16: 'Best wishes', 17: 'Good Night', 18: 'Congrats', 19: 'I dont understand',
     20: 'I am not feeling well', 21: 'I hate you', 22: 'Stop Doing That', 23: 'Who are you?',
     24: "It's okay", 25: 'I have no idea', 26: 'Well done, Keep it up'}

upper_left = (335, 3)
bottom_right = (635, 603)

json_file = open('model-bw.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model-bw.h5")


def function(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return res


def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break
        else:
            frame = cv2.flip(frame, 1)
            r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
            rect_img = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
            sketcher_rect = function(rect_img)
            sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
            frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect_rgb

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


l = []


@app.route('/predict', methods=['POST'])
def predictions():
    success, frame = camera.read()
    frame = cv2.flip(frame, 1)
    r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
    rect_img = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    sketcher_rect = function(rect_img)
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect_rgb

    sketcher_rect = cv2.resize(sketcher_rect, (128, 128))
    x = image.img_to_array(sketcher_rect)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    pre = loaded_model.predict(x)
    p_test = np.argmax(pre)
    a = d[p_test]
    # l.append(a)
    # str1 = ""
    # for ele in l:
    #     str1 += ele
    return {'pred': a}


@app.route('/stop', methods=['POST'])
def stopping():
    success, frame = camera.read()

    if success:
        frame = cv2.flip(frame, 1)
        r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
        rect_img = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
        sketcher_rect = function(rect_img)
        sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
        frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]] = sketcher_rect_rgb
        str1 = ""
        for ele in l:
            str1 += ele
        text_to_speech(str1, 'Female')
        l.clear()
        return {'pred': str1}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)
