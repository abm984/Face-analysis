import cv2
import dlib
import numpy as np
import imutils
from math import hypot
import csv
from datetime import datetime
import asyncio
import websockets
import json
import tensorflow as tf
import os 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = tf.keras.models.load_model('emotion_detection_model_100epochs.h5')
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']


font = cv2.FONT_HERSHEY_PLAIN


def midpoint(p1, p2):
    return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)


def get_face_direction(shape):
    left_eye = shape[36:42]
    right_eye = shape[42:48]
    
    nose = shape[27]
    chin = shape[8]
    
    left_midpoint = midpoint(left_eye[0], left_eye[3])
    right_midpoint = midpoint(right_eye[0], right_eye[3])
    
    eye_midpoint = midpoint(left_midpoint, right_midpoint)
    
    direction_vector = np.array([eye_midpoint[0] - nose[0], eye_midpoint[1] - nose[1]])
    
    return direction_vector


def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = facial_landmarks[eye_points[0]]
    right_point = facial_landmarks[eye_points[3]]
    center_top = midpoint(facial_landmarks[eye_points[1]], facial_landmarks[eye_points[2]])
    center_bottom = midpoint(facial_landmarks[eye_points[4]], facial_landmarks[eye_points[5]])
    
    hor_line_length = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_length = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    
    ratio = hor_line_length / ver_line_length
    return ratio


def get_gaze_ratio(eye_points, facial_landmarks, gray):
    left_eye_region = np.array([facial_landmarks[point] for point in eye_points], np.int32)
    
    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    
    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    
    if gray_eye.size == 0:
        return None

    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    
    if threshold_eye is None or threshold_eye.size == 0:
        return None

    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)
    
    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white
    
    return gaze_ratio


def get_vertical_gaze_ratio(eye_points, facial_landmarks, gray):
    eye_region = np.array([facial_landmarks[point] for point in eye_points], np.int32)
    
    height, width = gray.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [eye_region], True, 255, 2)
    cv2.fillPoly(mask, [eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)
    
    min_x = np.min(eye_region[:, 0])
    max_x = np.max(eye_region[:, 0])
    min_y = np.min(eye_region[:, 1])
    max_y = np.max(eye_region[:, 1])
    gray_eye = eye[min_y: max_y, min_x: max_x]
    
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    
    height, width = threshold_eye.shape
    upper_part = threshold_eye[0:int(height / 2), 0:width]
    lower_part = threshold_eye[int(height / 2):height, 0:width]
    
    upper_white = cv2.countNonZero(upper_part)
    lower_white = cv2.countNonZero(lower_part)
    
    if upper_white == 0:
        vertical_gaze_ratio = 1
    elif lower_white == 0:
        vertical_gaze_ratio = 5
    else:
        vertical_gaze_ratio = upper_white / lower_white
    
    return vertical_gaze_ratio


with open('gaze_data.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    # Header including landmark points
    header = ["Timestamp", "Horizontal Gaze", "Vertical Gaze", "Direction", "Emotion"] + [f"Landmark_{i}_x" for i in range(68)] + [f"Landmark_{i}_y" for i in range(68)]
    writer.writerow(header)


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    rects = detector(gray, 0)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    gaze_data = {}
    emotion_label = ""
    direction_text = "Not detected"  # Initialize direction_text

    for rect in rects:
        shape = predictor(gray, rect)
        shape = np.array([(point.x, point.y) for point in shape.parts()])

        
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        
        def draw_mesh(shape, points, is_closed=True):
            cv2.polylines(frame, [shape[points]], is_closed, (0, 255, 0), 1, cv2.LINE_AA)

        
        jaw = list(range(0, 17))
        left_eyebrow = list(range(17, 22))
        right_eyebrow = list(range(22, 27))
        nose_bridge = list(range(27, 31))
        lower_nose = list(range(31, 36))
        left_eye = list(range(36, 42))
        right_eye = list(range(42, 48))
        outer_lip = list(range(48, 60))
        inner_lip = list(range(60, 68))

        
        draw_mesh(shape, jaw, is_closed=False)
        draw_mesh(shape, left_eyebrow, is_closed=False)
        draw_mesh(shape, right_eyebrow, is_closed=False)
        draw_mesh(shape, nose_bridge, is_closed=False)
        draw_mesh(shape, lower_nose, is_closed=True)
        draw_mesh(shape, left_eye, is_closed=True)
        draw_mesh(shape, right_eye, is_closed=True)
        draw_mesh(shape, outer_lip, is_closed=True)
        draw_mesh(shape, inner_lip, is_closed=True)

        direction = get_face_direction(shape)
        
        direction_text = "Straight"
        if direction[0] > 5:
            direction_text = "Left"
        elif direction[0] < -5:
            direction_text = "Right"
        elif direction[1] > 5:
            direction_text = "Up"
        elif direction[1] < -5:
            direction_text = "Down"
        
        cv2.putText(frame, direction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        
        left_eye_ratio = get_blinking_ratio([36, 37, 38, 39, 40, 41], shape)
        right_eye_ratio = get_blinking_ratio([42, 43, 44, 45, 46, 47], shape)
        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2

        if blinking_ratio > 5.7:
            cv2.putText(frame, "BLINKING", (50, 150), font, 2, (255, 0, 0), 3)

        
        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], shape, gray)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], shape, gray)

        if gaze_ratio_left_eye is not None and gaze_ratio_right_eye is not None:
            gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2

            vertical_gaze_ratio_left_eye = get_vertical_gaze_ratio([36, 37, 38, 39, 40, 41], shape, gray)
            vertical_gaze_ratio_right_eye = get_vertical_gaze_ratio([42, 43, 44, 45, 46, 47], shape, gray)
            vertical_gaze_ratio = (vertical_gaze_ratio_right_eye + vertical_gaze_ratio_left_eye) / 2

            horizontal_gaze = ""
            vertical_gaze = ""

            if gaze_ratio <= 0.99:
                cv2.putText(frame, "LEFT", (50, 100), font, 2, (0, 0, 255), 3)
                horizontal_gaze = "LEFT"
            elif 0.99 < gaze_ratio < 1.15:
                cv2.putText(frame, "CENTER", (50, 100), font, 2, (0, 0, 255), 3)
                horizontal_gaze = "CENTER"
            else:
                cv2.putText(frame, "RIGHT", (50, 100), font, 2, (0, 0, 255), 3)
                horizontal_gaze = "RIGHT"

            if vertical_gaze_ratio < 0.8:
                cv2.putText(frame, "UP", (50, 50), font, 2, (0, 255, 255), 3)
                vertical_gaze = "UP"
            elif vertical_gaze_ratio > 1.4:
                cv2.putText(frame, "DOWN", (50, 50), font, 2, (0, 255, 255), 3)
                vertical_gaze = "DOWN"
            else:
                cv2.putText(frame, "CENTER", (50, 50), font, 2, (0, 255, 255), 3)
                vertical_gaze = "CENTER"

            gaze_data["horizontal_gaze"] = horizontal_gaze
            gaze_data["vertical_gaze"] = vertical_gaze

        
        for i, (x, y) in enumerate(shape):
            gaze_data[f"landmark_{i}_x"] = int(x)  # Convert to native Python int
            gaze_data[f"landmark_{i}_y"] = int(y)  # Convert to native Python int

    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        roi = np.expand_dims(roi_gray, axis=-1)
        roi = np.expand_dims(roi, axis=0)
        roi = roi.astype('float') / 255.0 

        preds = emotion_model.predict(roi)[0]
        emotion_label = class_labels[np.argmax(preds)]
        label_position = (x, y)
        cv2.putText(frame, emotion_label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
    
    with open('gaze_data.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        row = [datetime.now().isoformat(), gaze_data.get("horizontal_gaze", ""), gaze_data.get("vertical_gaze", ""), direction_text, emotion_label] 
        row += [gaze_data.get(f"landmark_{i}_x", "") for i in range(68)]
        row += [gaze_data.get(f"landmark_{i}_y", "") for i in range(68)]
        writer.writerow(row)
    
    return frame, gaze_data, direction_text, emotion_label



async def gaze_server(websocket, path):
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = imutils.resize(frame, width=450)
        frame = cv2.flip(frame, 1)
        
        frame, gaze_data, direction_text, emotion_label = process_frame(frame)
        
        
        gaze_data["emotion"] = emotion_label
        await websocket.send(json.dumps(gaze_data))

        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


start_server = websockets.serve(gaze_server, "127.0.0.1", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
