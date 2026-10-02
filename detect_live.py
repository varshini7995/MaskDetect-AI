import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

model = load_model("models/model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

live_stats = {
    "total_faces": 0,
    "with_mask": 0,
    "without_mask": 0
}

def detect_face_mask_frame(frame):
    global live_stats
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    faces_list = []
    num_mask = 0
    num_without_mask = 0

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        if face.shape[0] > 0 and face.shape[1] > 0:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            faces_list.append(face)

    if len(faces_list) > 0:
        faces_list = np.array(faces_list, dtype="float32")
        preds_list = model.predict(faces_list, batch_size=32)

        for (box, pred) in zip(faces, preds_list):
            (x, y, w, h) = box
            mask, withoutMask = pred

            is_mask = mask > withoutMask
            label = "Mask" if is_mask else "No Mask"
            color = (0, 255, 0) if is_mask else (0, 0, 255)

            if is_mask:
                num_mask += 1
            else:
                num_without_mask += 1

            label_text = f"{label}: {max(mask, withoutMask) * 100:.1f}%"
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    live_stats["total_faces"] = len(faces)
    live_stats["with_mask"] = num_mask
    live_stats["without_mask"] = num_without_mask

    return frame

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame = detect_face_mask_frame(frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')