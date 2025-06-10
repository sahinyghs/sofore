import cv2
import dlib
import numpy as np
import pygame
from ultralytics import YOLO

# EAR hesaplama fonksiyonu
def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Uyku alarmı
def play_sleep_alarm():
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    pygame.mixer.music.load("wrong.wav")
    pygame.mixer.music.play()

# Diğer durumlarda alarm
def play_once_alarm():
    if not pygame.mixer.get_init():
        pygame.mixer.init()
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load("beep.wav")
        pygame.mixer.music.play()

# Parametreler
EAR_THRESHOLD = 0.22
CONSEC_FRAMES = 30  # yaklaşık 1.5 saniye
eye_closed_counter = 0

# dlib modelleri
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = (42, 48)
(rStart, rEnd) = (36, 42)

# YOLOv8 modeli
model = YOLO("yolov8n.pt")
WARNING_CLASSES = ["cell phone", "book"]

# Başlat
pygame.init()
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()])
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = calculate_ear(leftEye)
        rightEAR = calculate_ear(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # Göz çizimleri
        cv2.polylines(frame, [leftEye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [rightEye], True, (0, 255, 0), 1)

        # Uyku tespiti
        if ear < EAR_THRESHOLD:
            eye_closed_counter += 1
            if eye_closed_counter >= CONSEC_FRAMES:
                cv2.putText(frame, "UYARI: Uyuma Tespit Edildi!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                play_sleep_alarm()
        else:
            eye_closed_counter = 0

    # YOLO ile davranış kontrolü
    results = model.predict(frame, conf=0.4, verbose=False)
    for result in results:
        boxes = result.boxes
        names = result.names
        for box in boxes:
            cls_id = int(box.cls[0])
            label = names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in WARNING_CLASSES:
                color = (0, 0, 255)
                cv2.putText(frame, f"UYARI: {label.upper()}!", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                play_once_alarm()

    # Görüntüyü göster
    cv2.imshow("Sofor Following System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kapat
cap.release()
cv2.destroyAllWindows()
pygame.quit()
