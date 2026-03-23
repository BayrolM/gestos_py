import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("modelo_gestos.pkl")

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

UMBRAL = 0.80   # confianza mínima para mostrar gesto

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand in result.multi_hand_landmarks:

            puntos = []

            for lm in hand.landmark:
                puntos.append(lm.x)
                puntos.append(lm.y)
                puntos.append(lm.z)

            # predicción
            probs = model.predict_proba([puntos])[0]
            clase = np.argmax(probs)
            confianza = probs[clase]

            if confianza > UMBRAL:

                pred = model.classes_[clase]

                # ---------- CAJA SEMI TRANSPARENTE ----------
                overlay = frame.copy()
                cv2.rectangle(overlay, (20,20), (280,90), (40,40,40), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)

                cv2.putText(frame,
                            f"Gesto: {pred}",
                            (40,65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,255,150),
                            2,
                            cv2.LINE_AA)

            # ---------- DIBUJAR MANO ----------
            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(255,255,255),
                    thickness=3,
                    circle_radius=3
                ),
                mp_drawing.DrawingSpec(
                    color=(128,128,128),
                    thickness=3
                )
            )

    cv2.putText(frame,
                "ESC para salir",
                (20, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200,200,200),
                2,
                cv2.LINE_AA)

    cv2.imshow("Reconocimiento de gestos", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()