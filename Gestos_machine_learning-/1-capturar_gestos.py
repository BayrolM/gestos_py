import cv2
import mediapipe as mp
import csv

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

archivo = open("gestos.csv", "a", newline="")
writer = csv.writer(archivo)

gesto = input("Nombre del gesto: ")

contador = 0
objetivo = 200   # número de muestras que quieres capturar

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

            # texto instrucciones
            cv2.putText(frame, "Presiona S para guardar", (10,40),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

            # contador de muestras
            cv2.putText(frame, f"Fotos: {contador}/{objetivo}", (10,80),
                        cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            key = cv2.waitKey(1)

            if key == ord("s"):
                writer.writerow(puntos + [gesto])
                contador += 1
                print(f"Guardadas: {contador}")

            # dibujar mano
            mp_drawing.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Captura de gestos", frame)

    # parar automáticamente al llegar al objetivo
    if contador >= objetivo:
        print("Dataset completo")
        break

    if cv2.waitKey(1) == 27:
        break

cap.release()
archivo.close()
cv2.destroyAllWindows()