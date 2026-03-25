import cv2
import mediapipe as mp
import joblib
import numpy as np
import os
import time
from collections import deque
import threading
import queue

# ==================== CONFIGURACIÓN OPTIMIZADA ====================
print("="*60)
print("SISTEMA DE RECONOCIMIENTO DE GESTOS - VERSIÓN OPTIMIZADA")
print("="*60)

# Verificar archivos necesarios
archivos_necesarios = ["modelo_gestos.pkl", "scaler.pkl", "label_encoder.pkl"]
archivos_faltantes = [f for f in archivos_necesarios if not os.path.exists(f)]

if archivos_faltantes:
    print("❌ Error: Faltan archivos:")
    for f in archivos_faltantes:
        print(f"   - {f}")
    print("\nPrimero ejecuta: python 2-entrenar_modelo.py")
    exit(1)

# Cargar modelo y preprocesadores
try:
    model = joblib.load("modelo_gestos.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    print(f"✓ Modelo cargado: {len(label_encoder.classes_)} gestos")
    print(f"✓ Gestos: {list(label_encoder.classes_)}")
except Exception as e:
    print(f"❌ Error al cargar: {e}")
    exit(1)

# ==================== CONFIGURACIÓN MEDIAPIPE ====================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# ==================== CONFIGURACIÓN DE CÁMARA ====================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ==================== PARÁMETROS OPTIMIZADOS ====================
UMBRAL_CONFIANZA = 0.65
HISTORIAL_LEN = 5
MIN_FRAMES_ESTABLES = 2

# Variables de estado
historial_predicciones = deque(maxlen=HISTORIAL_LEN)
historial_confianzas = deque(maxlen=HISTORIAL_LEN)
gesto_actual = None
frames_estables = 0
frame_count = 0
detecciones_totales = 0

tiempos_procesamiento = deque(maxlen=30)
fps = 0
ultimo_tiempo = time.time()
skip_frames = 0

# ==================== PALETA DE COLORES REDISEÑADA ====================
# Paleta neón sobre fondo oscuro — estilo terminal futurista
COLORES = {
    'pulgar_arriba':  (57, 255, 20),    # Verde neón
    'pulgar_abajo':   (0, 80, 255),     # Azul eléctrico
    'ok':             (0, 230, 255),    # Cian brillante
    'victoria':       (180, 0, 255),    # Violeta neón
    'paz':            (180, 0, 255),    # Violeta neón
    'corazon':        (0, 100, 255),    # Rojo-coral
    'rock':           (0, 165, 255),    # Naranja neón
    'spiderman':      (255, 0, 180),    # Magenta
    'puño':           (200, 200, 200),  # Gris platino
    'mano_abierta':   (0, 255, 200),    # Verde-cian
}
COLOR_DEFAULT = (0, 230, 255)

# Colores de estructura UI
UI_BG         = (10, 10, 18)       # Casi negro azulado
UI_ACCENT     = (0, 230, 255)      # Cian
UI_MUTED      = (80, 90, 110)      # Gris azulado
UI_TEXT       = (220, 230, 245)    # Blanco frío
UI_PANEL      = (15, 18, 30)       # Panel oscuro
UI_BAR_BG     = (30, 35, 50)      # Fondo barra
UI_HAND_LINE  = (0, 200, 180)      # Líneas de mano: teal
UI_HAND_JOINT = (0, 255, 160)      # Joints: verde menta
UI_HAND_BOX   = (0, 200, 255)      # Bounding box: cian

# ==================== FUNCIONES OPTIMIZADAS ====================

def predecir_gesto_optimizado(landmarks):
    try:
        landmarks_norm = scaler.transform(landmarks.reshape(1, -1))
        probs = model.predict_proba(landmarks_norm)[0]
        clase_idx = np.argmax(probs)
        confianza = probs[clase_idx]
        return clase_idx, confianza
    except:
        return None, 0

def suavizar_prediccion_rapida(clase_idx, confianza):
    global gesto_actual, frames_estables

    historial_predicciones.append(clase_idx)
    historial_confianzas.append(confianza)

    if len(historial_predicciones) < HISTORIAL_LEN:
        return None, 0, False

    from collections import Counter
    prediccion_frecuente = Counter(historial_predicciones).most_common(1)[0][0]
    confianza_promedio = np.mean([c for i, c in zip(historial_predicciones, historial_confianzas)
                                   if i == prediccion_frecuente])

    if prediccion_frecuente == gesto_actual:
        frames_estables += 1
    else:
        gesto_actual = prediccion_frecuente
        frames_estables = 1

    estable = frames_estables >= MIN_FRAMES_ESTABLES
    return prediccion_frecuente, confianza_promedio, estable


def dibujar_mano_optimizada(frame, hand_landmarks, confianza=None):
    """Dibujado rediseñado — líneas finas teal, joints brillantes"""
    h, w = frame.shape[:2]

    for connection in mp_hands.HAND_CONNECTIONS:
        start = hand_landmarks.landmark[connection[0]]
        end   = hand_landmarks.landmark[connection[1]]
        p1 = (int(start.x * w), int(start.y * h))
        p2 = (int(end.x * w),   int(end.y * h))
        cv2.line(frame, p1, p2, UI_HAND_LINE, 1, cv2.LINE_AA)

    # Todos los landmarks pequeños
    for idx, lm in enumerate(hand_landmarks.landmark):
        cx, cy = int(lm.x * w), int(lm.y * h)
        if idx in [0, 4, 8, 12, 16, 20]:  # Puntas y muñeca — más grandes
            cv2.circle(frame, (cx, cy), 4, UI_HAND_JOINT, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, UI_HAND_LINE, 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (cx, cy), 2, UI_HAND_LINE, -1, cv2.LINE_AA)

    x_coords = [lm.x for lm in hand_landmarks.landmark]
    y_coords = [lm.y for lm in hand_landmarks.landmark]
    x_min = int(min(x_coords) * w) - 10
    x_max = int(max(x_coords) * w) + 10
    y_min = int(min(y_coords) * h) - 10
    y_max = int(max(y_coords) * h) + 10

    # Esquinas del bounding box en lugar de rectángulo completo
    corner = 12
    t = 2
    c = UI_HAND_BOX
    # TL
    cv2.line(frame, (x_min, y_min), (x_min + corner, y_min), c, t, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_min), (x_min, y_min + corner), c, t, cv2.LINE_AA)
    # TR
    cv2.line(frame, (x_max, y_min), (x_max - corner, y_min), c, t, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_min), (x_max, y_min + corner), c, t, cv2.LINE_AA)
    # BL
    cv2.line(frame, (x_min, y_max), (x_min + corner, y_max), c, t, cv2.LINE_AA)
    cv2.line(frame, (x_min, y_max), (x_min, y_max - corner), c, t, cv2.LINE_AA)
    # BR
    cv2.line(frame, (x_max, y_max), (x_max - corner, y_max), c, t, cv2.LINE_AA)
    cv2.line(frame, (x_max, y_max), (x_max, y_max - corner), c, t, cv2.LINE_AA)


def dibujar_barra_confianza(frame, x, y, w_barra, h_barra, confianza, color):
    """Barra de confianza con estilo segmentado y glow"""
    # Fondo
    cv2.rectangle(frame, (x, y), (x + w_barra, y + h_barra), UI_BAR_BG, -1)

    # Segmentos (10 bloques con gap)
    n_segmentos = 10
    gap = 2
    seg_w = (w_barra - gap * (n_segmentos - 1)) // n_segmentos
    llenos = int(confianza * n_segmentos)

    for i in range(n_segmentos):
        sx = x + i * (seg_w + gap)
        if i < llenos:
            # Glow: dibuja versión ligeramente más grande y transparente primero
            overlay = frame.copy()
            cv2.rectangle(overlay, (sx - 1, y - 1), (sx + seg_w + 1, y + h_barra + 1), color, -1)
            cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
            cv2.rectangle(frame, (sx, y), (sx + seg_w, y + h_barra), color, -1)
        else:
            cv2.rectangle(frame, (sx, y), (sx + seg_w, y + h_barra), UI_MUTED, -1)

    # Borde exterior de la barra completa
    cv2.rectangle(frame, (x - 1, y - 1), (x + w_barra + 1, y + h_barra + 1), UI_MUTED, 1)


def dibujar_interfaz_optimizada(frame, gesto, confianza, fps, umbral):
    """Interfaz rediseñada — estilo HUD terminal futurista"""
    h, w = frame.shape[:2]

    # ---- Panel principal (izquierda arriba) ----
    panel_w, panel_h = 290, 110
    px, py = 8, 8

    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), UI_PANEL, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Borde izquierdo de acento
    cv2.rectangle(frame, (px, py), (px + 3, py + panel_h), UI_ACCENT, -1)

    # Línea superior decorativa
    cv2.line(frame, (px + 3, py), (px + panel_w, py), UI_MUTED, 1)
    cv2.line(frame, (px + 3, py + panel_h), (px + panel_w, py + panel_h), UI_MUTED, 1)

    if gesto:
        color = COLORES.get(gesto.lower(), COLOR_DEFAULT)
        nombre = gesto.upper().replace('_', ' ')

        # Etiqueta pequeña
        cv2.putText(frame, "GESTO DETECTADO", (px + 10, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, UI_MUTED, 1, cv2.LINE_AA)

        # Nombre del gesto — grande y con color
        cv2.putText(frame, nombre, (px + 10, py + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

        # Porcentaje
        pct_text = f"{confianza*100:.0f}%"
        cv2.putText(frame, pct_text, (px + 10, py + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, UI_TEXT, 1, cv2.LINE_AA)

        # Barra segmentada
        dibujar_barra_confianza(frame, px + 10, py + 82, panel_w - 22, 10, confianza, color)

    else:
        cv2.putText(frame, "ESPERANDO...", (px + 10, py + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, UI_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, "Sin gesto", (px + 10, py + 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, UI_MUTED, 1, cv2.LINE_AA)
        dibujar_barra_confianza(frame, px + 10, py + 82, panel_w - 22, 10, 0, UI_MUTED)

    # ---- Panel de métricas (esquina sup derecha) ----
    m_w, m_h = 90, 55
    mx = w - m_w - 8
    my = 8

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (mx, my), (mx + m_w, my + m_h), UI_PANEL, -1)
    cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (mx, my), (mx + m_w, my + 2), UI_ACCENT, -1)

    cv2.putText(frame, "FPS", (mx + 6, my + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, UI_MUTED, 1, cv2.LINE_AA)
    cv2.putText(frame, f"{int(fps)}", (mx + 6, my + 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, UI_ACCENT, 2, cv2.LINE_AA)

    cv2.putText(frame, f"U:{int(umbral*100)}%", (mx + 6, my + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, UI_MUTED, 1, cv2.LINE_AA)

    # ---- Barra umbral (debajo del panel de métricas) ----
    u_y = my + m_h + 6
    cv2.putText(frame, "UMBRAL", (mx + 4, u_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.28, UI_MUTED, 1, cv2.LINE_AA)
    dibujar_barra_confianza(frame, mx, u_y + 14, m_w, 6, umbral, UI_ACCENT)

    # ---- Controles (pie de página) ----
    overlay3 = frame.copy()
    cv2.rectangle(overlay3, (0, h - 22), (w, h), UI_PANEL, -1)
    cv2.addWeighted(overlay3, 0.7, frame, 0.3, 0, frame)
    cv2.line(frame, (0, h - 22), (w, h - 22), UI_MUTED, 1)

    controles = "[+] sube umbral    [-] baja umbral    [ESC] salir"
    cv2.putText(frame, controles, (10, h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, UI_MUTED, 1, cv2.LINE_AA)

    return frame

# ==================== LOOP PRINCIPAL ====================
print("\n" + "="*60)
print("🎯 SISTEMA OPTIMIZADO - INICIANDO...")
print("="*60)
print("Controles:")
print("  [+] Aumentar umbral")
print("  [-] Disminuir umbral")
print("  [ESC] Salir")
print("="*60)

frame_skip = 0
MAX_SKIP = 1
ultimo_procesamiento = 0
MIN_PROCESAMIENTO_MS = 30

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al leer cámara")
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    if frame_count % 10 == 0:
        tiempo_actual = time.time()
        fps = 10 / (tiempo_actual - ultimo_tiempo)
        ultimo_tiempo = tiempo_actual

    tiempo_actual_ms = time.time() * 1000
    procesar_frame = (tiempo_actual_ms - ultimo_procesamiento) >= MIN_PROCESAMIENTO_MS

    if procesar_frame:
        ultimo_procesamiento = tiempo_actual_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        gesto_detectado = None
        confianza_detectada = 0

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                puntos = []
                for lm in hand.landmark:
                    puntos.extend([lm.x, lm.y, lm.z])

                clase_idx, confianza = predecir_gesto_optimizado(np.array(puntos))

                if clase_idx is not None and confianza > UMBRAL_CONFIANZA:
                    clase_suavizada, conf_suavizada, estable = suavizar_prediccion_rapida(clase_idx, confianza)

                    if estable and clase_suavizada is not None:
                        gesto_detectado = label_encoder.inverse_transform([clase_suavizada])[0]
                        confianza_detectada = conf_suavizada
                        detecciones_totales += 1

                dibujar_mano_optimizada(frame, hand, confianza)
        else:
            historial_predicciones.clear()
            historial_confianzas.clear()
            gesto_actual = None
            frames_estables = 0

        frame = dibujar_interfaz_optimizada(frame, gesto_detectado, confianza_detectada, fps, UMBRAL_CONFIANZA)

    cv2.imshow("Reconocimiento de Gestos - Optimizado", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        print("\n🛑 Saliendo...")
        break
    elif key == ord('+') or key == ord('='):
        UMBRAL_CONFIANZA = min(0.95, UMBRAL_CONFIANZA + 0.05)
        print(f"✓ Umbral: {UMBRAL_CONFIANZA*100:.0f}%")
    elif key == ord('-') or key == ord('_'):
        UMBRAL_CONFIANZA = max(0.4, UMBRAL_CONFIANZA - 0.05)
        print(f"✓ Umbral: {UMBRAL_CONFIANZA*100:.0f}%")

# ==================== ESTADÍSTICAS FINALES ====================
cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("ESTADÍSTICAS FINALES")
print("="*60)
print(f"Frames procesados: {frame_count}")
print(f"Detecciones exitosas: {detecciones_totales}")
if frame_count > 0:
    print(f"Tasa de detección: {(detecciones_totales/frame_count)*100:.1f}%")
print("="*60)
print("✅ Sistema cerrado correctamente")