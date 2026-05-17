# =============================================================================
#  PROYECTO: RECONOCIMIENTO DE LENGUA DE SIGNOS CON INTELIGENCIA ARTIFICIAL
#  Asignatura: Tecnología e Informática · 3º ESO
# =============================================================================
#
#  DESCRIPCIÓN:
#  Este script hace tres cosas en orden:
#    1. RECOGER DATOS   → graba ejemplos de cada letra con la webcam
#    2. ENTRENAR        → enseña a la IA a reconocer las letras
#    3. USAR LA APP     → reconoce letras en tiempo real y habla
#
#  CÓMO USARLO (abre una terminal y ejecuta):
#    python lse_proyecto.py recoger    ← para grabar muestras
#    python lse_proyecto.py entrenar   ← para entrenar el modelo
#    python lse_proyecto.py usar       ← para usar la app
#
#  INSTALACIÓN DE DEPENDENCIAS (solo la primera vez):
#    pip install mediapipe opencv-python scikit-learn pyttsx3
#
#  COMPATIBLE CON: MediaPipe 0.10.x  ·  Python 3.8+
#
# =============================================================================

import sys
import os

# ─────────────────────────────────────────────────────────────────────────────
#  FIX: forzar backend X11 en OpenCV para evitar el error de fuentes Qt.
#  Debe hacerse ANTES de importar cv2.
# ─────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

import cv2
import mediapipe as mp
import numpy as np
import json
import pickle
import time
import threading
import urllib.request
from collections import deque, Counter

# ─────────────────────────────────────────────────────────────────────────────
#  NUEVA API DE MEDIAPIPE 0.10  (Tasks API)
#  A partir de la versión 0.10, MediaPipe ya no usa mp.solutions.hands.
#  Ahora usa mp.tasks y requiere un fichero de modelo .task descargado.
# ─────────────────────────────────────────────────────────────────────────────
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe import Image, ImageFormat

# Ruta local donde se guarda el modelo de detección de manos
MODELO_MP = "hand_landmarker.task"
MODELO_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def descargar_modelo_mp():
    """
    Descarga el fichero del modelo de MediaPipe si no existe.
    Solo se necesita hacer una vez. Pesa ~4 MB.
    """
    if os.path.exists(MODELO_MP):
        return
    print(f"\n  Descargando modelo de MediaPipe ({MODELO_URL[:60]}...)")
    print("  (Solo ocurre la primera vez, pesa ~4 MB)\n")
    try:
        urllib.request.urlretrieve(MODELO_URL, MODELO_MP)
        print(f"  Modelo guardado en '{MODELO_MP}' ✓\n")
    except Exception as e:
        print(f"\n  ERROR: No se pudo descargar el modelo de MediaPipe.")
        print(f"  Detalles: {e}")
        print(f"\n  Descárgalo manualmente desde:")
        print(f"  {MODELO_URL}")
        print(f"  y colócalo en la misma carpeta que este script.\n")
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN GENERAL
# ─────────────────────────────────────────────────────────────────────────────
LETRAS             = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
MUESTRAS_POR_LETRA = 80
FICHERO_DATOS      = 'lse_datos.json'
FICHERO_MODELO     = 'lse_modelo.pkl'


# =============================================================================
#  FUNCIÓN: CREAR EL DETECTOR DE MANOS (nueva API)
#  En MediaPipe 0.10 el detector se crea con HandLandmarker y un fichero
#  de modelo .task. Devuelve un objeto detector listo para usar.
# =============================================================================
def crear_detector(modo=RunningMode.IMAGE):
    """
    Crea y devuelve un detector de landmarks de mano.
    modo=RunningMode.IMAGE    → para procesar fotogramas uno a uno (recogida)
    modo=RunningMode.VIDEO    → para procesar vídeo en tiempo real (app)
    """
    opciones = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODELO_MP),
        running_mode=modo,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(opciones)


# =============================================================================
#  FUNCIÓN: PROCESAR FRAME Y OBTENER LANDMARKS
#  Recibe un fotograma BGR de OpenCV y devuelve un diccionario
#  {0: (x,y), 1: (x,y), ..., 20: (x,y)} o None si no hay mano.
# =============================================================================
def obtener_landmarks(detector, frame_bgr, timestamp_ms=0, modo=RunningMode.IMAGE):
    """
    Convierte el frame a formato MediaPipe, lo procesa y devuelve
    los 21 landmarks normalizados (coordenadas entre 0 y 1).
    """
    # MediaPipe 0.10 necesita el frame en RGB
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image  = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    if modo == RunningMode.VIDEO:
        resultado = detector.detect_for_video(mp_image, timestamp_ms)
    else:
        resultado = detector.detect(mp_image)

    if not resultado.hand_landmarks:
        return None   # No se detectó ninguna mano

    # Cogemos los landmarks de la primera mano detectada
    landmarks = resultado.hand_landmarks[0]
    return {i: (lm.x, lm.y) for i, lm in enumerate(landmarks)}


# =============================================================================
#  FUNCIÓN: DIBUJAR LANDMARKS SOBRE EL FRAME
#  En la nueva API no existe mp_drawing, así que lo hacemos manualmente.
# =============================================================================
# Conexiones entre puntos de la mano (igual que en la API antigua)
CONEXIONES = [
    (0,1),(1,2),(2,3),(3,4),         # Pulgar
    (0,5),(5,6),(6,7),(7,8),         # Índice
    (0,9),(9,10),(10,11),(11,12),    # Medio
    (0,13),(13,14),(14,15),(15,16),  # Anular
    (0,17),(17,18),(18,19),(19,20),  # Meñique
    (5,9),(9,13),(13,17),            # Nudillos
]

def dibujar_mano(frame, landmarks_dict, confianza=None):
    """
    Dibuja los puntos y líneas de la mano con colores distintos por dedo.
    Cada dedo tiene su color para que los alumnos identifiquen qué detecta la IA.
      Pulgar  → amarillo   Índice → cian   Medio → verde
      Anular  → naranja    Meñique → rosa  Muñeca → rojo
    Si se pasa 'confianza', dibuja también una barra de confianza.
    """
    h, w, _ = frame.shape
    pts = {i: (int(x * w), int(y * h)) for i, (x, y) in landmarks_dict.items()}

    # Colores por dedo: {id_punto: color_BGR}
    COLORES_DEDO = {
        # Pulgar (amarillo)
        1:(0,220,255), 2:(0,220,255), 3:(0,220,255), 4:(0,220,255),
        # Índice (cian)
        5:(255,220,0), 6:(255,220,0), 7:(255,220,0), 8:(255,220,0),
        # Medio (verde)
        9:(0,255,120), 10:(0,255,120), 11:(0,255,120), 12:(0,255,120),
        # Anular (naranja)
        13:(0,140,255), 14:(0,140,255), 15:(0,140,255), 16:(0,140,255),
        # Meñique (rosa)
        17:(200,0,255), 18:(200,0,255), 19:(200,0,255), 20:(200,0,255),
    }

    # Segmentos agrupados por dedo para colorear líneas igual que puntos
    SEGMENTOS_COLOR = [
        ([(0,1),(1,2),(2,3),(3,4)],   (0,220,255)),   # Pulgar
        ([(0,5),(5,6),(6,7),(7,8)],   (255,220,0)),   # Índice
        ([(0,9),(9,10),(10,11),(11,12)],(0,255,120)), # Medio
        ([(0,13),(13,14),(14,15),(15,16)],(0,140,255)),# Anular
        ([(0,17),(17,18),(18,19),(19,20)],(200,0,255)),# Meñique
        ([(5,9),(9,13),(13,17)],      (180,180,180)), # Nudillos
    ]

    # Líneas con color por dedo
    for segmentos, color in SEGMENTOS_COLOR:
        for a, b in segmentos:
            cv2.line(frame, pts[a], pts[b], color, 2)

    # Puntos
    for i, (px, py) in pts.items():
        if i == 0:
            cv2.circle(frame, (px, py), 7, (0, 0, 255), -1)   # Muñeca: rojo
        else:
            color = COLORES_DEDO.get(i, (255, 255, 255))
            radio = 6 if i in [4, 8, 12, 16, 20] else 4       # Puntas más grandes
            cv2.circle(frame, (px, py), radio, color, -1)
            cv2.circle(frame, (px, py), radio + 1, (0,0,0), 1) # Borde negro

    # Barra de confianza (si se proporciona)
    if confianza is not None:
        barra_x, barra_y, barra_w, barra_h = 12, 45, 180, 14
        # Fondo
        cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h),
                      (40, 40, 40), -1)
        # Relleno según confianza
        fill = int(barra_w * confianza)
        if confianza > 0.85:
            color_barra = (0, 220, 100)     # Verde: alta confianza
        elif confianza > 0.70:
            color_barra = (0, 180, 255)     # Naranja: confianza media
        else:
            color_barra = (0, 80, 220)      # Rojo: baja confianza
        cv2.rectangle(frame, (barra_x, barra_y), (barra_x + fill, barra_y + barra_h),
                      color_barra, -1)
        cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h),
                      (120, 120, 120), 1)
        cv2.putText(frame, f"Confianza: {int(confianza*100)}%",
                    (barra_x, barra_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


# =============================================================================
#  FUNCIÓN: EXTRAER CARACTERÍSTICAS DE LA MANO
#  Convierte los 21 landmarks en un vector de 42 números normalizado.
#
#  ¿Por qué normalizar?
#  Si la mano está cerca de la cámara, los valores son distintos que si
#  está lejos. Al dividir por la escala obtenemos el mismo resultado
#  independientemente de la distancia o posición en pantalla.
# =============================================================================
def extraer_caracteristicas(landmarks):
    """
    Entrada:  diccionario {0: (x,y), ..., 20: (x,y)}
    Salida:   lista de 42 números (las "características" de la mano)
              o None si hay algún problema
    """
    muneca = np.array(landmarks[0])       # Punto 0 = muñeca (referencia)

    # Escala: distancia muñeca → nudillo central (punto 9)
    escala = np.linalg.norm(np.array(landmarks[9]) - muneca)
    if escala == 0:
        return None

    caracteristicas = []
    for i in range(21):
        punto      = np.array(landmarks[i])
        normalizado = (punto - muneca) / escala   # Centramos y escalamos
        caracteristicas.extend(normalizado.tolist())

    return caracteristicas   # Lista de 42 números


# =============================================================================
#  FASE 1: RECOGER DATOS
# =============================================================================
def recoger_datos(modo_mas=False, solo_letra=None):
    print("\n╔══════════════════════════════════════╗")
    print("║   FASE 1: RECOGIDA DE DATOS           ║")
    print("╚══════════════════════════════════════╝")
    if modo_mas:
        print("  Modo: AÑADIR MÁS MUESTRAS (--mas) → se añaden 80 muestras más por letra")
    if solo_letra:
        print(f"  Solo se grabará la letra: '{solo_letra.upper()}'")
    print(f"  Letras: {len(LETRAS)}  ·  Muestras base: {MUESTRAS_POR_LETRA}")

    descargar_modelo_mp()

    # Cargamos datos existentes para no perder lo ya grabado
    if os.path.exists(FICHERO_DATOS):
        with open(FICHERO_DATOS, 'r') as f:
            datos = json.load(f)
        print(f"  Datos existentes cargados desde '{FICHERO_DATOS}'")
    else:
        datos = {letra: [] for letra in LETRAS}

    for letra in LETRAS:
        if letra not in datos:
            datos[letra] = []

    letras_a_grabar = [solo_letra.upper()] if solo_letra else LETRAS

    cap      = cv2.VideoCapture(0)
    detector = crear_detector(modo=RunningMode.IMAGE)

    for letra in letras_a_grabar:
        ya_tenemos = len(datos[letra])
        if not modo_mas and ya_tenemos >= MUESTRAS_POR_LETRA:
            print(f"  '{letra}' → ya tiene {ya_tenemos} muestras ✓")
            continue

        objetivo = ya_tenemos + MUESTRAS_POR_LETRA if modo_mas else MUESTRAS_POR_LETRA

        print(f"\n  Prepárate para la letra '{letra}' ({ya_tenemos} → objetivo {objetivo})")
        print("  Pulsa ESPACIO para grabar · Q para saltar esta letra")

        # ── Pantalla de espera ──────────────────────────────────────
        esperando = True
        while esperando:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            lm = obtener_landmarks(detector, frame)
            if lm:
                dibujar_mano(frame, lm)

            cv2.putText(frame, f"Letra: {letra}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 200, 255), 3)
            cv2.putText(frame, "ESPACIO = grabar  |  Q = saltar", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.putText(frame, f"Muestras: {ya_tenemos}/{objetivo}", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)
            cv2.imshow('LSE - Recogida de datos', frame)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord(' '):
                esperando = False
            elif tecla == ord('q'):
                esperando = False
                break

        # ── Grabación ───────────────────────────────────────────────
        contador = ya_tenemos
        while contador < objetivo:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)

            lm = obtener_landmarks(detector, frame)
            if lm:
                dibujar_mano(frame, lm)
                caract = extraer_caracteristicas(lm)
                if caract:
                    datos[letra].append(caract)
                    contador += 1

            # Barra de progreso
            progreso = int(((contador - ya_tenemos) / MUESTRAS_POR_LETRA) * 400)
            cv2.rectangle(frame, (10, 160), (410, 182), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, 160), (10 + progreso, 182), (0, 200, 100), -1)
            cv2.putText(frame, f"GRABANDO '{letra}': {contador}/{objetivo}",
                        (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 100), 2)
            cv2.imshow('LSE - Recogida de datos', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        with open(FICHERO_DATOS, 'w') as f:
            json.dump(datos, f)
        print(f"  '{letra}' guardada → {contador} muestras ✓")

    detector.close()
    cap.release()
    cv2.destroyAllWindows()
    print("\n  Recogida completada.")
    print(f"  Ahora ejecuta:  python lse_proyecto.py entrenar\n")


# =============================================================================
#  FASE 2: ENTRENAR EL MODELO
# =============================================================================
def entrenar_modelo():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    print("\n╔══════════════════════════════════════╗")
    print("║   FASE 2: ENTRENAMIENTO DEL MODELO    ║")
    print("╚══════════════════════════════════════╝\n")

    if not os.path.exists(FICHERO_DATOS):
        print(f"  ERROR: No se encuentra '{FICHERO_DATOS}'")
        print("  Primero ejecuta:  python lse_proyecto.py recoger\n")
        return

    with open(FICHERO_DATOS, 'r') as f:
        datos = json.load(f)

    X, y = [], []
    for letra, muestras in datos.items():
        for muestra in muestras:
            X.append(muestra)
            y.append(letra)

    X = np.array(X)
    y = np.array(y)

    print(f"  Dataset:  {len(X)} muestras  ·  {len(set(y))} letras")
    print(f"  Features: {X.shape[1]} por muestra\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        random_state=42,
        n_jobs=-1          # Usa todos los núcleos del procesador
    )

    print("  Entrenando... (puede tardar unos segundos)")
    modelo.fit(X_train, y_train)
    print("  ¡Completado!\n")

    y_pred   = modelo.predict(X_test)
    precision = (y_pred == y_test).mean() * 100
    print(f"  Precisión global: {precision:.1f}%\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    with open(FICHERO_MODELO, 'wb') as f:
        pickle.dump(modelo, f)
    print(f"\n  Modelo guardado en '{FICHERO_MODELO}' ✓")
    print(f"  Ahora ejecuta:  python lse_proyecto.py usar\n")


# =============================================================================
#  FASE 3: USAR LA APLICACIÓN EN TIEMPO REAL
# =============================================================================
class BufferLetras:
    """
    Acumula letras detectadas y forma palabras.
    - Una letra se confirma si aparece estable N frames seguidos.
    - Una pausa de 2 segundos sin mano pronuncia la palabra acumulada.
    """
    def __init__(self, frames_confirmar=15, pausa_segundos=2.0):
        self.frames_confirmar = frames_confirmar
        self.pausa_segundos   = pausa_segundos
        self.letra_actual     = None
        self.contador_frames  = 0
        self.palabra          = []
        self.ultima_anadida   = None
        self.ultima_mano      = time.time()

    def actualizar(self, letra):
        self.ultima_mano = time.time()
        if letra == self.letra_actual:
            self.contador_frames += 1
        else:
            self.letra_actual    = letra
            self.contador_frames = 1
        if self.contador_frames == self.frames_confirmar and letra != self.ultima_anadida:
            self.palabra.append(letra)
            self.ultima_anadida = letra
            return f"✓ Letra '{letra}' confirmada"
        return None

    def comprobar_pausa(self):
        if self.palabra and (time.time() - self.ultima_mano) > self.pausa_segundos:
            palabra = ''.join(self.palabra)
            self.palabra        = []
            self.ultima_anadida = None
            return palabra
        return None


def usar_app():
    # ── Síntesis de voz ───────────────────────────────────────────
    try:
        import pyttsx3
        motor_voz = pyttsx3.init()
        motor_voz.setProperty('rate', 145)
        motor_voz.setProperty('volume', 1.0)
        for voz in motor_voz.getProperty('voices'):
            if 'spanish' in voz.name.lower() or 'es' in voz.id.lower():
                motor_voz.setProperty('voice', voz.id)
                break
        voz_disponible = True
    except Exception:
        print("  AVISO: pyttsx3 no disponible. La app funcionará sin voz.")
        voz_disponible = False

    def hablar(texto):
        if not voz_disponible:
            return
        def _hablar():
            motor_voz.say(texto)
            motor_voz.runAndWait()
        threading.Thread(target=_hablar, daemon=True).start()

    print("\n╔══════════════════════════════════════╗")
    print("║   FASE 3: APLICACIÓN EN TIEMPO REAL   ║")
    print("╚══════════════════════════════════════╝\n")

    if not os.path.exists(FICHERO_MODELO):
        print(f"  ERROR: No se encuentra '{FICHERO_MODELO}'")
        print("  Primero ejecuta:  python lse_proyecto.py entrenar\n")
        return

    descargar_modelo_mp()

    with open(FICHERO_MODELO, 'rb') as f:
        modelo = pickle.load(f)
    print("  Modelo cargado ✓")
    print("  Controles: Q = salir · Espacio = borrar palabra\n")

    # ── Inicialización de cámara ──────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # ── Detección del backend de ventanas disponible ──────────────
    # OpenCV compilado con Qt pero sin fuentes (caso conda) no puede
    # abrir ventanas. Lo detectamos y usamos matplotlib como fallback.
    NOMBRE_VENTANA   = 'LSE - Voz en tiempo real'
    usar_matplotlib  = False

    import io, contextlib, subprocess
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
            prueba = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imshow(NOMBRE_VENTANA, prueba)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
        # Si el stderr contiene el error de fuentes Qt, usamos matplotlib
        if 'QFontDatabase' in buf.getvalue() or 'Cannot find font' in buf.getvalue():
            usar_matplotlib = True
            print("  Qt sin fuentes detectado → cambiando a matplotlib")
        else:
            print("  Backend de ventanas: OpenCV ✓")
    except Exception as e:
        usar_matplotlib = True
        print(f"  OpenCV window error: {e} → cambiando a matplotlib")

    # Configuramos matplotlib si es necesario
    if usar_matplotlib:
        try:
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            fig.canvas.manager.set_window_title(NOMBRE_VENTANA)
            plt.axis('off')
            img_plot = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            plt.tight_layout(pad=0)
            plt.ion()
            plt.show()
            print("  Ventana matplotlib (TkAgg) abierta ✓")
        except Exception:
            import matplotlib
            matplotlib.use('Qt5Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.axis('off')
            img_plot = ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
            plt.tight_layout(pad=0)
            plt.ion()
            plt.show()
            print("  Ventana matplotlib (Qt5Agg) abierta ✓")
        print("  NOTA: Para salir pulsa Q en la terminal o cierra la ventana\n")
    else:
        cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(NOMBRE_VENTANA, 640, 480)
        print("  Ventana OpenCV abierta ✓\n")

    detector  = crear_detector(modo=RunningMode.VIDEO)
    buffer    = BufferLetras(frames_confirmar=15, pausa_segundos=2.0)
    historial = deque(maxlen=7)

    msg_estado     = ""
    msg_tiempo     = 0
    ultima_palabra = ""
    timestamp_ms   = 0
    confianza_actual = 0.0

    # Leyenda de colores de dedos (para mostrar en pantalla)
    LEYENDA = [
        ("Pulgar",  (0, 220, 255)),
        ("Indice",  (255, 220, 0)),
        ("Medio",   (0, 255, 120)),
        ("Anular",  (0, 140, 255)),
        ("Menique", (200, 0, 255)),
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)   # Espejo: la mano derecha aparece a la derecha
        h, w, _ = frame.shape

        timestamp_ms += 33
        lm = obtener_landmarks(detector, frame, timestamp_ms, RunningMode.VIDEO)
        letra_detectada = None

        # ── Detección y clasificación ─────────────────────────────
        if lm:
            caract = extraer_caracteristicas(lm)
            if caract:
                probabilidades   = modelo.predict_proba([caract])[0]
                confianza_actual = float(probabilidades.max())

                if confianza_actual > 0.70:
                    letra = modelo.classes_[probabilidades.argmax()]
                    historial.append(letra)
                    letra_votada    = Counter(historial).most_common(1)[0][0]
                    letra_detectada = letra_votada

                    msg = buffer.actualizar(letra_votada)
                    if msg:
                        msg_estado = msg
                        msg_tiempo = time.time()

            # Dibujamos la mano con colores por dedo y barra de confianza
            dibujar_mano(frame, lm, confianza=confianza_actual)
        else:
            confianza_actual = 0.0

        # ── Pausa → pronunciar palabra ────────────────────────────
        palabra_lista = buffer.comprobar_pausa()
        if palabra_lista:
            ultima_palabra = palabra_lista
            print(f"  Pronunciando: '{palabra_lista}'")
            hablar(palabra_lista)
            msg_estado = f"Pronunciando: '{palabra_lista}'"
            msg_tiempo = time.time()

        # ════════════════════════════════════════════════════════
        #  HUD — interfaz visual superpuesta sobre la cámara
        # ════════════════════════════════════════════════════════

        # ── Marco guía de encuadre ───────────────────────────────
        # Rectángulo central que indica dónde colocar la mano
        margen_x = int(w * 0.25)
        margen_y = int(h * 0.15)
        # Color del marco: verde si detecta mano, gris si no
        color_marco = (0, 220, 100) if lm else (100, 100, 100)
        grosor_marco = 3 if lm else 1
        cv2.rectangle(frame,
                      (margen_x, margen_y),
                      (w - margen_x, h - margen_y - 100),
                      color_marco, grosor_marco)
        # Texto dentro del marco si no hay mano
        if not lm:
            texto_guia = "Coloca tu mano aqui"
            tx = margen_x + 10
            ty = margen_y + 30
            cv2.putText(frame, texto_guia, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (140, 140, 140), 1)

        # Etiquetas de esquina del marco
        tam = 18
        for (px, py) in [(margen_x, margen_y),
                         (w - margen_x, margen_y),
                         (margen_x, h - margen_y - 100),
                         (w - margen_x, h - margen_y - 100)]:
            cv2.circle(frame, (px, py), 5, color_marco, -1)

        # ── Franja inferior semitransparente ─────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 100), (w, h), (10, 10, 28), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

        # ── Letra detectada (grande, esquina superior derecha) ────
        if letra_detectada:
            # Fondo de la letra
            cv2.rectangle(frame, (w - 105, 8), (w - 8, 105), (20, 20, 50), -1)
            cv2.rectangle(frame, (w - 105, 8), (w - 8, 105), (0, 220, 100), 2)
            cv2.putText(frame, letra_detectada, (w - 88, 92),
                        cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0, 220, 100), 5)

        # ── Progreso de confirmación (mini barra bajo la letra) ───
        if letra_detectada and buffer.letra_actual:
            progreso_conf = min(buffer.contador_frames / buffer.frames_confirmar, 1.0)
            bx, by, bw, bht = w - 105, 108, 97, 8
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bht), (40,40,40), -1)
            cv2.rectangle(frame, (bx, by), (bx + int(bw * progreso_conf), by + bht),
                          (0, 200, 80), -1)

        # ── Palabra en construcción ───────────────────────────────
        palabra_actual = ''.join(buffer.palabra)
        cv2.putText(frame, f"Palabra: {palabra_actual}_",
                    (12, h - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 100), 2)

        # ── Última palabra pronunciada ────────────────────────────
        if ultima_palabra:
            cv2.putText(frame, f"Dicho: {ultima_palabra}",
                        (12, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 210, 255), 2)

        # ── Mensaje de estado temporal ────────────────────────────
        if time.time() - msg_tiempo < 2.5:
            cv2.putText(frame, msg_estado,
                        (12, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (160, 160, 160), 1)

        # ── Leyenda de colores de dedos (esquina inferior derecha) ─
        ley_x = w - 115
        ley_y = h - 98
        for i, (nombre, color) in enumerate(LEYENDA):
            y = ley_y + i * 18
            cv2.circle(frame, (ley_x + 6, y), 5, color, -1)
            cv2.putText(frame, nombre, (ley_x + 16, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

        # ── Controles ─────────────────────────────────────────────
        cv2.putText(frame, "Q=salir  ESPACIO=borrar",
                    (12, h - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

        # ── Mostrar frame (backend automático) ───────────────────
        if usar_matplotlib:
            img_plot.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fig.canvas.draw()
            fig.canvas.flush_events()
            if not plt.fignum_exists(fig.number):
                break
        else:
            cv2.imshow(NOMBRE_VENTANA, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord('q') or tecla == 27:
                break
            elif tecla == ord(' '):
                buffer.palabra        = []
                buffer.ultima_anadida = None
                msg_estado = "Palabra borrada"
                msg_tiempo = time.time()

    # ── Limpieza ──────────────────────────────────────────────────
    detector.close()
    cap.release()
    if usar_matplotlib:
        plt.close('all')
    else:
        cv2.destroyAllWindows()
    print("\n  Aplicación cerrada.\n")


# =============================================================================
#  PUNTO DE ENTRADA
# =============================================================================
def mostrar_ayuda():
    print("""
  ╔═══════════════════════════════════════════════════════╗
  ║   PROYECTO LSE · Lengua de Signos + IA               ║
  ║   Compatible con MediaPipe 0.10.x                     ║
  ╚═══════════════════════════════════════════════════════╝

  Uso:
    python lse_proyecto.py recoger              → Graba muestras (salta letras con 80)
    python lse_proyecto.py recoger --mas        → Añade 80 muestras más a TODAS las letras
    python lse_proyecto.py recoger --letra A    → Graba solo la letra A (con o sin --mas)
    python lse_proyecto.py entrenar             → Entrena el modelo de IA
    python lse_proyecto.py usar                 → Usa la app en tiempo real

  Ejemplos para mejorar letras concretas:
    python lse_proyecto.py recoger --letra B --mas   → añade muestras a la B
    python lse_proyecto.py recoger --letra N         → graba la N desde cero

  Orden recomendado:
    1. recoger    (~15-20 minutos)
    2. entrenar   (~10 segundos)
    3. usar       ¡a disfrutar!

  Instalación:
    pip install mediapipe opencv-python scikit-learn pyttsx3

  Nota: la primera vez que ejecutes 'recoger' o 'usar',
  el script descargará automáticamente el modelo de MediaPipe (~4 MB).
""")

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        mostrar_ayuda()
    elif args[0] == 'recoger':
        modo_mas   = '--mas'    in args
        solo_letra = None
        if '--letra' in args:
            idx = args.index('--letra')
            if idx + 1 < len(args):
                solo_letra = args[idx + 1]
            else:
                print("  ERROR: --letra necesita un argumento (ej: --letra A)")
                sys.exit(1)
        recoger_datos(modo_mas=modo_mas, solo_letra=solo_letra)
    elif args[0] == 'entrenar':
        entrenar_modelo()
    elif args[0] == 'usar':
        usar_app()
    else:
        print(f"\n  Comando desconocido: '{args[0]}'")
        mostrar_ayuda()
