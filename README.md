# 🤟 SignIA — Lengua de Signos con Inteligencia Artificial

> Convierte la **Lengua de Signos Española (LSE)** en voz en tiempo real usando visión artificial y Machine Learning. Solo necesitas un ordenador y una webcam.

![Python](https://img.shields.io/badge/Python-3.8--3.12-blue?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey)
![Offline](https://img.shields.io/badge/Offline-100%25-brightgreen)

---

## ¿Qué es SignIA?

SignIA es una aplicación de escritorio que usa la **cámara del ordenador** para leer el alfabeto dactilológico de la Lengua de Signos Española y convertirlo en voz en tiempo real.

El sistema funciona en tres pasos:

```
📷 Cámara  →  🖐 21 landmarks  →  📊 42 números  →  🤖 Random Forest  →  🔊 Voz
```

1. **La cámara captura** 30 fotogramas por segundo
2. **MediaPipe** (Google) detecta 21 puntos clave de la mano
3. Los puntos se **normalizan** en un vector de 42 números
4. Un **Random Forest** clasifica la letra con >90% de precisión
5. Las letras forman palabras que se **pronuncian en voz alta**

Todo funciona **100% offline**, sin enviar datos a ningún servidor.

---

## ✨ Características

- 🎯 **Alta precisión** — más del 90% con las 26 letras del alfabeto
- ⚡ **Tiempo real** — respuesta en menos de 1 segundo
- 🌐 **Sin internet** — funciona completamente offline
- 🖥️ **Multiplataforma** — Windows, macOS y Linux
- 🎨 **Interfaz visual** — colores por dedo, barra de confianza, marco de encuadre
- 🔄 **Modelo mejorable** — añade más muestras en cualquier momento
- 🏫 **Orientado a educación** — proyecto de 3º ESO sobre IA y accesibilidad

---

## 🚀 Instalación

### Requisitos previos

- Python **3.8 – 3.12** (⚠️ MediaPipe no soporta 3.13 ni superior)
- Webcam

### 1. Clona el repositorio

```bash
git clone https://github.com/tu-usuario/signia.git
cd signia
```

### 2. Instala las dependencias

**Linux / Windows:**
```bash
pip install mediapipe opencv-python scikit-learn pyttsx3
```

**macOS con Homebrew (Python 3.11 recomendado):**
```bash
brew install python@3.11
pip3.11 install mediapipe opencv-python scikit-learn pyttsx3 --break-system-packages
```

> **Nota:** La primera vez que ejecutes la app se descargará automáticamente el modelo de MediaPipe (~4 MB). Necesitas conexión a internet solo ese primer momento.

---

## 📖 Uso

El proyecto tiene **tres fases** que se ejecutan en orden:

### Fase 1 — Recoger datos de entrenamiento

```bash
python lse_proyecto.py recoger
```

- Se abre la cámara con la letra a grabar en pantalla
- Pulsa **ESPACIO** para empezar a grabar esa letra
- El programa captura **80 fotogramas** automáticamente
- Repite para las 26 letras (~15-20 minutos en total)
- Los datos se guardan en `lse_datos.json`

**Opciones avanzadas:**
```bash
# Añadir más muestras a todas las letras
python lse_proyecto.py recoger --mas

# Grabar o mejorar solo una letra concreta
python lse_proyecto.py recoger --letra A
python lse_proyecto.py recoger --letra B --mas
```

### Fase 2 — Entrenar el modelo

```bash
python lse_proyecto.py entrenar
```

- Lee todos los datos de `lse_datos.json`
- Entrena un **Random Forest** con 200 árboles de decisión
- Muestra la precisión global y un informe por letra
- Guarda el modelo en `lse_modelo.pkl`

Tiempo de entrenamiento: **~10 segundos** con las 26 letras.

### Fase 3 — Usar la app en tiempo real

```bash
python lse_proyecto.py usar
# En macOS con Python 3.11:
python3.11 lse_proyecto.py usar
```

- Se abre la ventana con la imagen de la cámara en espejo
- Coloca la mano dentro del **marco verde** de encuadre
- La **letra detectada** aparece en grande (esquina superior derecha)
- La **barra de confianza** indica la seguridad del modelo
- Las letras confirmadas se acumulan formando una **palabra**
- Al retirar la mano **2 segundos**, la app pronuncia la palabra en voz alta

**Controles:**

| Tecla | Acción |
|-------|--------|
| `Q` | Cerrar la aplicación |
| `Espacio` | Borrar la palabra en construcción |
| Pausa 2 seg | Pronunciar la palabra acumulada |

---

## 📁 Estructura de ficheros

```
signia/
├── lse_proyecto.py        ← Script principal
├── hand_landmarker.task   ← Modelo MediaPipe (se descarga automáticamente)
├── lse_datos.json         ← Datos de entrenamiento (se genera al recoger)
├── lse_modelo.pkl         ← Modelo entrenado (se genera al entrenar)
└── README.md
```

---

## 🛠️ Tecnología

| Librería | Versión | Función |
|----------|---------|---------|
| [MediaPipe](https://mediapipe.dev) | 0.10.x | Detección de 21 landmarks de la mano |
| [OpenCV](https://opencv.org) | 4.x | Captura y procesado de vídeo |
| [scikit-learn](https://scikit-learn.org) | 1.x | Algoritmo Random Forest |
| [pyttsx3](https://pyttsx3.readthedocs.io) | 2.x | Síntesis de voz (Linux/Windows) |
| `say` (macOS) | — | Síntesis de voz nativa en macOS |

### ¿Por qué Random Forest y no una red neuronal?

Con ~2.080 ejemplos (80 por letra × 26 letras) y sin GPU, Random Forest:
- Entrena en **~10 segundos**
- Alcanza **>90% de precisión**
- No requiere hardware especializado

Una CNN necesitaría miles de imágenes más y tiempo de entrenamiento muy superior.

---

## 🌍 Impacto social — ODS

Este proyecto contribuye a los Objetivos de Desarrollo Sostenible de la ONU:

| ODS | Relación |
|-----|----------|
| **ODS 10** — Reducción de desigualdades | Elimina la barrera entre personas sordas y oyentes sin intérprete |
| **ODS 4** — Educación de calidad | Participación autónoma de estudiantes con discapacidad auditiva |
| **ODS 3** — Salud y bienestar | Mejora la comunicación en entornos médicos |
| **ODS 9** — Industria e innovación | IA accesible que resuelve problemas sociales reales |

---

## ⚠️ Limitaciones conocidas

- Solo reconoce **letras estáticas** del alfabeto dactilológico LSE. Las letras con movimiento (J, Ñ, RR) no están soportadas.
- La precisión depende de la **iluminación** — evitar sombras fuertes sobre la mano.
- El modelo es **personal** — funciona mejor con manos similares a las usadas en el entrenamiento. Cualquier persona puede añadir sus propias muestras con `recoger`.

---

## 🤝 Contribuir

1. Haz fork del repositorio
2. Crea una rama: `git checkout -b mejora/nombre-de-la-mejora`
3. Haz commit de tus cambios: `git commit -m 'Añade soporte para letras dinámicas'`
4. Push a la rama: `git push origin mejora/nombre-de-la-mejora`
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la licencia **MIT**. Consulta el fichero [LICENSE](LICENSE) para más detalles.

---

## 👨‍🏫 Créditos

Desarrollado como proyecto de **Tecnología e Informática — 3º ESO**  
Exposición de Robótica · ALCOIBOT 2026  
Colegio Calasancio Alicante

---

<div align="center">
  <sub>Hecho con 🤟 para reducir barreras de comunicación</sub>
</div>
