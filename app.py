import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import tensorflow as tf
import matplotlib.pyplot as plt

# Cargar el clasificador Haar para detección de personas
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

st.set_page_config(page_title="Clasificador en vivo", page_icon="🎥", layout="wide")
st.title("🎥 Clasificación en vivo con Keras + Streamlit")
st.caption("Cámara dentro de la página y resultados en la misma interfaz. Incluye selector de cámara/calidad y registro a CSV.")

MODEL_PATH = "keras_Model.h5"
LABELS_PATH = "labels.txt"

@st.cache_resource
def load_model_cached(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)

@st.cache_data
def load_labels(labels_path: str):
    with open(labels_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# Cargar recursos
try:
    model = load_model_cached(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"No se pudo cargar el modelo/etiquetas: {e}")
    st.stop()

# --- Sidebar: opciones de cámara y logging ---
st.sidebar.header("Ajustes de cámara")
facing = st.sidebar.selectbox(
    "Tipo de cámara (facingMode)",
    options=["auto (por defecto)", "user (frontal)", "environment (trasera)"],
    index=0
)
quality = st.sidebar.selectbox(
    "Calidad de video",
    options=["640x480", "1280x720", "1920x1080"],
    index=1
)
w, h = map(int, quality.split("x"))

# Media constraints para WebRTC
video_constraints: dict = {"width": w, "height": h}
if facing != "auto (por defecto)":
    video_constraints["facingMode"] = facing.split(" ")[0]
media_constraints = {"video": video_constraints, "audio": False}

st.sidebar.header("Registro de predicciones")
enable_log = st.sidebar.checkbox("Habilitar registro (CSV)", value=True)
log_every_n_seconds = st.sidebar.slider("Intervalo de registro (s)", 0.2, 5.0, 1.0, 0.2)

if "pred_log" not in st.session_state:
    st.session_state.pred_log = pd.DataFrame(columns=["timestamp", "label", "confidence"])
if "last_log_ts" not in st.session_state:
    st.session_state.last_log_ts = 0.0

# Configuración STUN para WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self) -> None:
        self.latest = {"class": None, "confidence": 0.0}
        self.model = model
        self.labels = labels

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Detección de personas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        persons = person_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Dibujar cuadros alrededor de las personas detectadas
        for (x, y, w, h) in persons:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Preparación a 224x224 y normalización
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0

        # Predicción
        pred = self.model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = self.labels[idx] if idx < len(self.labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        self.latest = {"class": label, "confidence": conf}

        # Overlay
        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
        return overlay

# Layout
left, right = st.columns([2, 1], gap="large")
with left:
    st.subheader("Cámara en vivo")
    webrtc_ctx = webrtc_streamer(
        key="keras-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_constraints,
        video_transformer_factory=VideoTransformer,
        async_processing=True,
    )
    st.info(
        "Si no ves tu cámara, concede permisos del navegador o prueba con otro (Chrome recomendado). "
        "En móviles, 'user' = frontal y 'environment' = trasera (si el dispositivo lo soporta).",
        icon="ℹ️",
    )

with right:
    st.subheader("Resultados")
    result_placeholder = st.empty()
    progress_placeholder = st.empty()

    if enable_log and not st.session_state.pred_log.empty:
        if st.button("🧹 Limpiar registro"):
            st.session_state.pred_log = st.session_state.pred_log.iloc[0:0]
            st.session_state.last_log_ts = 0.0

    csv_bytes = st.session_state.pred_log.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar CSV de predicciones",
        data=csv_bytes,
        file_name=f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        disabled=st.session_state.pred_log.empty,
    )

# Actualización periódica de resultados y logging
if webrtc_ctx and webrtc_ctx.state.playing:
    for _ in range(300000):
        if not webrtc_ctx.state.playing:
            break
        vt = webrtc_ctx.video_transformer
        if vt is not None and vt.latest["class"] is not None:
            cls = vt.latest["class"]
            conf = vt.latest["confidence"]
            result_placeholder.markdown(f"**Clase detectada:** {cls}\n\n**Confianza:** {conf*100:.2f}%")
            progress_placeholder.progress(min(max(conf, 0.0), 1.0))
            if enable_log:
                now = time.time()
                if now - st.session_state.last_log_ts >= log_every_n_seconds:
                    st.session_state.pred_log.loc[len(st.session_state.pred_log)] = [
                        datetime.utcnow().isoformat(),
                        cls,
                        round(conf, 6),
                    ]
                    st.session_state.last_log_ts = now
        time.sleep(0.2)
else:
    st.write("Activa la cámara para ver aquí las predicciones.")

st.markdown("---")
with st.expander("⚠️ Modo alternativo (captura por foto, sin WebRTC)"):
    st.write("Si tu red bloquea WebRTC, usa una foto para predecir de forma puntual.")
    snap = st.camera_input("Captura una imagen")
    if snap is not None:
        file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = resized.astype(np.float32).reshape(1, 224, 224, 3)
        x = (x / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx] if idx < len(labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        st.image(img, caption=f"{label} | {conf*100:.2f}%")
        st.success(f"Predicción: **{label}** ({conf*100:.2f}%)")

# ------------------ GRAFICAS ------------------
st.markdown("## 📊 Estadísticas de predicciones")

if not st.session_state.pred_log.empty:
    df = st.session_state.pred_log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1️⃣ Conteo de cada clase detectada
    fig1, ax1 = plt.subplots()
    df["label"].value_counts().plot(kind="bar", ax=ax1, color="skyblue")
    ax1.set_title("Cantidad de predicciones por clase")
    ax1.set_ylabel("Cantidad")
    st.pyplot(fig1)

    # 2️⃣ Confianza promedio por clase
    fig2, ax2 = plt.subplots()
    df.groupby("label")["confidence"].mean().plot(kind="bar", ax=ax2, color="orange")
    ax2.set_title("Confianza promedio por clase")
    ax2.set_ylabel("Confianza")
    st.pyplot(fig2)

    # 3️⃣ Evolución temporal de predicciones
    fig3, ax3 = plt.subplots()
    df.set_index("timestamp")["confidence"].plot(ax=ax3, marker="o", linestyle="-")
    ax3.set_title("Confianza en el tiempo")
    ax3.set_ylabel("Confianza")
    ax3.set_xlabel("Tiempo")
    st.pyplot(fig3)

    # 4️⃣ Distribución de confianza
    fig4, ax4 = plt.subplots()
    df["confidence"].plot(kind="hist", bins=20, ax=ax4, color="green")
    ax4.set_title("Distribución de confianza")
    ax4.set_xlabel("Confianza")
    st.pyplot(fig4)

    # 5️⃣ Últimas 10 predicciones
    fig5, ax5 = plt.subplots()
    df.tail(10).set_index("timestamp")["label"].value_counts().plot(
        kind="bar", ax=ax5, color="purple"
    )
    ax5.set_title("Últimas 10 predicciones por clase")
    ax5.set_ylabel("Cantidad")
    st.pyplot(fig5)
else:
    st.info("No hay datos para mostrar gráficas aún. Activa la cámara y registra predicciones.")
