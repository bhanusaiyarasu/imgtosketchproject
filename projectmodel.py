import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import onnxruntime as ort
import io

# --- Load AI Models ---
STYLE_MODEL = "la_muse.t7"
SKETCH_MODEL = "sketch.onnx"

if not os.path.exists(STYLE_MODEL):
    st.error(f"Missing AI style model: {STYLE_MODEL}")
    st.stop()

if not os.path.exists(SKETCH_MODEL):
    st.error(f"Missing AI sketch model: {SKETCH_MODEL}")
    st.stop()

style_net = cv2.dnn.readNetFromTorch(STYLE_MODEL)
onnx_session = ort.InferenceSession(SKETCH_MODEL)

# --- Functions ---
def opencv_sketch(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    blur = cv2.GaussianBlur(inv, (21, 21), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)
    return sketch

def ai_style_sketch(image):
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
                                 (103.939, 116.779, 123.680),
                                 swapRB=False, crop=False)
    style_net.setInput(blob)
    output = style_net.forward()
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output = output.transpose(1, 2, 0)
    return np.clip(output, 0, 255).astype("uint8")

def ai_sketch_onnx(image):
    resized = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    blob = np.transpose(resized, (2, 0, 1))[None, :, :, :]
    ort_inputs = {onnx_session.get_inputs()[0].name: blob}
    out = onnx_session.run(None, ort_inputs)[0][0, 0]
    sketch = (out * 255).astype("uint8")
    sketch = cv2.resize(sketch, (image.shape[1], image.shape[0]))
    sketch = cv2.GaussianBlur(sketch, (3, 3), 0)
    return sketch

def convert_to_bytes(image, mode="GRAY"):
    """Convert numpy image to PNG bytes for download"""
    if mode == "GRAY":
        pil_img = Image.fromarray(image)
    else:
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()

# --- Page Setup ---
st.set_page_config(page_title="🎨 AI Image to Sketch Demo", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .main { background-color: #f9f9f9; }
    h1 { color: #4a90e2; text-align: center; font-family: 'Trebuchet MS', sans-serif; }
    .footer { text-align: center; font-size: 13px; color: #999; }
    </style>
""", unsafe_allow_html=True)

st.title("🎨 AI Image to Sketch Demo")
st.write("Turn your images or webcam feed into beautiful **pencil sketches** or **artistic styles** ✨")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
source = st.sidebar.radio("📸 Input Source", ["Upload Image", "Capture Once", "Go Live Webcam"])
style = st.sidebar.radio("🎭 Sketch Type", ["OpenCV Sketch", "AI Sketch (ONNX)", "AI Stylized (La Muse)"])

# --- Processing Function ---
def process(image):
    if style == "OpenCV Sketch":
        return opencv_sketch(image), "OpenCV Pencil Sketch", "GRAY"
    elif style == "AI Sketch (ONNX)":
        return ai_sketch_onnx(image), "AI Sketch (ONNX)", "GRAY"
    else:
        return ai_style_sketch(image), "AI Stylized (La Muse)", "RGB"

# --- Main Flow ---
if source == "Upload Image":
    file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if file:
        image = np.array(Image.open(file).convert("RGB"))[:, :, ::-1]
        sketch, label, mode = process(image)

        tab1, tab2 = st.tabs(["🖼️ Original", "✏️ Sketch"])
        with tab1:
            st.image(image[:, :, ::-1], caption="Original Image", use_column_width=True)
        with tab2:
            st.image(sketch, caption=label, use_column_width=True, channels=mode)
            st.download_button(
                label="💾 Download Sketch",
                data=convert_to_bytes(sketch, mode),
                file_name="sketch.png",
                mime="image/png"
            )

elif source == "Capture Once":
    if st.button("📷 Capture from Webcam"):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            sketch, label, mode = process(frame)
            tab1, tab2 = st.tabs(["🖼️ Original", "✏️ Sketch"])
            with tab1:
                st.image(frame[:, :, ::-1], caption="Original Image", use_column_width=True)
            with tab2:
                st.image(sketch, caption=label, use_column_width=True, channels=mode)
                st.download_button(
                    label="💾 Download Sketch",
                    data=convert_to_bytes(sketch, mode),
                    file_name="sketch.png",
                    mime="image/png"
                )
        else:
            st.error("⚠️ Webcam error.")

elif source == "Go Live Webcam":
    st.write("🎥 Streaming live... click Stop when done.")
    frame_window = st.empty()
    cap = cv2.VideoCapture(0)
    stop = st.button("⏹️ Stop Live")
    while cap.isOpened() and not stop:
        ret, frame = cap.read()
        if not ret:
            break
        sketch, label, mode = process(frame)
        frame_window.image(sketch, caption=label, channels=mode)
    cap.release()

st.markdown('<p class="footer">🚀 Built with Streamlit + OpenCV + ONNX</p>', unsafe_allow_html=True)
