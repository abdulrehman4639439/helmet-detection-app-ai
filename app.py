
import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="⛑️"
)

st.title("⛑️ Helmet Detection System")
st.write("Upload an image to detect Helmet, Head and Person.")

MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

confidence = st.slider(
    "Confidence Level",
    0.1, 1.0, 0.25
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Original Image")
    st.image(image, use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting objects..."):
            results = model.predict(
                source=image,
                conf=confidence
            )

            result = results[0]
            annotated_image = result.plot()[:, :, ::-1]

            st.subheader("Detection Result")
            st.image(
                annotated_image,
                use_container_width=True
            )

            if result.boxes is not None and len(result.boxes) > 0:
                st.success(
                    f"{len(result.boxes)} object(s) detected!"
                )
            else:
                st.warning("No objects detected.")
