"""Thin Streamlit WebUI wrapper for the video synopsis pipeline."""

import os
import logging
from datetime import datetime

import streamlit as st
import cv2
from PIL import Image

from video_synopsis.config import Config
from video_synopsis.pipeline import Pipeline

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Page config
PAGE_CONFIG = {
    "page_title": "Video Synopsis",
    "layout": "wide",
    "initial_sidebar_state": "auto",
}

try:
    icon_path = os.path.join(os.path.dirname(__file__), "..", "static", "vs_clip.png")
    if os.path.exists(icon_path):
        PAGE_CONFIG["page_icon"] = Image.open(icon_path)
except Exception:
    pass

st.set_page_config(**PAGE_CONFIG)


def handle_file_upload(uploaded_file):
    if uploaded_file:
        filename = os.path.abspath(uploaded_file.name)
        with open(filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return filename
    return None


with st.form("input_form"):
    st.title("Video Synopsis Configuration")

    uploaded_file = st.file_uploader("Upload Video:", type=["mp4", "avi"])
    video_path = handle_file_upload(uploaded_file) if uploaded_file else None

    if video_path:
        _, left_col, _ = st.columns([1, 8, 1])
        with left_col:
            st.subheader("Original Video")
            st.video(video_path, format="video/mp4", start_time=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        batch_size = st.number_input("Batch Size:", value=8, format="%d")
        segmenter = st.selectbox("Segmenter:", ["rfdetr", "people", "fastsam"])
        input_model = st.text_input("Model:", value="Unet_2020-07-20")
    with col2:
        tracker = st.selectbox("Tracker:", ["botsort", "sort", "sam3"])
        optimizer = st.selectbox("Optimizer:", ["mcts", "energy", "pso"])
        epochs = st.number_input("Epochs:", value=2000, min_value=1, format="%d")
    with col3:
        collision_method = st.selectbox("Collision Method:", ["repulsion", "iou"])
        energy_opt = st.checkbox("Enable Optimization:", value=True)
        use_npz = st.checkbox("Use NPZ Storage:", value=True)

    submitted = st.form_submit_button("Run")

if submitted and video_path:
    config = Config(
        video=video_path,
        batch_size=batch_size,
        segmenter=segmenter,
        input_model=input_model,
        tracker=tracker,
        optimizer=optimizer,
        epochs=epochs,
        collision_method=collision_method,
        energy_optimization=energy_opt,
        use_npz=use_npz,
    )

    with st.spinner("Running Video Synopsis pipeline..."):
        pipeline = Pipeline(config)
        output_video = pipeline.run()

    if output_video and os.path.exists(output_video):
        # Re-encode with ffmpeg for browser compatibility
        final_path = output_video.replace(".mp4", "_final.mp4")
        os.system(
            f'ffmpeg -loglevel error -i "{output_video}" '
            f'-vcodec libx264 -crf 23 -preset fast "{final_path}"'
        )
        if os.path.exists(final_path):
            os.remove(output_video)
            st.video(final_path)
        else:
            st.video(output_video)
    else:
        st.error("Pipeline did not produce output.")
elif submitted:
    st.error("Please upload a video file to proceed.")
