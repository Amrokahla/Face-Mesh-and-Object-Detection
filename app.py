import streamlit as st
import os
import numpy as np
from modules.face_mesh import FaceMeshDetector
from modules.object_detector import ObjectDetector
from modules.utils import COCO_OBJECTS
import cv2

# Set page config
st.set_page_config(
    page_title="Face & Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Page title and description
st.title("Face Mesh & Object Detection")
st.markdown("""
This application uses MediaPipe to perform:
1. 💡 **Face Mesh Detection**: Identifies facial landmarks
2. 🔍 **Object Detection**: Recognizes common objects
""")

# Sidebar configuration
st.sidebar.title("Configuration")

# Face mesh options
st.sidebar.header("Face Mesh Settings")
show_face_mesh = st.sidebar.checkbox("Enable Face Mesh", value=True)
face_mesh_confidence = st.sidebar.slider("Face Mesh Confidence", 0.1, 1.0, 0.5, 0.1)

# Object detection options
st.sidebar.header("Object Detection Settings")
show_object_detection = st.sidebar.checkbox("Enable Object Detection", value=True)
object_confidence = st.sidebar.slider("Object Detection Confidence", 0.1, 1.0, 0.5, 0.1)

# Object filtering
st.sidebar.header("Object Filtering")
objects_to_detect = st.sidebar.multiselect(
    "Select objects to detect",
    options=COCO_OBJECTS,
    default=["person", "cell phone", "laptop"]
)

# Load models only once
if "face_detector" not in st.session_state:
    st.session_state.face_detector = FaceMeshDetector(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=face_mesh_confidence,
        min_tracking_confidence=face_mesh_confidence
    )

if "object_detector" not in st.session_state:
    model_path = os.path.join(os.path.dirname(__file__), "model", "efficientdet_lite0.tflite")
    st.session_state.object_detector = ObjectDetector(
        model_path=model_path,
        score_threshold=object_confidence
    )

# Function to process video
def process_video(video_file):
    temp_video_path = f"temp_video.{video_file.name.split('.')[-1]}"
    with open(temp_video_path, "wb") as temp_video:
        temp_video.write(video_file.read())

    cap = cv2.VideoCapture(temp_video_path)
    if not cap.isOpened():
        st.error("Could not open video file.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    out_video_path = "processed_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_video_path, fourcc, fps / 5, (640, 360))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 360))

        if show_face_mesh:
            landmarks = st.session_state.face_detector.get_face_landmarks(frame)
            if landmarks:
                frame = st.session_state.face_detector.draw_face_landmarks(frame, landmarks)

        if show_object_detection:
            detection_result = st.session_state.object_detector.detect_objects(frame)
            frame, _ = st.session_state.object_detector.draw_detection_result(
                frame, detection_result, objects_to_detect if objects_to_detect else None
            )

        out.write(frame)

    out.release()
    cap.release()
    os.remove(temp_video_path)

    return out_video_path

# Main content
st.header("Upload and Process Video")
video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    if "processed_video_path" not in st.session_state:
        st.session_state.processed_video_path = process_video(video_file)

    st.success("Video processing completed successfully!")
    st.download_button(
        label="Download the processed video",
        data=open(st.session_state.processed_video_path, "rb").read(),
        file_name="processed_video.mp4",
        mime="video/mp4"
    )
