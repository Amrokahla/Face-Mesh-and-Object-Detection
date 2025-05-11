import streamlit as st
import os
import numpy as np
import cv2
from tempfile import NamedTemporaryFile
from modules.face_mesh import FaceMeshDetector
from modules.object_detector import ObjectDetector
from modules.utils import COCO_OBJECTS

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

# Initialize models
face_detector = FaceMeshDetector(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=face_mesh_confidence,
    min_tracking_confidence=face_mesh_confidence
)

model_path = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")
object_detector = None
if os.path.exists(model_path):
    object_detector = ObjectDetector(
        model_path=model_path,
        score_threshold=object_confidence
    )
else:
    st.sidebar.error(f"Model file not found: {model_path}")

# Main UI elements
st.header("Upload Video for Detection")
video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    # Display upload progress
    upload_progress = st.progress(0)
    upload_progress.progress(50)  # Simulate instant upload progress (may vary in an actual environment)

    temp_video_path = NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}").name
    with open(temp_video_path, "wb") as temp_video:
        temp_video.write(video_file.read())

    upload_progress.progress(100)  # Completion of upload

    st.success("Video has been uploaded!")

    # Process the uploaded video
    processing_progress = st.progress(0)
    cap = cv2.VideoCapture(temp_video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_output_file = NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_output_file.close()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_file.name, fourcc, fps, (width, height))

    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        processed_frame = frame.copy()

        if show_face_mesh:
            landmarks = face_detector.get_face_landmarks(processed_frame)
            if landmarks:
                processed_frame = face_detector.draw_face_landmarks(processed_frame, landmarks)

        if show_object_detection:
            detection_result = object_detector.detect_objects(processed_frame)
            processed_frame, _ = object_detector.draw_detection_result(
                processed_frame,
                detection_result,
                objects_to_detect=objects_to_detect if objects_to_detect else None
            )

        # Write the frame to output video
        out.write(processed_frame)

        processed_frames += 1
        processing_progress.progress(int(processed_frames / total_frames * 100))

    # Cleanup VideoCapture and VideoWriter
    cap.release()
    out.release()

    processing_progress.progress(100)  # Completion of processing
    st.success("Processing complete!")

    # Provide download link
    with open(temp_output_file.name, "rb") as file:
        st.download_button(
            label="Download Processed Video",
            data=file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )