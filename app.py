import streamlit as st
import cv2
import os
import numpy as np
from modules.face_mesh import FaceMeshDetector
from modules.object_detector import ObjectDetector
from modules.utils import COCO_OBJECTS

# Set page config
st.set_page_config(
    page_title="Live Face & Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Page title and description
st.title("Live Face Mesh & Object Detection")
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

model_path = './models/efficientdet_lite0.tflite'
object_detector = None
if os.path.exists(model_path):
    object_detector = ObjectDetector(
        model_path=model_path,
        score_threshold=object_confidence
    )
else:
    st.sidebar.error(f"Model file not found: {model_path}")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    st.header("Live Detection from Camera")
    run_camera = st.checkbox('Activate Camera')

    # Placeholder for video display
    video_placeholder = st.empty()

if run_camera:
    cap = cv2.VideoCapture(0)  # Open the first available camera

    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        while run_camera:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            # Resize frame for faster processing
            frame = cv2.resize(frame, (640, 360))

            # Process the frame
            processed_frame = frame.copy()

            face_count = 0
            frame_objects = {}

            # Face mesh detection
            if show_face_mesh:
                landmarks = face_detector.get_face_landmarks(processed_frame)
                if landmarks:
                    processed_frame = face_detector.draw_face_landmarks(processed_frame, landmarks)
                    face_count = 1

            # Object detection
            if show_object_detection:
                detection_result = object_detector.detect_objects(processed_frame)
                processed_frame, detected_objects = object_detector.draw_detection_result(
                    processed_frame,
                    detection_result,
                    objects_to_detect=objects_to_detect if objects_to_detect else None
                )

                # Update object counts
                frame_objects = detected_objects
                if frame_objects:
                    print(frame_objects)  # For debugging purposes

            # Convert BGR to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

            # Exit by checking a key press, not needed for Streamlit (browser-based)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()

st.markdown("---")
st.caption("Powered by MediaPipe, OpenCV and Streamlit")