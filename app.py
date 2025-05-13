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
    default=["person", "cell phone", "car"]
)

st.sidebar.text("Loading Face Mesh Model...")
face_detector = FaceMeshDetector(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=face_mesh_confidence,
    min_tracking_confidence=face_mesh_confidence
)
st.sidebar.success("Face Mesh Model loaded successfully!")

st.sidebar.text("Checking Object Detection Model path...")
model_path = os.path.join(os.path.dirname(__file__), "model", "efficientdet_lite0.tflite")
if not os.path.exists(model_path):
    st.sidebar.error(f"Model file not found: {model_path}")
    st.sidebar.info("Please download the model file and place it in the models directory")
else:
    st.sidebar.text("Loading Object Detection Model...")
    object_detector = ObjectDetector(
        model_path=model_path,
        score_threshold=object_confidence
    )
    st.sidebar.success("Object Detection Model loaded successfully!")

# Main content
st.header("Upload and Process Video")
video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])
uploaded_video_message = st.empty()

# Initialize session state for video path
if 'processed_video_path' not in st.session_state:
    st.session_state.processed_video_path = None

# Video processing logic
if 'frame_counter' not in st.session_state:
    st.session_state.frame_counter = 0

if video_file:
    try:
        uploaded_video_message.success("Video uploaded successfully!")
        if st.session_state.processed_video_path is None:
            with st.spinner("Processing video..."):
                # Processing video only if not already processed
                temp_video_path = f"temp_video.{video_file.name.split('.')[-1]}"
                with open(temp_video_path, "wb") as temp_video:
                    temp_video.write(video_file.read())

                cap = cv2.VideoCapture(temp_video_path)

                if not cap.isOpened():
                    st.error("Could not open video file.")
                else:
                    face_count = 0
                    object_counts = {}
                    frame_skip = 1
                    
                    # Get original video's fps and size
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    out_video_path = "processed_video.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    out = cv2.VideoWriter(out_video_path, fourcc, fps/frame_skip, (width, height))

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        st.session_state.frame_counter += 1
                        if st.session_state.frame_counter % frame_skip != 0:
                            continue

                        # No need to resize the frame
                        processed_frame = frame.copy()

                        face_count = 0
                        frame_objects = {}

                        if show_face_mesh:
                            landmarks_list = face_detector.get_face_landmarks(processed_frame)
                            if landmarks_list:
                                for landmarks in landmarks_list:
                                    processed_frame = face_detector.draw_face_landmarks(processed_frame, landmarks)
                                face_count = len(landmarks_list)

                        if show_object_detection:
                            detection_result = object_detector.detect_objects(processed_frame)
                            processed_frame, detected_objects = object_detector.draw_detection_result(
                                processed_frame, 
                                detection_result, 
                                objects_to_detect=objects_to_detect if objects_to_detect else None
                            )
                            frame_objects = detected_objects
                            for obj in frame_objects:
                                if obj in object_counts:
                                    object_counts[obj] = max(object_counts[obj], frame_objects[obj])
                                else:
                                    object_counts[obj] = frame_objects[obj]

                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        out.write(processed_frame)

                    out.release()
                    cap.release()
                    os.remove(temp_video_path)
                    st.session_state.processed_video_path = out_video_path

        uploaded_video_message.empty()
        st.success("Video processing completed successfully!")
        st.markdown("## Download Processed Video")
        processed_video_file = open(st.session_state.processed_video_path, "rb").read()
        st.download_button(
            label="Download the processed video",
            data=processed_video_file,
            file_name="processed_video.mp4",
            mime="video/mp4"
        )

    except Exception as e:
        st.error(f"Error processing the video: {e}")