import streamlit as st
import os
import cv2
from tempfile import NamedTemporaryFile
from modules.face_mesh import FaceMeshDetector
from modules.object_detector import ObjectDetector
from modules.utils import COCO_OBJECTS

# Set page config
st.set_page_config(
    page_title="Face & Object Detection",
    page_icon="🔍",
    layout="wide",
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

# Initialize models safely
face_detector = None
object_detector = None

# Load face mesh model
try:
    face_detector = FaceMeshDetector(
        static_image_mode=False,
        max_num_faces=1,
        min_detection_confidence=face_mesh_confidence,
        min_tracking_confidence=face_mesh_confidence,
    )
except Exception as e:
    st.sidebar.error(f"Error loading Face Mesh model: {e}")

# Load object detection model
model_path = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")
if not os.path.exists(model_path):
    st.sidebar.error(f"Object detection model file not found: {model_path}")
else:
    try:
        object_detector = ObjectDetector(
            model_path=model_path,
            score_threshold=object_confidence,
        )
    except Exception as e:
        st.sidebar.error(f"Error loading Object Detector model: {e}")

# Main UI
st.header("Upload Video for Detection")
video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

if video_file:
    # Upload progress (simulate - in real usage Streamlit handles upload)
    upload_progress = st.progress(0)
    upload_progress.progress(50)

    temp_video_path = NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}").name
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    upload_progress.progress(100)
    st.success("Video uploaded successfully!")

    # Open the video
    cap = cv2.VideoCapture(temp_video_path)

    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Temporary output video file
        temp_output_file = NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output_file.close()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_file.name, fourcc, fps, (width, height))

        processing_progress = st.progress(0)
        video_placeholder = st.empty()
        stats_placeholder = st.empty()

        processed_frames = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = frame.copy()

            # Face mesh detection if model loaded and enabled
            if show_face_mesh and face_detector is not None:
                landmarks = face_detector.get_face_landmarks(processed_frame)
                if landmarks:
                    processed_frame = face_detector.draw_face_landmarks(processed_frame, landmarks)

            # Object detection if model loaded and enabled
            detected_objects = {}
            if show_object_detection and object_detector is not None:
                detection_result = object_detector.detect_objects(processed_frame)
                processed_frame, detected_objects = object_detector.draw_detection_result(
                    processed_frame,
                    detection_result,
                    objects_to_detect=objects_to_detect if objects_to_detect else None,
                )

            # Write processed frame to output video
            out.write(processed_frame)

            # Update display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)

            # Update stats text
            stats_text = f"""
            ## Current Frame: {processed_frames +1}/{frame_count}
            
            ### Detected Objects:
            """
            if detected_objects:
                for obj, count in detected_objects.items():
                    stats_text += f"- **{obj}**: {count}\n"
            else:
                stats_text += "No objects detected in this frame."

            stats_placeholder.markdown(stats_text)

            processed_frames += 1
            processing_progress.progress(int(processed_frames / frame_count * 100))

        cap.release()
        out.release()
        processing_progress.progress(100)
        st.success("Video processing complete!")

        # Provide download button
        with open(temp_output_file.name, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4",
            )

        # Optionally cleanup temp files if desired
        os.remove(temp_video_path)
        # Do NOT remove temp_output_file immediately if user wants to download (remove later if you want)