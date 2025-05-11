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

# Initializing the models with a spinner
st.sidebar.markdown("---")
try:
    with st.sidebar.spinner("Loading models..."):
        # Load Face Mesh Model
        face_detector = FaceMeshDetector(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=face_mesh_confidence,
            min_tracking_confidence=face_mesh_confidence
        )

        # Load Object Detection Model
        model_path = os.path.join(os.path.dirname(__file__), "models", "efficientdet_lite0.tflite")
        if not os.path.exists(model_path):
            st.sidebar.error(f"Model file not found: {model_path}")
            st.sidebar.info("Please download the model file and place it in the models directory")
        else:
            object_detector = ObjectDetector(
                model_path=model_path,
                score_threshold=object_confidence
            )
            st.sidebar.success("Models loaded successfully!")

except Exception as e:
    st.sidebar.error(f"Unexpected error during model initialization: {e}")

# Main content
col1, col2 = st.columns([3, 2])

with col1:
    # Video capture
    st.header("Live Detection")
    video_placeholder = st.empty()

    # Start/stop button with session state to handle toggling
    if 'run_video' not in st.session_state:
        st.session_state.run_video = False

    def toggle_video():
        st.session_state.run_video = not st.session_state.run_video

    st.button("Start/Stop Video", on_click=toggle_video)

with col2:
    # Display statistics and info
    st.header("Detection Information")
    stats_placeholder = st.empty()

    # Display the selected objects
    st.subheader("Selected Objects")
    if objects_to_detect:
        for obj in objects_to_detect:
            st.write(f"- {obj}")
    else:
        st.write("No objects selected for filtering")

    # Information about face mesh
    st.subheader("About Face Mesh")
    st.write("""
    The face mesh detector identifies 468 landmarks on a human face, 
    allowing for detailed facial analysis and tracking.
    """)

    # Credits
    st.markdown("---")
    st.caption("Powered by MediaPipe, OpenCV and Streamlit")

# Video processing
if st.session_state.run_video:
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open webcam. Please check your camera connection.")
    else:
        try:
            # Detection counters
            face_count = 0
            object_counts = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture video frame")
                    break

                # Process the frame
                processed_frame = frame.copy()

                # Reset counters for this frame
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
                    for obj in frame_objects:
                        if obj in object_counts:
                            object_counts[obj] = max(object_counts[obj], frame_objects[obj])
                        else:
                            object_counts[obj] = frame_objects[obj]

                # Convert BGR to RGB for display
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # Display the processed frame
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_column_width=True)

                # Update stats
                stats_text = f"""
                ## Current Frame Stats
                - **Faces Detected:** {face_count}
                
                ## Objects Detected
                """
                for obj, count in frame_objects.items():
                    stats_text += f"- **{obj}:** {count}\n"

                stats_placeholder.markdown(stats_text)

                # Break the loop if toggle is switched by button
                if not st.session_state.run_video:
                    break

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            # Release the webcam
            cap.release()
            cv2.destroyAllWindows()