import mediapipe as mp
import cv2

class FaceMeshDetector:
    """Face mesh detector class using MediaPipe."""
    
    def __init__(
        self,
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ):
        """
        Initialize the face mesh detector.
        
        Args:
            static_image_mode: Whether to treat input images as static (not video)
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around the eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for face tracking
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def get_face_landmarks(self, image):
        """
        Detect face landmarks in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Face landmarks or None if no face detected
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        return results.multi_face_landmarks if results.multi_face_landmarks else []
    
    def draw_face_landmarks(self, image, landmarks, color=(0, 255, 0), radius=1):
        """
        Draw face landmarks on the image.
        
        Args:
            image: Input image to draw on
            landmarks: Face landmarks from MediaPipe
            color: Color of landmarks (BGR format)
            radius: Radius of landmark points
            
        Returns:
            Image with landmarks drawn
        """
        for lm in landmarks.landmark:
            x = int(lm.x * image.shape[1])
            y = int(lm.y * image.shape[0])
            cv2.circle(image, (x, y), radius, color, -1)
        return image
