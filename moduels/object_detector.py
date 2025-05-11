import mediapipe as mp
import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class ObjectDetector:
    """Object detector class using MediaPipe."""
    
    def __init__(self, model_path, score_threshold=0.5):
        """
        Initialize the object detector.
        
        Args:
            model_path: Path to the TFLite model file
            score_threshold: Minimum confidence score for detection
        """
        # Load the model
        try:
            with open(model_path, "rb") as f:
                model_data = f.read()
                
            # Initialize the detector with the model buffer
            base_options = python.BaseOptions(model_asset_buffer=model_data)
            options = vision.ObjectDetectorOptions(
                base_options=base_options,
                score_threshold=score_threshold
            )
            self.detector = vision.ObjectDetector.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to load object detection model: {str(e)}")
    
    def detect_objects(self, image):
        """
        Detect objects in an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Detection results from MediaPipe
        """
        # Convert BGR to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Create MediaPipe image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        # Detect objects
        return self.detector.detect(mp_image)
    
    def draw_detection_result(self, image, detection_result, objects_to_detect=None):
        """
        Draw bounding boxes and labels for detected objects.
        
        Args:
            image: Input image to draw on
            detection_result: Object detection results from MediaPipe
            objects_to_detect: List of object classes to display. If None, display all objects.
                               Example: ['person', 'car', 'dog']
                               
        Returns:
            Tuple of:
            - Image with detections drawn
            - Dictionary of detected objects and their counts
        """
        if not detection_result.detections:
            return image, {}
        
        # Create a copy of the image to draw on
        annotated_image = image.copy()
        
        # Dictionary to track detected objects and their counts
        detected_objects = {}
        
        for detection in detection_result.detections:
            # Get category information
            category = detection.categories[0]
            category_name = category.category_name
            probability = round(category.score, 2)
            
            # Skip if not in allowed objects list
            if objects_to_detect and category_name.lower() not in [obj.lower() for obj in objects_to_detect]:
                continue
            
            # Update detected objects count
            if category_name in detected_objects:
                detected_objects[category_name] += 1
            else:
                detected_objects[category_name] = 1
            
            # Get bounding box coordinates
            bbox = detection.bounding_box
            x1 = bbox.origin_x
            y1 = bbox.origin_y
            x2 = x1 + bbox.width
            y2 = y1 + bbox.height
            
            # Draw bounding box
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            # Draw label
            label = f"{category_name}: {probability}"
            
            # Position the label text above the rectangle
            cv2.putText(
                annotated_image, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 255), 
                2
            )
        
        return annotated_image, detected_objects
