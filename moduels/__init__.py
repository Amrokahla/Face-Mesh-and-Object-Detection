# Make modules directory a package
from modules.face_mesh import FaceMeshDetector
from modules.object_detector import ObjectDetector
from modules.utils import COCO_OBJECTS, COMMON_OBJECTS

__all__ = ['FaceMeshDetector', 'ObjectDetector', 'COCO_OBJECTS', 'COMMON_OBJECTS']
