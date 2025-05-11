# List of object classes in the COCO dataset
COCO_OBJECTS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# Used for object detection task
COMMON_OBJECTS = [
    'person', 'car', 'cell phone', 'laptop', 'chair', 'cup', 
    'bottle', 'tv', 'book', 'dog', 'cat', 'keyboard', 'mouse'
]

def get_category_color(category_name):
    """
    Generate a consistent color for a given object category.
    
    Args:
        category_name: Name of the object category
        
    Returns:
        BGR color tuple for the category
    """
    hash_value = sum(ord(c) for c in category_name) % 255
    if hash_value < 85:
        return (hash_value * 3, 255, 255 - hash_value * 3)
    elif hash_value < 170:
        return (255 - (hash_value - 85) * 3, 255, (hash_value - 85) * 3)
    else:
        return (255, 255 - (hash_value - 170) * 3, 255)

def create_markdown_table(data_dict):
    """
    Create a markdown table from a dictionary.
    
    Args:
        data_dict: Dictionary where keys are column 1 and values are column 2
        
    Returns:
        Markdown formatted table string
    """
    if not data_dict:
        return ""
    
    markdown = "| Category | Count |\n| --- | --- |\n"
    for key, value in data_dict.items():
        markdown += f"| {key} | {value} |\n"
    
    return markdown
