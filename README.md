# Face Mesh & Object Detection

A Streamlit web application that performs real-time face mesh and object detection using MediaPipe and OpenCV.

## Features

- 💡 **Face Mesh Detection**: Identifies and visualizes 468 facial landmarks
- 🔍 **Object Detection**: Recognizes 90 common object categories
- 🎛️ **Customizable Settings**: Adjust confidence thresholds and filter objects
- 📊 **Live Statistics**: View real-time detection counts

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/face-object-detection.git
cd face-object-detection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download the model file:
   - Download the [EfficientDet Lite0 model](https://storage.googleapis.com/mediapipe-assets/efficientdet_lite0.tflite)
   - Place it in the `models/` directory

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. The web interface will open in your browser. Configure settings in the sidebar:
   - Adjust face mesh detection confidence
   - Adjust object detection confidence
   - Select which objects to detect

3. Click the "Start/Stop Video" button to begin real-time detection

## Structure

```
face-object-detection/
│
├── app.py                    # Main Streamlit application
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
│
├── models/                   # Store model files
│   └── efficientdet_lite0.tflite
│
└── modules/                  # Modular code organization
    ├── __init__.py           # Make modules a package
    ├── face_mesh.py          # Face mesh detection module
    ├── object_detector.py    # Object detection module 
    └── utils.py              # Utility functions
```

## Detectable Objects

The application can detect 90 common object categories from the COCO dataset, including:
- People
- Vehicles (car, bicycle, motorcycle)
- Electronics (laptop, cell phone, TV)
- Animals (dog, cat, bird)
- Furniture (chair, table, couch)
- And many more!

## Requirements

- Python 3.7+
- Webcam access
- Libraries: Streamlit, MediaPipe, OpenCV, NumPy

## License

MIT

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the detection models
- [Streamlit](https://streamlit.io/) for the web application framework
- [OpenCV](https://opencv.org/) for image processing utilities
