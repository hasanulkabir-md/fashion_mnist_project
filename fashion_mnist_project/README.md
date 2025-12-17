# Fashion MNIST Project

This project detects and classifies clothing items in images, extracts additional attributes, and provides visual annotations.

## Requirements

- TensorFlow
- OpenCV
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/fashion_mnist_project.git
    cd fashion_mnist_project
    ```

2. Install the required packages:
    ```sh
    pip install tensorflow opencv-python numpy matplotlib
    ```

## Usage

1. Ensure the object detection model and classification model are available at the specified paths:
    - `correct/path/to/object_detection_model`
    - `models/fashion_mnist_model.keras`

2. Run the `detect_and_classify.py` script:
    ```sh
    python detect_and_classify.py
    ```

3. The script will load an example image, detect and classify clothing items, and visualize the detections with bounding boxes and attribute annotations.

## Example

Replace `"path/to/example_image.jpg"` with the path to your own image in the `detect_and_classify.py` script.

```python
# Load an example image
image = cv2.imread("path/to/example_image.jpg")
```

## License

This project is licensed under the MIT License.
